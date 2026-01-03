import os
import threading
import subprocess
import tempfile
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import sys
import time

# ---------------------------------
# Drag-&-Drop initialisieren
# ---------------------------------
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

# ---------------------------------
# Externe Bibliotheken
# ---------------------------------
import whisperx                      # WhisperX für ASR + Diarisierung
import torch

# Diarisierungs-Pipeline (neue WhisperX-API)
try:
    from whisperx.diarize import DiarizationPipeline
    _DIARIZE_FROM_SUBMODULE = True
except Exception:
    DiarizationPipeline = None
    _DIARIZE_FROM_SUBMODULE = False

from huggingface_hub import HfApi
try:
    from huggingface_hub.utils import HfHubHTTPError
except Exception:
    try:
        from huggingface_hub.errors import HfHubHTTPError
    except Exception:
        class HfHubHTTPError(Exception):
            pass

from fpdf import FPDF
from docx import Document  # Word-Ausgabe

# ---------------------------------
# Konsole für EXE unterdrücken
# ---------------------------------
if getattr(sys, "frozen", False):     
    class _SilentIO:
        def write(self, *_):  pass
        def flush(self):      pass
    sys.stdout = sys.stdout or _SilentIO()
    sys.stderr = sys.stderr or _SilentIO()

# ---------------------------------
# FFMPEG-Pfad anpassen (falls nötig)
# ---------------------------------
if getattr(sys, "frozen", False):
    base_path = sys._MEIPASS  # type: ignore[attr-defined]
else:
    base_path = os.path.abspath(".")

FFMPEG_BIN_PATH = os.path.join(base_path, "ffmpeg", "bin")
if FFMPEG_BIN_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + FFMPEG_BIN_PATH

# ---------------------------------
# Helferfunktionen
# ---------------------------------
def log(text: str):
    log_text.configure(state="normal")
    log_text.insert(tk.END, text + "\n")
    log_text.configure(state="disabled")
    log_text.see(tk.END)

def update_progress(val: int, txt: str = ""):
    val = max(0, min(val, 100))
    progress_var.set(val)
    progress_label.config(text=f"{val}% – {txt}" if txt else f"{val}%")
    window.update_idletasks()

def fmt_secs(s: float) -> str:
    return f"{s:.2f}s"

# ---------------------------------
# Globale State
# ---------------------------------
selected_file: str | None = None
runtime_override: tuple[str, str] | None = None   # (device, compute_type)

# ---------------------------------
# Drag-&-Drop-Callback
# ---------------------------------
def on_drop(event):
    global selected_file
    try:
        paths = window.tk.splitlist(event.data)  # type: ignore[attr-defined]
        if not paths:
            return
        first_path = os.path.normpath(paths[0])
        if not os.path.isfile(first_path):
            messagebox.showerror("Ungültige Datei", f"Die Datei konnte nicht gefunden werden:\n{first_path}")
            return
        selected_file = first_path
        drop_zone.config(text=os.path.basename(first_path))
        transcribe_btn.config(state="normal")
        log(f"[INFO] Datei gesetzt: {selected_file}")
    except Exception as e:
        messagebox.showerror("Drag-&-Drop-Fehler", str(e))

# ---------------------------------
# Datei per Dialog wählen (Fallback)
# ---------------------------------
def choose_file():
    global selected_file
    path = filedialog.askopenfilename(
        title="Wähle eine Datei",
        filetypes=[("Audio/Videodateien", "*.mp3 *.wav *.m4a *.mp4 *.mkv"), ("Alle Dateien", "*.*")],
    )
    if not path:
        return
    selected_file = path
    drop_zone.config(text=os.path.basename(path))
    transcribe_btn.config(state="normal")
    log(f"[INFO] Datei gesetzt: {selected_file}")

# ---------------------------------
# Hilfsfunktionen für Diarisierungsausgabe
# ---------------------------------
def build_diarized_text(result_with_speakers: dict) -> str:
    if not result_with_speakers or "segments" not in result_with_speakers:
        return ""
    speaker_map: dict[str, str] = {}
    next_idx = 1
    def map_speaker(raw: str | None) -> str:
        nonlocal next_idx
        if raw is None:
            raw = "UNKNOWN"
        if raw not in speaker_map:
            speaker_map[raw] = f"Sprecher {next_idx}"
            next_idx += 1
        return speaker_map[raw]
    lines: list[str] = []
    current_speaker: str | None = None
    buffer_words: list[str] = []
    for seg in result_with_speakers.get("segments", []):
        for w in seg.get("words", []):
            spk = w.get("speaker")
            word = (w.get("word") or "").strip()
            if not word:
                continue
            if spk != current_speaker:
                if buffer_words and current_speaker is not None:
                    lines.append(f"{map_speaker(current_speaker)}: {' '.join(buffer_words).strip()}")
                    buffer_words = []
                current_speaker = spk
            buffer_words.append(word)
    if buffer_words and current_speaker is not None:
        lines.append(f"{map_speaker(current_speaker)}: {' '.join(buffer_words).strip()}")
    if not lines:
        for seg in result_with_speakers.get("segments", []):
            spk = seg.get("speaker")
            txt = seg.get("text", "").strip()
            if txt:
                lines.append(f"{map_speaker(spk)}: {txt}")
    return "\n".join(lines).strip()

# --------- Sichere Token-Abfrage ---------
def ask_hf_token_mainthread() -> str:
    token = os.environ.get("HUGGINGFACE_TOKEN", "").strip()
    if token:
        return token
    return simpledialog.askstring(
        "Hugging Face Token",
        "Für die Sprechererkennung wird ein Hugging-Face-Access-Token benötigt.\n"
        "Bitte Token eingeben (wird nur für diesen Lauf verwendet):",
        show='*'
    ) or ""

def extract_text_from_asr(asr_result: dict) -> str:
    txt = (asr_result.get("text") or "").strip()
    if txt:
        return txt
    parts = []
    for seg in asr_result.get("segments", []):
        s = (seg.get("text") or "").strip()
        if s:
            parts.append(s)
    return " ".join(parts).strip()

def _annotation_to_df(maybe_annotation):
    try:
        import pandas as pd
        if hasattr(maybe_annotation, "itertracks"):
            rows = []
            for segment, _, speaker in maybe_annotation.itertracks(yield_label=True):
                rows.append({"start": segment.start, "end": segment.end, "speaker": speaker})
            return pd.DataFrame(rows)
        if isinstance(maybe_annotation, list) and maybe_annotation and isinstance(maybe_annotation[0], dict):
            return pd.DataFrame(maybe_annotation)
    except Exception:
        pass
    return maybe_annotation

def ensure_pyannote_access(token: str) -> tuple[bool, str]:
    if not token:
        return False, "Kein Token übergeben."
    try:
        api = HfApi()
        api.model_info("pyannote/speaker-diarization-3.1", token=token)
        api.model_info("pyannote/segmentation-3.0", token=token)
        return True, ""
    except HfHubHTTPError:
        return False, (
            "Kein Zugriff auf die pyannote-Pipelines.\n"
            "Bitte auf Hugging Face die Bedingungen der Repos akzeptieren "
            "und mit demselben Konto einen READ-Token verwenden."
        )
    except Exception as e:
        return False, f"Zugriffsprüfung fehlgeschlagen: {e}"

# ---------------------------------
# Hardware-Check & Auto-Wahl
# ---------------------------------
def pick_runtime() -> tuple[str, str, dict]:
    info = {"device_name": "CPU", "cc": None, "vram_gb": None, "note": ""}
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024**3), 1)
            info.update({"device_name": name, "cc": f"{major}.{minor}", "vram_gb": vram_gb})
            if major >= 7 or (major == 6 and minor == 0):
                info["note"] = "GPU mit schnellem FP16 erkannt"
                return "cuda", "float16", info
            info["note"] = "Pascal 6.1 erkannt – CPU int8 bevorzugt"
            return "cpu", "int8", info
        except Exception as e:
            info["note"] = f"CUDA-Infos nicht lesbar: {e}"
            return "cpu", "int8", info
    info["note"] = "Keine CUDA-GPU – CPU int8"
    return "cpu", "int8", info

def safe_load_asr_model(model_key: str, device: str, compute_type: str):
    log(f"[INFO] Lade Modell – Wunsch: device={device}, compute_type={compute_type}")
    try:
        model = whisperx.load_model(model_key, device=device, compute_type=compute_type)
        return model, device, compute_type
    except Exception as e:
        log(f"[WARN] load_model(device={device}, compute_type={compute_type}) fehlgeschlagen: {e}")
    
    # Fallbacks...
    if device == "cuda":
        try:
            log("[INFO] GPU-Fallback: compute_type=float32 …")
            model = whisperx.load_model(model_key, device="cuda", compute_type="float32")
            return model, "cuda", "float32"
        except Exception as e2:
            log(f"[WARN] GPU (float32) ging nicht: {e2}")

    log("[INFO] Fallback auf CPU float32 …")
    model = whisperx.load_model(model_key, device="cpu", compute_type="float32")
    return model, "cpu", "float32"

# ---------------------------------
# Snippet & Benchmark
# ---------------------------------
def make_10s_snippet(src_path: str) -> tuple[str, str]:
    tmpdir = tempfile.mkdtemp(prefix="whx_bench_")
    out_wav = os.path.join(tmpdir, "snippet.wav")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", "0", "-t", "10",
        "-i", src_path,
        "-ac", "1", "-ar", "16000", "-vn",
        out_wav
    ]
    subprocess.run(cmd, check=True)
    return out_wav, tmpdir

def do_benchmark(snippet_path: str, model_key: str):
    candidates: list[tuple[str, str]] = [("cpu", "int8"), ("cuda", "float16")]
    times: dict[tuple[str, str], float] = {}

    for want_dev, want_ct in candidates:
        label = f"{want_dev.upper()} {want_ct}"
        log(f"[BENCH] Prüfe: {label} ...")
        t0 = time.perf_counter()
        model, used_dev, used_ct = safe_load_asr_model(model_key, want_dev, want_ct)
        t1 = time.perf_counter()
        _t0 = time.perf_counter()
        _ = model.transcribe(snippet_path, batch_size=16)
        _t1 = time.perf_counter()
        total = (t1 - t0) + (_t1 - _t0)
        times[(used_dev, used_ct)] = total
        log(f"[BENCH] Ergebnis {used_dev.upper()} {used_ct}: {fmt_secs(total)}s")
        try:
            del model
        except Exception:
            pass
        if used_dev == "cuda":
            try:
                import torch as _t
                _t.cuda.empty_cache()
            except Exception:
                pass

    if not times:
        raise RuntimeError("Kein Benchmark möglich.")
    best = min(times.items(), key=lambda kv: kv[1])
    (best_dev, best_ct), best_time = best
    return best_dev, best_ct, best_time, times

# ---------------------------------
# Speichern
# ---------------------------------
def save_as_pdf(path: str, text: str):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(path, dest="F")

def save_as_docx(path: str, text: str):
    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(path)

def ask_and_save(text: str):
    path = filedialog.asksaveasfilename(
        title="Transkript speichern als …",
        initialdir=os.path.expanduser("~"),
        defaultextension=".docx",
        filetypes=[("Word-Dokument", "*.docx"), ("PDF", "*.pdf")],
        initialfile=f"Transkript_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    if not path:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        save_as_pdf(path, text)
    else:
        if ext not in (".docx", ".pdf"):
            path = path + ".docx"
        save_as_docx(path, text)
    return path

# ---------------------------------
# Transkription
# ---------------------------------
def transcribe():
    if selected_file is None:
        messagebox.showwarning("Keine Datei", "Bitte zuerst eine Datei wählen oder ziehen.")
        return

    hf_token_main = ask_hf_token_mainthread() if diarize_var.get() else ""

    def task(hf_token: str = hf_token_main):
        try:
            log(f"[DEBUG] Transkribiere: {selected_file}")
            global runtime_override
            tmpdir = None
            if runtime_override is None:
                log("[INFO] Starte automatischen Benchmark...")
                try:
                    snippet, tmpdir = make_10s_snippet(selected_file)
                    best_dev, best_ct, best_time, _times = do_benchmark(snippet, model_var.get())
                    runtime_override = (best_dev, best_ct)
                    log(f"[BENCH] Gewinner: {best_dev.upper()} {best_ct} ({fmt_secs(best_time)})")
                except Exception as e:
                    log(f"[BENCH] Auto-Benchmark nicht möglich, nutze Heuristik: {e}")
                finally:
                    if tmpdir:
                        try:
                            shutil.rmtree(tmpdir, ignore_errors=True)
                        except Exception:
                            pass

            if runtime_override:
                device, compute_type = runtime_override
                info = {"note": "Benchmark-Ergebnis"}
            else:
                device, compute_type, info = pick_runtime()

            log(f"[INFO] Laufzeitwahl: {device.upper()} ({info})")

            model_key = model_var.get()
            update_progress(5, "Modell laden …")
            t0 = time.time()
            asr_model, used_device, compute_type = safe_load_asr_model(model_key, device, compute_type)
            log(f"[DEBUG] ASR-Modell in {time.time() - t0:.2f}s geladen.")

            update_progress(25, "Transkription …")
            t1 = time.time()
            asr_result = asr_model.transcribe(selected_file, batch_size=16)
            lang = asr_result.get("language", "unbekannt")
            log(f"[INFO] Erkannte Sprache: {str(lang).upper()}")
            log(f"[DEBUG] Transkription dauerte {time.time() - t1:.2f}s.")

            update_progress(45, "Ausrichten …")
            t2 = time.time()
            try:
                align_model, metadata = whisperx.load_align_model(language_code=asr_result["language"], device=used_device)
                aligned_result = whisperx.align(
                    asr_result["segments"], align_model, metadata, selected_file, used_device, return_char_alignments=False
                )
            except Exception as e:
                log(f"[WARN] Alignment nicht möglich, nutze rohe Segmente: {e}")
                aligned_result = {"segments": asr_result.get("segments", []), "text": asr_result.get("text", "")}

            diarization_ok = False
            diarized_result = aligned_result
            if diarize_var.get():
                update_progress(65, "Sprechererkennung …")
                try:
                    ok, msg = ensure_pyannote_access(hf_token)
                    if not ok:
                        log(f"[HINWEIS] {msg}")
                        window.after(0, lambda: messagebox.showinfo("Sprechererkennung nicht verfügbar", msg))
                    else:
                        if _DIARIZE_FROM_SUBMODULE and DiarizationPipeline is not None:
                            diarize_pipeline = DiarizationPipeline(use_auth_token=hf_token, device=used_device)
                        else:
                            diarize_pipeline = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=used_device)

                        diarize_segments = diarize_pipeline(selected_file)
                        diarize_segments = _annotation_to_df(diarize_segments)
                        diarized_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
                        diarization_ok = True
                        log(f"[INFO] Sprechererkennung erfolgreich.")
                except Exception as e:
                    log(f"[WARN] Diarisierung fehlgeschlagen: {e}")
            
            update_progress(80, "Text aufbereiten …")
            base_text = extract_text_from_asr(asr_result)
            output_text = build_diarized_text(diarized_result) if diarization_ok else base_text
            if not output_text.strip():
                raise RuntimeError("Leeres Transkript.")

            update_progress(92, "Datei speichern …")
            out_path = ask_and_save(output_text)
            if out_path:
                log(f"[INFO] Datei gespeichert: {out_path}")
            else:
                log("[ABBRUCH] Speichern abgebrochen.")
            update_progress(100, "Fertig")

        except Exception as e:
            log(f"[FEHLER] {e}")
            window.after(0, lambda: messagebox.showerror("Fehler", str(e)))
        finally:
            update_progress(0, "")

    threading.Thread(target=task, args=(hf_token_main,), daemon=True).start()

# ---------------------------------
# Diagnose
# ---------------------------------
def check_token_access():
    token = ask_hf_token_mainthread()
    status_var.set("Prüfe Zugriff …")
    status_lbl.configure(foreground="#666666")
    def _run():
        ok, msg = ensure_pyannote_access(token)
        def _update():
            if ok:
                status_var.set("Zugriff OK")
                status_lbl.configure(foreground="#008000")
            else:
                status_var.set("Kein Zugriff")
                status_lbl.configure(foreground="#cc0000")
                if msg:
                    messagebox.showinfo("Hinweis", msg)
        window.after(0, _update)
    threading.Thread(target=_run, daemon=True).start()

# ---------------------------------
# GUI
# ---------------------------------
if DND_AVAILABLE:
    window: 'TkinterDnD.Tk' = TkinterDnD.Tk()  # type: ignore
else:
    window = tk.Tk()

window.title("Tims Transkriptionsmodul")
window.resizable(False, False)

# --- Drop-Zone ---
drop_label_txt = "Datei hierher ziehen …" if DND_AVAILABLE else "Datei auswählen"
label = tk.Label(window, text=drop_label_txt, font=("Arial", 11))
label.pack(pady=(10, 2))

zone_opts = {"relief": "groove", "width": 50, "height": 4, "bg": "#fafafa", "text": "⇩ Drag & Drop ⇩"}
drop_zone = tk.Label(window, **zone_opts)
if DND_AVAILABLE:
    drop_zone.drop_target_register(DND_FILES)  # type: ignore
    drop_zone.dnd_bind("<<Drop>>", on_drop)  # type: ignore
    drop_zone.pack(padx=20, pady=(0, 8))
else:
    drop_zone.pack(padx=20, pady=(0, 4))
    tk.Button(window, text="Datei auswählen …", command=choose_file).pack(pady=(0, 8))

# --- Modell Auswahl ---
model_var = tk.StringVar(value="medium")
model_dd = ttk.Combobox(window, textvariable=model_var, state="readonly", width=40)
model_dd["values"] = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
model_dd.pack()

tk.Label(window, text="tiny | base | small | medium (empfohlen) | large", font=("Arial", 9)).pack(pady=(2, 10))

# --- Optionen ---
options = tk.LabelFrame(window, text="Optionen", padx=10, pady=6)
options.pack(fill="x", padx=15, pady=(0, 10))
diarize_var = tk.BooleanVar(value=True)
ttk.Checkbutton(options, text="Automatische Sprechererkennung", variable=diarize_var).pack(anchor="w")
ttk.Button(options, text="Token prüfen", command=check_token_access).pack(side="left", pady=5)

# --- Buttons ---
transcribe_btn = tk.Button(window, text="Transkription beginnen", command=transcribe, state="disabled")
transcribe_btn.pack(pady=(0, 15))

# --- Log ---
progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(window, orient="horizontal", length=400, mode="determinate", variable=progress_var)
progress_bar.pack()
progress_label = tk.Label(window, text="0%")
progress_label.pack()
log_text = tk.Text(window, width=90, height=18, state="disabled")
log_text.pack(padx=10, pady=(5, 10))
status_var = tk.StringVar(value="")
status_lbl = tk.Label(options, textvariable=status_var)
status_lbl.pack(anchor="w")

if not DND_AVAILABLE:
    log("[HINWEIS] tkinterdnd2 nicht installiert – Drag & Drop deaktiviert.")

window.mainloop()