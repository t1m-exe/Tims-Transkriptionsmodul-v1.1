import os
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import sys
import time
import webbrowser

# ---------------------------------
# 1. Drag-&-Drop initialisieren
# ---------------------------------
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

# ---------------------------------
# 2. Externe Bibliotheken
# ---------------------------------
import whisperx
import torch

try:
    from whisperx.diarize import DiarizationPipeline
    _DIARIZE_FROM_SUBMODULE = True
except Exception:
    DiarizationPipeline = None
    _DIARIZE_FROM_SUBMODULE = False

from huggingface_hub import HfApi
try:
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
except Exception:
    try:
        from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
    except Exception:
        class HfHubHTTPError(Exception): pass
        class RepositoryNotFoundError(Exception): pass

from fpdf import FPDF
from docx import Document

# ---------------------------------
# 3. Sprachen-Definition
# ---------------------------------
LANGUAGES_RAW = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish", "ru": "russian", 
    "ko": "korean", "fr": "french", "ja": "japanese", "pt": "portuguese", "tr": "turkish", 
    "pl": "polish", "ca": "catalan", "nl": "dutch", "ar": "arabic", "sv": "swedish", 
    "it": "italian", "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese", 
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay", "cs": "czech", 
    "ro": "romanian", "da": "danish", "hu": "hungarian", "ta": "tamil", "no": "norwegian", 
    "th": "thai", "ur": "urdu", "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", 
    "la": "latin", "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak", 
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali", "sr": "serbian", 
    "az": "azerbaijani", "sl": "slovenian", "kn": "kannada", "et": "estonian", 
    "mk": "macedonian", "br": "breton", "eu": "basque", "is": "icelandic", "hy": "armenian", 
    "ne": "nepali", "mn": "mongolian", "bs": "bosnian", "kk": "kazakh", "sq": "albanian", 
    "sw": "swahili", "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala", 
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali", "af": "afrikaans", 
    "oc": "occitan", "ka": "georgian", "be": "belarusian", "tg": "tajik", "sd": "sindhi", 
    "gu": "gujarati", "am": "amharic", "yi": "yiddish", "lo": "lao", "uz": "uzbek", 
    "fo": "faroese", "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk", 
    "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar", "bo": "tibetan", 
    "tl": "tagalog", "mg": "malagasy", "as": "assamese", "tt": "tatar", "haw": "hawaiian", 
    "ln": "lingala", "ha": "hausa", "ba": "bashkir", "jw": "javanese", "su": "sundanese", 
    "yue": "cantonese",
}

GERMAN_NAMES = {
    "de": "Deutsch", "en": "Englisch", "fr": "Französisch", "es": "Spanisch", "it": "Italienisch",
    "pt": "Portugiesisch", "nl": "Niederländisch", "pl": "Polnisch", "ru": "Russisch", "zh": "Chinesisch",
    "ja": "Japanisch", "ko": "Koreanisch", "tr": "Türkisch", "sv": "Schwedisch", "da": "Dänisch",
    "no": "Norwegisch", "fi": "Finnisch", "cs": "Tschechisch", "el": "Griechisch", "hu": "Ungarisch",
    "ro": "Rumänisch", "uk": "Ukrainisch", "ar": "Arabisch", "hi": "Hindi", "th": "Thailändisch",
    "vi": "Vietnamesisch", "id": "Indonesisch"
}

def get_display_name(code, eng_name):
    name = GERMAN_NAMES.get(code, eng_name.title()) 
    return f"{name} ({code})"

LANGUAGE_LIST = sorted(
    [get_display_name(code, name) for code, name in LANGUAGES_RAW.items()]
)

# ---------------------------------
# Utility Class: ToolTip
# ---------------------------------
class ToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.waittime = 500 
        self.wraplength = 300 
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        x = y = 0
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#ffffe0", relief='solid', borderwidth=1,
                       wraplength = self.wraplength, font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tw
        self.tw= None
        if tw: tw.destroy()

# ---------------------------------
# Konsole unterdrücken (EXE)
# ---------------------------------
if getattr(sys, "frozen", False):      
    class _SilentIO:
        def write(self, *_):  pass
        def flush(self):      pass
    sys.stdout = sys.stdout or _SilentIO()
    sys.stderr = sys.stderr or _SilentIO()

# ---------------------------------
# FFMPEG
# ---------------------------------
if getattr(sys, "frozen", False):
    base_path = sys._MEIPASS  # type: ignore
else:
    base_path = os.path.abspath(".")

FFMPEG_BIN_PATH = os.path.join(base_path, "ffmpeg", "bin")
if FFMPEG_BIN_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] += os.pathsep + FFMPEG_BIN_PATH

# ---------------------------------
# Helper
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

# ---------------------------------
# Drag & Drop
# ---------------------------------
def on_drop(event):
    global selected_file
    try:
        paths = window.tk.splitlist(event.data)
        if not paths: return
        first_path = os.path.normpath(paths[0])
        if not os.path.isfile(first_path):
            messagebox.showerror("Ungültige Datei", f"Datei nicht gefunden:\n{first_path}")
            return
        selected_file = first_path
        drop_zone.config(text=os.path.basename(first_path))
        transcribe_btn.config(state="normal")
        log(f"[INFO] Datei gesetzt: {selected_file}")
    except Exception as e:
        messagebox.showerror("Drag-&-Drop-Fehler", str(e))

def choose_file():
    global selected_file
    path = filedialog.askopenfilename(
        title="Wähle eine Datei",
        filetypes=[("Audio/Videodateien", "*.mp3 *.wav *.m4a *.mp4 *.mkv"), ("Alle Dateien", "*.*")],
    )
    if not path: return
    selected_file = path
    drop_zone.config(text=os.path.basename(path))
    transcribe_btn.config(state="normal")
    log(f"[INFO] Datei gesetzt: {selected_file}")

# ---------------------------------
# Text-Building
# ---------------------------------
def build_diarized_text(result_with_speakers: dict, with_timestamps: bool = True) -> str:
    def _fmt_ts_hhmmss(t: float) -> str:
        try:
            t = max(0.0, float(t))
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        except Exception: return "00:00:00"

    if not result_with_speakers or "segments" not in result_with_speakers:
        return ""

    speaker_map: dict[str, str] = {}
    next_idx = 1
    def map_speaker(raw: str | None) -> str:
        nonlocal next_idx
        if raw is None: raw = "UNKNOWN"
        if raw not in speaker_map:
            speaker_map[raw] = f"Sprecher {next_idx}"
            next_idx += 1
        return speaker_map[raw]

    lines: list[str] = []
    current_speaker: str | None = None
    buffer_words: list[str] = []
    chunk_start: float | None = None
    last_end: float | None = None

    used_word_level = False
    for seg in result_with_speakers.get("segments", []):
        words = seg.get("words", [])
        if words:
            used_word_level = True
            for w in words:
                spk = w.get("speaker")
                word = (w.get("word") or "").strip()
                w_start = w.get("start")
                w_end   = w.get("end")
                if not word: continue

                if spk != current_speaker:
                    if buffer_words and current_speaker is not None:
                        ts = ""
                        if with_timestamps and (chunk_start is not None) and (last_end is not None):
                            ts = f"[{_fmt_ts_hhmmss(chunk_start)}–{_fmt_ts_hhmmss(last_end)}] "
                        lines.append(f"{ts}{map_speaker(current_speaker)}: {' '.join(buffer_words).strip()}")
                        buffer_words = []
                    current_speaker = spk
                    chunk_start = w_start
                buffer_words.append(word)
                if isinstance(w_end, (int, float)):
                    last_end = w_end

    if not used_word_level:
        for seg in result_with_speakers.get("segments", []):
            spk = seg.get("speaker")
            txt = (seg.get("text") or "").strip()
            if not txt: continue
            ts = ""
            if with_timestamps and ("start" in seg) and ("end" in seg):
                ts = f"[{_fmt_ts_hhmmss(seg['start'])}–{_fmt_ts_hhmmss(seg['end'])}] "
            label = map_speaker(spk)
            lines.append(f"{ts}{label}: {txt}")

    if used_word_level and buffer_words and current_speaker is not None:
        ts = ""
        if with_timestamps and (chunk_start is not None) and (last_end is not None):
            ts = f"[{_fmt_ts_hhmmss(chunk_start)}–{_fmt_ts_hhmmss(last_end)}] "
        lines.append(f"{ts}{map_speaker(current_speaker)}: {' '.join(buffer_words).strip()}")

    return "\n".join(lines).strip()

# ---------------------------------
# Token Logic
# ---------------------------------
class UserAbortException(Exception):
    pass

def ask_hf_token_mainthread(mode="transcribe") -> str | None:
    if mode == "check":
        new_token = simpledialog.askstring(
            "Hugging Face Token",
            "Bitte Token eingeben (wird für diese Sitzung gespeichert):",
            show='*'
        )
        if new_token:
            os.environ["HUGGINGFACE_TOKEN"] = new_token.strip()
            return new_token.strip()
        else:
            return None

    env_token = os.environ.get("HUGGINGFACE_TOKEN", "").strip()
    
    if env_token:
        preview = env_token[:5] + "..." if len(env_token) > 5 else "***"
        response = messagebox.askyesno(
            "Token gefunden",
            f"Token gefunden in: System-Variablen (Arbeitsspeicher)\n"
            f"Vorschau: {preview}\n\n"
            "Eine erneute Eingabe ist nicht notwendig.\n"
            "Möchten Sie mit diesem Token fortfahren?",
            icon='info'
        )
        if response:
            return env_token

    new_token = simpledialog.askstring(
        "Hugging Face Token",
        "Bitte Token eingeben (wird für diese Sitzung gespeichert):",
        show='*'
    )

    if not new_token:
        go_without = messagebox.askyesno(
            "Kein Token", 
            "Es wurde kein Token übergeben. Möchten Sie ohne automatische Sprechererkennung fortfahren?",
            icon='warning'
        )
        if go_without:
            return None 
        else:
            raise UserAbortException("Benutzer hat abgebrochen (Kein Token).")

    if new_token:
        os.environ["HUGGINGFACE_TOKEN"] = new_token.strip()

    return new_token.strip()

def extract_text_from_asr(asr_result: dict) -> str:
    txt = (asr_result.get("text") or "").strip()
    if txt: return txt
    parts = []
    for seg in asr_result.get("segments", []):
        s = (seg.get("text") or "").strip()
        if s: parts.append(s)
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
    except Exception: pass
    return maybe_annotation

def ensure_pyannote_access(token: str) -> tuple[bool, str]:
    if not token:
        return False, "Kein Token übergeben."
    api = HfApi()
    try:
        user = api.whoami(token=token)
    except Exception:
        return False, "Token ist ungültig (Format falsch oder Account existiert nicht)."

    try:
        api.model_info("pyannote/speaker-diarization-3.1", token=token)
        api.model_info("pyannote/segmentation-3.0", token=token)
        return True, f"Zugriff OK (User: {user.get('name', 'Unbekannt')})"
    except (HfHubHTTPError, RepositoryNotFoundError):
        return False, "Token gültig, aber Bedingungen auf Hugging Face nicht akzeptiert."
    except Exception as e:
        return False, f"Fehler bei Prüfung: {e}"

# ---------------------------------
# Hardware-Check
# ---------------------------------
def pick_runtime() -> tuple[str, str, dict]:
    info = {"device_name": "Unbekannt", "cc": None, "vram_gb": None, "note": ""}
    
    # 1. Version prüfen
    torch_version = torch.__version__
    
    # 2. CUDA Check (NVIDIA)
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024**3), 1)
            
            info.update({"device_name": name, "cc": f"{major}.{minor}", "vram_gb": vram_gb})
            
            
            if major >= 6:
                info["note"] = f"GPU aktiv (PyTorch {torch_version}). Pascal+ erkannt."
                return "cuda", "float16", info
            
            info["note"] = f"GPU '{name}' zu alt (< 6.0), nutze CPU."
            return "cpu", "int8", info

        except Exception as e:
            info["note"] = f"GPU-Fehler ({e}) -> CPU."
            return "cpu", "int8", info

    # 3. Apple Silicon Check
    try:
        if torch.backends.mps.is_available():
            info["device_name"] = "Apple Silicon (M-Chip)"
            info["note"] = "Mac erkannt. Nutze CPU (float32) für maximale Stabilität."
            return "cpu", "float32", info
    except Exception:
        pass

    # 4. Standard CPU Fallback
    if "cpu" in torch_version:
        info["note"] = f"PyTorch-Version ist '{torch_version}' (CPU-Only). Bitte 'cuda'-Version installieren für NVIDIA!"
    else:
        info["note"] = f"Keine GPU gefunden. Nutze CPU."
        
    return "cpu", "int8", info


def safe_load_asr_model(model_key: str, device: str, compute_type: str):
    log(f"[INFO] Lade Modell – Wunsch: {device} ({compute_type})")
    try:
        model = whisperx.load_model(model_key, device=device, compute_type=compute_type)
        return model, device, compute_type
    except Exception as e:
        log(f"[WARN] Laden fehlgeschlagen: {e}")
    if device == "cuda":
        try:
            log("[INFO] GPU-Fallback: Versuche float32 …")
            model = whisperx.load_model(model_key, device="cuda", compute_type="float32")
            return model, "cuda", "float32"
        except Exception as e2:
            log(f"[WARN] GPU (float32) ging nicht: {e2}")
    log("[INFO] Letzter Fallback: CPU float32 …")
    model = whisperx.load_model(model_key, device="cpu", compute_type="float32")
    return model, "cpu", "float32"


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
    if not path: return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf": save_as_pdf(path, text)
    else:
        if ext not in (".docx", ".pdf"): path = path + ".docx"
        save_as_docx(path, text)
    return path

# ---------------------------------
# Main Logic
# ---------------------------------
def transcribe():
    if selected_file is None:
        messagebox.showwarning("Keine Datei", "Bitte zuerst eine Datei wählen oder ziehen.")
        return
    hf_token_main = ""
    run_diarization = False
    if diarize_var.get():
        try:
            result = ask_hf_token_mainthread(mode="transcribe")
            if result is None:
                run_diarization = False
                log("[INFO] User hat 'Ohne Sprechererkennung' gewählt.")
            else:
                hf_token_main = result
                run_diarization = True
        except UserAbortException:
            log("[ABBRUCH] Vorgang vom Benutzer abgebrochen.")
            return
    else:
        run_diarization = False

    def task(hf_token: str, do_diarize: bool):
        try:
            log(f"[DEBUG] Transkribiere: {selected_file}")
            device, compute_type, info = pick_runtime()
            dev_name = info.get("device_name", "Unbekannt")
            note = info.get("note", "")
            log(f"--------------------------------------------------")
            log(f"[HARDWARE] Grafikkarte: {dev_name}")
            log(f"[HARDWARE] Gewählter Modus: {device.upper()} ({compute_type})")
            if note: log(f"[INFO] Erklärung: {note}")
            log(f"--------------------------------------------------")

            model_key = model_var.get()
            update_progress(5, "Modell laden …")
            t0 = time.time()
            asr_model, used_device, compute_type = safe_load_asr_model(model_key, device, compute_type)
            log(f"[DEBUG] ASR-Modell in {time.time() - t0:.2f}s geladen.")

            update_progress(25, "Transkription …")
            t1 = time.time()
            transcribe_kwargs = { "batch_size": 16, "task": "transcribe" }
            if not auto_lang_var.get():
                raw_sel = lang_combobox.get()
                if "(" in raw_sel and raw_sel.endswith(")"):
                    lang_code = raw_sel.split("(")[-1].strip(")")
                    transcribe_kwargs["language"] = lang_code
                    log(f"[INFO] Manuelle Sprache: {lang_code}")
                else:
                    log("[WARN] Sprache nicht erkannt, Auto-Detect.")
            else:
                log("[INFO] Sprache wird automatisch erkannt.")

            asr_result = asr_model.transcribe(selected_file, **transcribe_kwargs)
            detected_lang = asr_result.get("language", "unbekannt")
            log(f"[INFO] Erkannte Sprache: {str(detected_lang).upper()}")
            log(f"[DEBUG] Transkription fertig in {time.time() - t1:.2f}s.")

            update_progress(45, "Ausrichten …")
            aligned_result = {"segments": asr_result.get("segments", []), "text": asr_result.get("text", "")}
            try:
                align_model, metadata = whisperx.load_align_model(language_code=asr_result["language"], device=used_device)
                aligned_result = whisperx.align(
                    asr_result["segments"], align_model, metadata, selected_file, used_device, return_char_alignments=False
                )
            except Exception as e:
                log(f"[WARN] Alignment fehlgeschlagen, nutze rohe Segmente: {e}")

            diarization_ok = False
            diarized_result = aligned_result
            if do_diarize:
                update_progress(65, "Sprechererkennung …")
                try:
                    ok, msg = ensure_pyannote_access(hf_token)
                    if not ok:
                        log(f"[HINWEIS] {msg}")
                        window.after(0, lambda: messagebox.showinfo("Fehler", msg))
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
            else:
                log("[INFO] Sprechererkennung übersprungen (kein Token oder deaktiviert).")
            
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
            err_msg = str(e)
            log(f"[FEHLER] {err_msg}")
            window.after(0, lambda: messagebox.showerror("Fehler", err_msg))
        finally:
            update_progress(0, "")

    threading.Thread(target=task, args=(hf_token_main, run_diarization), daemon=True).start()

def check_token_access():
    token = ask_hf_token_mainthread(mode="check")
    if not token: 
        status_var.set("Abbruch")
        return
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
                if msg: messagebox.showinfo("Hinweis", msg)
        window.after(0, _update)
    threading.Thread(target=_run, daemon=True).start()

def open_links(event=None):
    webbrowser.open("https://huggingface.co/pyannote/segmentation-3.0")
    webbrowser.open("https://huggingface.co/pyannote/speaker-diarization-3.1")

# ---------------------------------
# GUI Setup
# ---------------------------------
if DND_AVAILABLE:
    window: 'TkinterDnD.Tk' = TkinterDnD.Tk()  # type: ignore
else:
    window = tk.Tk()

window.title("Tims Transkriptionsmodul")
window.resizable(False, False)

# --- Drop Zone ---
drop_label_txt = "Datei hierher ziehen …" if DND_AVAILABLE else "Datei auswählen"
label = tk.Label(window, text=drop_label_txt, font=("Arial", 11))
label.pack(pady=(10, 2))

zone_opts = {"relief": "groove", "width": 50, "height": 4, "bg": "#fafafa", "text": "⇩ Drag & Drop ⇩"}
drop_zone = tk.Label(window, **zone_opts)
if DND_AVAILABLE:
    drop_zone.drop_target_register(DND_FILES)
    drop_zone.dnd_bind("<<Drop>>", on_drop)
    drop_zone.pack(padx=20, pady=(0, 8))
else:
    drop_zone.pack(padx=20, pady=(0, 4))
    tk.Button(window, text="Datei auswählen …", command=choose_file).pack(pady=(0, 8))

# --- Model Selection ---
model_var = tk.StringVar(value="medium")
model_dd = ttk.Combobox(window, textvariable=model_var, state="readonly", width=40)
model_dd["values"] = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
model_dd.pack()
tk.Label(window, text="tiny | base | small | medium (empfohlen) | large", font=("Arial", 9)).pack(pady=(2, 10))

# --- Options ---
options = tk.LabelFrame(window, text="Optionen", padx=10, pady=6)
options.pack(fill="x", padx=15, pady=(0, 10))

col1 = tk.Frame(options)
col1.pack(side="left", anchor="n", padx=(0, 20))

check_frame = tk.Frame(col1)
check_frame.pack(anchor="w")

diarize_var = tk.BooleanVar(value=True)
ttk.Checkbutton(check_frame, text="Automatische Sprechererkennung", variable=diarize_var).pack(side="left")

help_lbl = tk.Label(check_frame, text="[?]", fg="blue", cursor="hand2", font=("Arial", 9, "bold"))
help_lbl.pack(side="left", padx=5)
help_lbl.bind("<Button-1>", open_links)

tooltip_text = (
    "Für die Sprechererkennung benötigen Sie einen Hugging-Face-Account.\n"
    "Sie müssen folgende Bedingungen akzeptieren:\n\n"
    "1. pyannote/segmentation-3.0\n"
    "2. pyannote/speaker-diarization-3.1\n\n"
    "(Klicken Sie auf dieses [?], um die Webseiten zu öffnen)"
)
ToolTip(help_lbl, tooltip_text)

ttk.Button(col1, text="Token prüfen", command=check_token_access).pack(anchor="w", pady=(5,0))

col2 = tk.Frame(options)
col2.pack(side="left", anchor="n")

def toggle_lang_dropdown():
    if auto_lang_var.get():
        lang_combobox.configure(state="disabled")
    else:
        lang_combobox.configure(state="readonly")

auto_lang_var = tk.BooleanVar(value=True)
ttk.Checkbutton(col2, text="Sprache automatisch erkennen", variable=auto_lang_var, command=toggle_lang_dropdown).pack(anchor="w")

lang_combobox = ttk.Combobox(col2, values=LANGUAGE_LIST, state="disabled", width=25)
lang_combobox.set("Deutsch (de)") 
lang_combobox.pack(anchor="w", pady=(5,0))

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
status_lbl = tk.Label(window, textvariable=status_var) 
status_lbl.pack(anchor="w", padx=20)

# ---------------------------------
# Kontaktdaten
# ---------------------------------
def open_linkedin(event=None):
    webbrowser.open("https://de.linkedin.com/in/tim-lagemann-a78014187")

footer_frame = tk.Frame(window)
footer_frame.pack(side="bottom", pady=(0, 10))

author_lbl = tk.Label(footer_frame, text="Autor: Tim Lagemann", fg="black", cursor="hand2", font=("Arial", 9, "bold"))
author_lbl.pack(side="left", padx=(0, 5))
author_lbl.bind("<Button-1>", open_linkedin)

icon_canvas = tk.Canvas(footer_frame, width=20, height=20, bg="white", highlightthickness=0, cursor="hand2")
icon_canvas.pack(side="left")
icon_canvas.create_rectangle(0, 0, 20, 20, fill="#0077b5", outline="")
icon_canvas.create_text(10, 10, text="in", fill="white", font=("Arial", 12, "bold"))
icon_canvas.bind("<Button-1>", open_linkedin)

if not DND_AVAILABLE:
    log("[HINWEIS] tkinterdnd2 nicht installiert – Drag & Drop deaktiviert.")

window.mainloop()
