# 🎙️ Transkriptions-Modul für Forschung & Lehre

<img width="50%" alt="image" src="https://github.com/user-attachments/assets/a2c15d5b-4f71-450c-a9c9-f9abdb791ddd" />

Dieses Softwaremodul wurde speziell für den Einsatz in **universitären Einrichtungen und für akademische Forschungszwecke** entwickelt. Es ermöglicht die automatisierte, hochpräzise Transkription und Sprechererkennung (Diarization) von Audio- und Videodateien.

### 🔒 Datenschutz & Lokale Verarbeitung
Im Gegensatz zu kommerziellen Cloud-Diensten erfolgt die Datenverarbeitung **vollständig lokal** auf dem Endgerät. Es werden keinerlei Audiodaten an externe Server gesendet. Dies gewährleistet maximalen Datenschutz und eignet sich besonders für **sensible Forschungsdaten** (z. B. qualitative Interviews), die den universitären Serverraum nicht verlassen dürfen.

---

## ✨ Funktionen

* **Engine:** Basiert auf **WhisperX** (OpenAI Whisper mit Phonem-Alignment) für präzise Zeitstempel.
* **Sprechererkennung:** Automatische Unterscheidung verschiedener Sprecher (via *pyannote.audio*).
* **Hardware-Beschleunigung:**
    * ✅ **NVIDIA:** Voller CUDA-Support.
    * ✅ **Apple Silicon:** Unterstützung für Mac Chips.
    * ✅ **CPU-Fallback:** Automatische Nutzung der CPU, falls keine GPU erkannt wird.
* **Output:** Exportiert formatierte Transkripte als **Word (.docx)** oder **PDF**.
* **GUI:** Einfache Bedienung per Drag & Drop.

---

## 🚀 Installation

### Voraussetzungen
1.  **Python** (3.11x)
2.  **FFmpeg** muss auf dem System installiert und im System-PATH hinterlegt sein.
3.  **Hugging Face Token** (wird für die automatische Sprechererkennung benötigt).
    > *Hinweis: Sie müssen auf Hugging Face die Nutzungsbedingungen für `pyannote/segmentation-3.0` und `pyannote/speaker-diarization-3.1` akzeptieren.*

---

## 🤝 Mitwirkung & Kontakt
Da dieses Tool primär für den Forschungskontext entwickelt wurde, ist der Quellcode offen für Anpassungen. Feedback zur Funktionalität, Bug-Reports sowie Vorschläge zur Code-Optimierung aus der Community sind willkommen.

**Autor:** Tim Lagemann  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profil-blue?style=flat&logo=linkedin)](https://de.linkedin.com/in/tim-lagemann-a78014187)
