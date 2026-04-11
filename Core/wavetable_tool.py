"""
Wavetable Analyzer & Converter
Analyzes, visualizes and exports WAV wavetable files.
Dependencies: numpy (pip install numpy  /  uv run --with numpy wavetable_tool.py)
Everything else (tkinter, wave, struct) is part of the Python standard library.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import wave
import struct
import io
import os
import numpy as np


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
CYCLE_SIZES  = [256, 512, 1024, 2048]
EXPORT_SIZES = [256, 512, 1024, 2048]

COLORS = {
    "bg":        "#1a1a2e",
    "panel":     "#16213e",
    "accent":    "#0f3460",
    "highlight": "#e94560",
    "text":      "#eaeaea",
    "muted":     "#7a7a9a",
    "wave":      "#4fc3f7",
    "fft":       "#81c784",
    "grid":      "#2a2a4a",
}
LABEL_COLORS = {
    "sin":      "#4fc3f7",
    "square":   "#81c784",
    "saw":      "#ffb74d",
    "triangle": "#ce93d8",
    "complex":  "#7a7a9a",
}

# CLM chunk layout (38 bytes total, Serum / Deluge compatible):
#   [0:4]  chunk id    b'clm '
#   [4:8]  data size   little-endian uint32 = 30
#   [8:38] payload     b'<!>NNNN' + spaces to reach 30 bytes
CLM_PAYLOAD_SIZE = 30


# ---------------------------------------------------------------------------
#  CLM chunk helpers
# ---------------------------------------------------------------------------
def build_clm_chunk(cycle_size: int) -> bytes:
    """Return the 38-byte 'clm ' chunk for the given cycle size."""
    marker  = f"<!>{cycle_size}".encode("ascii")
    payload = marker + b" " * (CLM_PAYLOAD_SIZE - len(marker))
    return b"clm " + struct.pack("<I", CLM_PAYLOAD_SIZE) + payload


def write_wav_with_clm(path: str, audio_f32: np.ndarray,
                       sr: int, cycle_size: int, sampwidth: int = 2) -> None:
    """
    Write a mono WAV file that includes a 'clm ' wavetable chunk.

    Strategy: build a plain WAV in memory with the wave module, then splice
    the clm chunk between the fmt and data chunks before writing to disk.
    """
    # Build plain WAV in memory
    buf    = io.BytesIO()
    dtype  = {1: np.int8, 2: np.int16, 4: np.int32}[sampwidth]
    maxval = np.iinfo(dtype).max
    pcm    = (np.clip(audio_f32, -1.0, 1.0) * maxval).astype(dtype)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(sampwidth)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    raw = buf.getvalue()

    # Locate the 'data' chunk and splice clm before it
    data_offset = raw.find(b"data")
    if data_offset == -1:
        # Unexpected structure — fall back to plain WAV
        with open(path, "wb") as f:
            f.write(raw)
        return

    clm     = build_clm_chunk(cycle_size)
    new_raw = raw[:data_offset] + clm + raw[data_offset:]

    # Fix the RIFF size field (bytes 4-7)
    new_raw = new_raw[:4] + struct.pack("<I", len(new_raw) - 8) + new_raw[8:]

    with open(path, "wb") as f:
        f.write(new_raw)


def write_wav_plain(path: str, audio_f32: np.ndarray,
                    sr: int, sampwidth: int = 2) -> None:
    """Write a standard mono WAV file without any wavetable metadata."""
    dtype  = {1: np.int8, 2: np.int16, 4: np.int32}[sampwidth]
    maxval = np.iinfo(dtype).max
    pcm    = (np.clip(audio_f32, -1.0, 1.0) * maxval).astype(dtype)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(sampwidth)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_clm_cycle_size(path: str) -> int | None:
    """
    Read the cycle size from a 'clm ' chunk in a WAV file.
    Returns the integer cycle size, or None if no valid clm chunk is found.
    """
    with open(path, "rb") as f:
        raw = f.read()
    offset = raw.find(b"clm ")
    if offset == -1:
        return None
    payload = raw[offset + 8: offset + 8 + CLM_PAYLOAD_SIZE]
    text    = payload.decode("ascii", errors="ignore").strip()
    if text.startswith("<!>"):
        try:
            return int(text[3:].strip())
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
#  Audio helpers
# ---------------------------------------------------------------------------
def read_wav(path: str) -> tuple:
    """Read a WAV file and return (float32 mono audio, sample_rate)."""
    with wave.open(path, "rb") as w:
        sr  = w.getframerate()
        ch  = w.getnchannels()
        sw  = w.getsampwidth()
        nf  = w.getnframes()
        raw = w.readframes(nf)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch == 2:
        audio = audio[::2]          # keep left channel only
    audio /= np.iinfo(dtype).max
    return audio, sr


def resample_cycle(cycle: np.ndarray, target: int) -> np.ndarray:
    """Linearly resample a single waveform cycle to target number of samples."""
    if len(cycle) == target:
        return cycle
    x_old = np.linspace(0, 1, len(cycle), endpoint=False)
    x_new = np.linspace(0, 1, target,     endpoint=False)
    return np.interp(x_new, x_old, cycle)


def detect_cycle_size(audio: np.ndarray) -> tuple:
    """
    Estimate the most likely cycle size by measuring cosine similarity
    between successive candidate cycles.
    Returns (best_size: int, scores: dict[int, float]).
    """
    scores = {}
    for cs in CYCLE_SIZES:
        n = len(audio) // cs
        if n < 2:
            continue
        cycles = [audio[i * cs:(i + 1) * cs] for i in range(min(n, 4))]
        sims   = []
        for i in range(len(cycles) - 1):
            a, b = cycles[i], cycles[i + 1]
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if norm > 0:
                sims.append(abs(float(np.dot(a, b)) / norm))
        scores[cs] = float(np.mean(sims)) if sims else 0.0
    if not scores:
        return 2048, {}
    return max(scores, key=scores.get), scores


def classify_cycle(cycle: np.ndarray) -> tuple:
    """
    Classify a waveform cycle as sin / square / saw / triangle / complex
    via FFT harmonic analysis.
    Returns (label: str, normalized_fft[:16]: np.ndarray).
    """
    fft = np.abs(np.fft.rfft(cycle))
    if fft.max() == 0:
        return "complex", fft[:16]
    fft_norm = fft / fft.max()
    fund  = fft[1] if len(fft) > 1 else 1.0
    total = float(sum(fft[1:10])) if len(fft) > 10 else 1.0
    odds  = float(sum(fft[k] for k in range(1, 10, 2) if k < len(fft)))
    evens = float(sum(fft[k] for k in range(2, 10, 2) if k < len(fft)))
    h3    = fft[3] if len(fft) > 3 else 0.0
    h5    = fft[5] if len(fft) > 5 else 0.0
    odd_ratio = odds / total if total > 0 else 0.0
    if odd_ratio > 0.7:
        if fund > 0 and h3 / fund < 0.15:
            label = "sin"
        elif h3 > 0 and h5 / h3 < 0.25:
            label = "triangle"
        else:
            label = "square"
    elif evens / (odds + evens + 1e-9) > 0.25:
        label = "saw"
    else:
        label = "complex"
    return label, fft_norm[:16]


# ---------------------------------------------------------------------------
#  Main application
# ---------------------------------------------------------------------------
class WavetableTool(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wavetable Analyzer & Converter")
        self.configure(bg=COLORS["bg"])
        self.geometry("1040x780")
        self.minsize(820, 620)

        # Internal state
        self.audio       = None
        self.sr          = 44100
        self.cycle_size  = tk.IntVar(value=2048)
        self.current_idx = 0
        self.cycles      = []
        self.filepath    = None
        self.mode        = tk.StringVar(value="wavetable")

        # Export options
        self.export_size     = tk.IntVar(value=2048)
        self.export_n_cycles = tk.IntVar(value=0)   # 0 means "all"
        self.export_clm      = tk.BooleanVar(value=True)

        self._build_ui()

    # -----------------------------------------------------------------------
    #  UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self):
        # Header bar
        hdr = tk.Frame(self, bg=COLORS["bg"], pady=8)
        hdr.pack(fill="x", padx=16)
        tk.Label(hdr, text="WAVETABLE ANALYZER", font=("Consolas", 14, "bold"),
                 bg=COLORS["bg"], fg=COLORS["highlight"]).pack(side="left")
        tk.Label(hdr, text="& CONVERTER", font=("Consolas", 14),
                 bg=COLORS["bg"], fg=COLORS["muted"]).pack(side="left", padx=(4, 0))

        # Mode selector
        mode_bar = tk.Frame(self, bg=COLORS["panel"], pady=6)
        mode_bar.pack(fill="x", padx=16, pady=(0, 8))
        for label, val in [("  Wavetable (single file)  ", "wavetable"),
                           ("  Batch (folder → wavetable)  ", "batch")]:
            tk.Radiobutton(mode_bar, text=label, variable=self.mode, value=val,
                           command=self._on_mode_change,
                           bg=COLORS["panel"], fg=COLORS["text"],
                           selectcolor=COLORS["accent"],
                           activebackground=COLORS["panel"],
                           font=("Consolas", 10), indicatoron=False,
                           relief="flat", padx=10, pady=4,
                           bd=0, highlightthickness=0).pack(side="left", padx=4)

        # Main body
        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        self.left = tk.Frame(body, bg=COLORS["panel"], width=268)
        self.left.pack(side="left", fill="y", padx=(0, 8))
        self.left.pack_propagate(False)
        self._build_left_panel()

        self.right = tk.Frame(body, bg=COLORS["bg"])
        self.right.pack(side="left", fill="both", expand=True)
        self._build_right_panel()

        # Status bar
        self.status_var = tk.StringVar(value="Load a WAV file to get started.")
        tk.Label(self, textvariable=self.status_var,
                 font=("Consolas", 9), bg=COLORS["accent"], fg=COLORS["text"],
                 anchor="w", padx=10).pack(fill="x", side="bottom")

    def _build_left_panel(self):
        # FILE
        self._section("FILE")
        self.file_label = tk.Label(self.left, text="No file loaded",
                                   font=("Consolas", 9), bg=COLORS["panel"],
                                   fg=COLORS["text"], wraplength=240, justify="left")
        self.file_label.pack(anchor="w", padx=12, pady=(0, 4))
        self._btn("Open WAV...", self._open_file).pack(fill="x", padx=12, pady=2)
        self._sep()

        # ANALYSIS CYCLE SIZE
        self._section("ANALYSIS CYCLE SIZE")
        self.detect_label = tk.Label(self.left, text="—", font=("Consolas", 9),
                                     bg=COLORS["panel"], fg=COLORS["wave"])
        self.detect_label.pack(anchor="w", padx=12, pady=(0, 4))
        for cs in CYCLE_SIZES:
            tk.Radiobutton(self.left, text=f"{cs} samples",
                           variable=self.cycle_size, value=cs,
                           command=self._on_cycle_size_change,
                           bg=COLORS["panel"], fg=COLORS["text"],
                           selectcolor=COLORS["accent"],
                           activebackground=COLORS["panel"],
                           font=("Consolas", 10)).pack(anchor="w", padx=12)
        self._sep()

        # FILE INFO
        self._section("FILE INFO")
        self.info_label = tk.Label(self.left, text="—", font=("Consolas", 9),
                                   bg=COLORS["panel"], fg=COLORS["text"],
                                   justify="left", wraplength=240)
        self.info_label.pack(anchor="w", padx=12)
        self._sep()

        # EXPORT — output cycle size
        self._section("EXPORT")
        tk.Label(self.left, text="Output cycle size:",
                 font=("Consolas", 9), bg=COLORS["panel"],
                 fg=COLORS["text"]).pack(anchor="w", padx=12)
        ttk.Combobox(self.left, textvariable=self.export_size,
                     values=EXPORT_SIZES, state="readonly", width=10,
                     font=("Consolas", 10)).pack(anchor="w", padx=12, pady=(2, 8))

        # EXPORT — number of cycles
        tk.Label(self.left, text="Number of cycles to export:",
                 font=("Consolas", 9), bg=COLORS["panel"],
                 fg=COLORS["text"]).pack(anchor="w", padx=12)
        tk.Label(self.left, text="(0 = export all cycles)",
                 font=("Consolas", 8), bg=COLORS["panel"],
                 fg=COLORS["muted"]).pack(anchor="w", padx=12)
        n_row = tk.Frame(self.left, bg=COLORS["panel"])
        n_row.pack(anchor="w", padx=12, pady=(4, 8))
        self._btn("−", self._dec_n, small=True).pack(side="left")
        tk.Label(n_row, textvariable=self.export_n_cycles, width=4,
                 font=("Consolas", 12, "bold"),
                 bg=COLORS["panel"], fg=COLORS["text"]).pack(side="left", padx=8)
        self._btn("+", self._inc_n, small=True).pack(side="left")
        self._sep()

        # WAVETABLE HEADER (clm chunk)
        self._section("WAVETABLE HEADER")
        tk.Checkbutton(self.left, text="Write 'clm' chunk",
                       variable=self.export_clm,
                       command=self._on_clm_toggle,
                       bg=COLORS["panel"], fg=COLORS["text"],
                       selectcolor=COLORS["accent"],
                       activebackground=COLORS["panel"],
                       font=("Consolas", 10)).pack(anchor="w", padx=12)
        self.clm_desc = tk.Label(self.left, text=self._clm_text(),
                                 font=("Consolas", 8), bg=COLORS["panel"],
                                 fg=COLORS["muted"], wraplength=240, justify="left")
        self.clm_desc.pack(anchor="w", padx=12, pady=(2, 8))
        self._sep()

        # EXPORT buttons
        self._btn("Export separate WAVs", self._export_separate).pack(fill="x", padx=12, pady=2)
        self._btn("Export unified WAV",   self._export_unified).pack(fill="x", padx=12, pady=2)

    def _build_right_panel(self):
        # Navigation row
        nav = tk.Frame(self.right, bg=COLORS["bg"])
        nav.pack(fill="x", pady=(0, 6))
        self._btn("◀ Prev", self._prev_cycle, small=True).pack(side="left")
        self.nav_label = tk.Label(nav, text="— / —", font=("Consolas", 11, "bold"),
                                  bg=COLORS["bg"], fg=COLORS["text"], padx=16)
        self.nav_label.pack(side="left")
        self._btn("Next ▶", self._next_cycle, small=True).pack(side="left")
        self.cycle_badge = tk.Label(nav, text="", font=("Consolas", 10, "bold"),
                                    bg=COLORS["bg"], fg=COLORS["highlight"], padx=8)
        self.cycle_badge.pack(side="left")

        # Oscilloscope + FFT
        canvases = tk.Frame(self.right, bg=COLORS["bg"])
        canvases.pack(fill="both", expand=True)
        canvases.columnconfigure(0, weight=3)
        canvases.columnconfigure(1, weight=2)
        canvases.rowconfigure(0, weight=1)

        wf = tk.Frame(canvases, bg=COLORS["panel"])
        wf.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(wf, text="OSCILLOSCOPE", font=("Consolas", 8),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=8, pady=(6, 0))
        self.wave_canvas = tk.Canvas(wf, bg=COLORS["panel"], highlightthickness=0)
        self.wave_canvas.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.wave_canvas.bind("<Configure>", lambda e: self._draw_wave())

        ff = tk.Frame(canvases, bg=COLORS["panel"])
        ff.grid(row=0, column=1, sticky="nsew")
        tk.Label(ff, text="FFT SPECTRUM", font=("Consolas", 8),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=8, pady=(6, 0))
        self.fft_canvas = tk.Canvas(ff, bg=COLORS["panel"], highlightthickness=0)
        self.fft_canvas.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.fft_canvas.bind("<Configure>", lambda e: self._draw_fft())

        # Thumbnail strip
        tk.Label(self.right, text="ALL CYCLES", font=("Consolas", 8),
                 bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w", pady=(6, 2))
        thumb_outer = tk.Frame(self.right, bg=COLORS["bg"], height=72)
        thumb_outer.pack(fill="x")
        thumb_outer.pack_propagate(False)
        self.thumb_scroll = tk.Canvas(thumb_outer, bg=COLORS["bg"],
                                      highlightthickness=0, height=72)
        sb = ttk.Scrollbar(thumb_outer, orient="horizontal",
                           command=self.thumb_scroll.xview)
        self.thumb_scroll.configure(xscrollcommand=sb.set)
        sb.pack(side="bottom", fill="x")
        self.thumb_scroll.pack(fill="both", expand=True)
        self.thumb_frame = tk.Frame(self.thumb_scroll, bg=COLORS["bg"])
        self.thumb_scroll.create_window((0, 0), window=self.thumb_frame, anchor="nw")
        self.thumb_frame.bind("<Configure>",
                              lambda e: self.thumb_scroll.configure(
                                  scrollregion=self.thumb_scroll.bbox("all")))

    # -----------------------------------------------------------------------
    #  UI helpers
    # -----------------------------------------------------------------------
    def _btn(self, text, cmd, small=False):
        return tk.Button(self, text=text, command=cmd,
                         font=("Consolas", 9 if small else 10),
                         bg=COLORS["accent"], fg=COLORS["text"],
                         activebackground=COLORS["highlight"],
                         activeforeground="#ffffff",
                         relief="flat", bd=0, padx=8, pady=4, cursor="hand2")

    def _section(self, text):
        tk.Label(self.left, text=text, font=("Consolas", 9, "bold"),
                 bg=COLORS["panel"], fg=COLORS["muted"]).pack(anchor="w", padx=12, pady=(10, 2))

    def _sep(self):
        tk.Frame(self.left, bg=COLORS["grid"], height=1).pack(fill="x", padx=12, pady=4)

    def _clm_text(self) -> str:
        if self.export_clm.get():
            return (f"'clm' chunk will be written\n"
                    f"cycle size = {self.export_size.get()} samples\n"
                    f"Compatible: Deluge · Serum · Vital")
        return "Plain WAV — no wavetable header.\nSynths will use their own\nfallback cycle size."

    # -----------------------------------------------------------------------
    #  Event handlers
    # -----------------------------------------------------------------------
    def _on_mode_change(self):
        if self.mode.get() == "batch":
            self._open_batch()

    def _on_cycle_size_change(self):
        if self.audio is not None:
            self._slice_cycles()
            self._refresh_display()

    def _on_clm_toggle(self):
        self.clm_desc.config(text=self._clm_text())

    def _inc_n(self):
        v = self.export_n_cycles.get()
        if self.cycles:
            self.export_n_cycles.set(min(v + 1, len(self.cycles)))

    def _dec_n(self):
        self.export_n_cycles.set(max(0, self.export_n_cycles.get() - 1))

    # -----------------------------------------------------------------------
    #  File loading
    # -----------------------------------------------------------------------
    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open a wavetable WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.audio, self.sr = read_wav(path)
            self.filepath = path
            self.file_label.config(text=os.path.basename(path))
            self._auto_detect(path)
            self._slice_cycles()
            self._refresh_display()
            self.status_var.set(
                f"Loaded: {os.path.basename(path)}  |  "
                f"{len(self.audio)} samples @ {self.sr} Hz")
        except Exception as e:
            messagebox.showerror("Error", f"Could not read file:\n{e}")

    def _open_batch(self):
        folder = filedialog.askdirectory(title="Select folder of waveform WAVs")
        if not folder:
            self.mode.set("wavetable")
            return
        wavs = sorted([f for f in os.listdir(folder) if f.lower().endswith(".wav")])
        if not wavs:
            messagebox.showwarning("Batch", "No WAV files found in this folder.")
            self.mode.set("wavetable")
            return
        target       = self.export_size.get()
        cycles_audio = []
        for fname in wavs:
            try:
                a, _ = read_wav(os.path.join(folder, fname))
                src  = a[:target] if len(a) >= target else a
                cycles_audio.append(resample_cycle(src, target))
            except Exception:
                pass
        if not cycles_audio:
            messagebox.showerror("Batch", "No valid WAV files could be read.")
            self.mode.set("wavetable")
            return
        self.audio    = np.concatenate(cycles_audio)
        self.filepath = os.path.join(folder, "batch_result.wav")
        self.cycle_size.set(target)
        self.file_label.config(text=f"Batch: {len(wavs)} files")
        self.export_n_cycles.set(0)
        self._slice_cycles()
        self._refresh_display()
        self.status_var.set(
            f"Batch: {len(wavs)} files → {len(cycles_audio)} cycles of {target} samples")
        self.mode.set("wavetable")

    # -----------------------------------------------------------------------
    #  Analysis
    # -----------------------------------------------------------------------
    def _auto_detect(self, path: str):
        """Prefer the clm chunk; fall back to correlation-based detection."""
        clm_cs = read_clm_cycle_size(path)
        if clm_cs and clm_cs in CYCLE_SIZES:
            self.cycle_size.set(clm_cs)
            self.detect_label.config(text=f"From 'clm' chunk: {clm_cs}")
            # Pre-fill export size to match what was found in the file
            self.export_size.set(clm_cs)
            self.clm_desc.config(text=self._clm_text())
            return
        best, scores = detect_cycle_size(self.audio)
        self.cycle_size.set(best)
        lines = []
        for cs in CYCLE_SIZES:
            s = scores.get(cs, 0.0)
            n = len(self.audio) // cs
            lines.append(f"{cs}: {n} cycles  conf={s:.2f}{'  ◀' if cs == best else ''}")
        self.detect_label.config(text=f"Auto-detected: {best}")
        tip = "\n".join(lines)
        self.detect_label.bind("<Enter>", lambda e, t=tip: self.status_var.set(t))
        self.detect_label.bind("<Leave>", lambda e: self.status_var.set(""))

    def _slice_cycles(self):
        if self.audio is None:
            return
        cs            = self.cycle_size.get()
        n             = len(self.audio) // cs
        self.cycles      = [self.audio[i * cs:(i + 1) * cs] for i in range(n)]
        self.current_idx = 0
        self.export_n_cycles.set(0)

    # -----------------------------------------------------------------------
    #  Display
    # -----------------------------------------------------------------------
    def _refresh_display(self):
        if not self.cycles:
            return
        self.nav_label.config(text=f"Cycle  {self.current_idx + 1}  /  {len(self.cycles)}")
        cycle = self.cycles[self.current_idx]
        label, _ = classify_cycle(cycle)
        self.cycle_badge.config(text=label.upper(),
                                fg=LABEL_COLORS.get(label, COLORS["muted"]))
        cs = self.cycle_size.get()
        self.info_label.config(text=(
            f"Total samples : {len(self.audio)}\n"
            f"Cycles        : {len(self.cycles)}\n"
            f"Cycle size    : {cs} samples\n"
            f"Sample rate   : {self.sr} Hz\n"
            f"Cycle duration: {cs / self.sr * 1000:.1f} ms"
        ))
        self.clm_desc.config(text=self._clm_text())
        self._draw_wave()
        self._draw_fft()
        self._build_thumbs()

    def _draw_wave(self):
        c = self.wave_canvas
        c.delete("all")
        if not self.cycles:
            return
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10 or h < 10:
            return
        for yf in [0.25, 0.5, 0.75]:
            c.create_line(0, int(h * yf), w, int(h * yf), fill=COLORS["grid"])
        c.create_line(0, h // 2, w, h // 2, fill=COLORS["muted"], dash=(4, 4))
        samples = self.cycles[self.current_idx]
        pad = 10
        pts = []
        for i, s in enumerate(samples):
            pts.extend([pad + i / (len(samples) - 1) * (w - 2 * pad),
                        h // 2 - s * (h // 2 - pad)])
        if len(pts) >= 4:
            c.create_line(*pts, fill=COLORS["wave"], width=1.5, smooth=True)

    def _draw_fft(self):
        c = self.fft_canvas
        c.delete("all")
        if not self.cycles:
            return
        w, h = c.winfo_width(), c.winfo_height()
        if w < 10 or h < 10:
            return
        _, fft = classify_cycle(self.cycles[self.current_idx])
        n      = min(len(fft), 12)
        pad    = 12
        slot_w = (w - 2 * pad) / n
        bar_w  = max(4, int(slot_w * 0.7))
        labels = ["F", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
        for i in range(n):
            bh = int(fft[i] * (h - 32))
            x  = int(pad + i * slot_w + (slot_w - bar_w) / 2)
            c.create_rectangle(x, h - 20 - bh, x + bar_w, h - 20,
                               fill=COLORS["highlight"] if i == 0 else COLORS["fft"],
                               outline="")
            c.create_text(x + bar_w // 2, h - 8, text=labels[i],
                          font=("Consolas", 8), fill=COLORS["muted"])

    def _build_thumbs(self):
        for child in self.thumb_frame.winfo_children():
            child.destroy()
        for i, cycle in enumerate(self.cycles):
            label, _ = classify_cycle(cycle)
            border    = COLORS["highlight"] if i == self.current_idx else COLORS["panel"]
            frm       = tk.Frame(self.thumb_frame, bg=border, padx=1, pady=1)
            frm.pack(side="left", padx=2)
            th = tk.Canvas(frm, width=48, height=44, bg=COLORS["panel"],
                           highlightthickness=0, cursor="hand2")
            th.pack()
            color = LABEL_COLORS.get(label, COLORS["muted"])
            pts   = []
            for j, s in enumerate(cycle):
                pts.extend([(j / (len(cycle) - 1)) * 48, 22 - s * 18])
            if len(pts) >= 4:
                th.create_line(*pts, fill=color, width=1)
            idx = i
            th.bind("<Button-1>", lambda e, idx=idx: self._goto_cycle(idx))
            tk.Label(frm, text=label[:3].upper(), font=("Consolas", 7),
                     bg=border, fg=color).pack()

    # -----------------------------------------------------------------------
    #  Navigation
    # -----------------------------------------------------------------------
    def _prev_cycle(self):
        if self.cycles:
            self.current_idx = (self.current_idx - 1) % len(self.cycles)
            self._refresh_display()

    def _next_cycle(self):
        if self.cycles:
            self.current_idx = (self.current_idx + 1) % len(self.cycles)
            self._refresh_display()

    def _goto_cycle(self, idx: int):
        self.current_idx = idx
        self._refresh_display()

    # -----------------------------------------------------------------------
    #  Export
    # -----------------------------------------------------------------------
    def _get_export_cycles(self):
        if not self.cycles:
            messagebox.showwarning("Export", "No cycles loaded.")
            return None
        target = self.export_size.get()
        n      = self.export_n_cycles.get()
        src    = self.cycles[:n] if 0 < n <= len(self.cycles) else self.cycles
        return [resample_cycle(c, target) for c in src]

    def _write(self, path: str, audio: np.ndarray, cycle_size: int):
        """Dispatch to the appropriate writer based on the clm checkbox."""
        if self.export_clm.get():
            write_wav_with_clm(path, audio, self.sr, cycle_size)
        else:
            write_wav_plain(path, audio, self.sr)

    def _export_separate(self):
        cycles = self._get_export_cycles()
        if not cycles:
            return
        folder = filedialog.askdirectory(title="Choose export folder")
        if not folder:
            return
        base   = os.path.splitext(os.path.basename(self.filepath or "wavetable"))[0]
        target = self.export_size.get()
        suffix = "_clm" if self.export_clm.get() else ""
        for i, c in enumerate(cycles):
            label, _ = classify_cycle(self.cycles[i])
            fname    = f"{base}_cycle{i + 1:02d}_{label}_{target}{suffix}.wav"
            self._write(os.path.join(folder, fname), c, target)
        msg = (f"{len(cycles)} files exported to:\n{folder}\n"
               f"CLM chunk: {'yes — Deluge/Serum/Vital compatible' if self.export_clm.get() else 'no'}")
        self.status_var.set(msg.replace("\n", "  |  "))
        messagebox.showinfo("Export complete", msg)

    def _export_unified(self):
        cycles = self._get_export_cycles()
        if not cycles:
            return
        target  = self.export_size.get()
        base    = os.path.splitext(os.path.basename(self.filepath or "wavetable"))[0]
        suffix  = "_clm" if self.export_clm.get() else ""
        default = f"{base}_{len(cycles)}cycles_{target}{suffix}.wav"
        path    = filedialog.asksaveasfilename(
            title="Save unified wavetable WAV",
            initialfile=default,
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        self._write(path, np.concatenate(cycles), target)
        clm_note = "  |  CLM chunk written — Deluge/Serum/Vital compatible" if self.export_clm.get() else ""
        msg = (f"{os.path.basename(path)}\n"
               f"{len(cycles)} cycles × {target} samples{clm_note}")
        self.status_var.set(msg.replace("\n", "  |  "))
        messagebox.showinfo("Export complete", msg)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = WavetableTool()
    app.mainloop()
