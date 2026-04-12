"""
Wavetable Analyzer & Converter  —  v5
Analyzes, visualizes and exports WAV wavetable files.

Supported WAV formats : PCM int 8 / 16 / 24 / 32-bit,  IEEE float 32-bit
Detected WT chunks    : clm  (Serum / Deluge / Vital)
                        srge (Surge XT)
                        uhWT (u-he: Hive, Zebra)

Modes:
  Open File       — load a single wavetable bank
  Open Waveforms  — pick multiple single-cycle WAVs, assemble into one bank
  Open Banks      — pick multiple wavetable banks, browse one page per bank

Layout:
  Part A — top toolbar  (mode buttons + Clear)
  Part B — left panel   (global settings, file info, export controls)
  Part C — right area   (oscilloscope, FFT, cycle thumbnails)
  Part D — status bar   (aligned with Part C, not full-width)

Dependencies : numpy
Launch       : uv run --with numpy wavetable_tool.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import struct
import io
import os
import wave
import numpy as np


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
CYCLE_SIZES  = [256, 512, 1024, 2048]
EXPORT_SIZES = [256, 512, 1024, 2048]

# Dark theme palette
C = {
    "bg":        "#1a1a2e",
    "panel":     "#16213e",
    "accent":    "#0f3460",
    "hot":       "#e94560",   # active mode highlight
    "text":      "#eaeaea",
    "muted":     "#7a7a9a",
    "wave":      "#4fc3f7",
    "fft":       "#81c784",
    "grid":      "#2a2a4a",
}

LABEL_COLORS = {
    "sin":       "#4fc3f7",
    "triangle":  "#ce93d8",
    "square":    "#81c784",
    "saw":       "#ffb74d",
    "undefined": "#e0a060",
    "complex":   "#7a7a9a",
}

CLM_PAYLOAD_SIZE = 30  # bytes in our written clm chunk payload


# ---------------------------------------------------------------------------
#  WAV reading  (PCM 8/16/24/32-bit + IEEE float 32-bit)
# ---------------------------------------------------------------------------
def _decode_pcm24(data: bytes) -> np.ndarray:
    """Convert raw 24-bit PCM bytes to float32 in [-1, 1]."""
    raw    = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
    sign   = ((raw[:, 2] & 0x80) >> 7).astype(np.uint8) * 0xff
    padded = np.column_stack([raw, sign.reshape(-1, 1)]).flatten()
    return np.frombuffer(padded.tobytes(), dtype=np.int32).astype(np.float32) / 8388608.0


def read_wav(path: str) -> tuple:
    """
    Read a WAV file and return (audio_f32, sample_rate, bit_depth, chunk_info).

    audio_f32  : mono float32 ndarray in [-1, 1]
    chunk_info : dict with keys 'clm ', 'srge', 'uhWT' (bytes or None)
    """
    with open(path, "rb") as f:
        raw = f.read()

    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV/RIFF file.")

    fmt_off   = raw.find(b"fmt ")
    if fmt_off == -1:
        raise ValueError("No 'fmt ' chunk found.")
    audio_fmt = struct.unpack("<H", raw[fmt_off +  8: fmt_off + 10])[0]
    channels  = struct.unpack("<H", raw[fmt_off + 10: fmt_off + 12])[0]
    sr        = struct.unpack("<I", raw[fmt_off + 12: fmt_off + 16])[0]
    bit_depth = struct.unpack("<H", raw[fmt_off + 22: fmt_off + 24])[0]
    sampwidth = bit_depth // 8

    if audio_fmt not in (1, 3):
        raise ValueError(
            f"Unsupported WAV format code {audio_fmt}. "
            "Only PCM integer (1) and IEEE float (3) are supported.")

    data_off  = raw.find(b"data")
    if data_off == -1:
        raise ValueError("No 'data' chunk found.")
    data_size  = struct.unpack("<I", raw[data_off + 4: data_off + 8])[0]
    data_bytes = raw[data_off + 8: data_off + 8 + data_size]

    if audio_fmt == 3:
        audio = np.frombuffer(data_bytes, dtype=np.float32).copy()
    elif bit_depth == 24:
        audio = _decode_pcm24(data_bytes)
    else:
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth)
        if dtype is None:
            raise ValueError(f"Unsupported PCM bit depth: {bit_depth}")
        audio = np.frombuffer(data_bytes, dtype=dtype).astype(np.float32)
        audio /= np.iinfo(dtype).max

    if channels == 2:
        audio = audio[::2]   # keep left channel

    # Scan for known wavetable metadata chunks
    chunk_info: dict = {"clm ": None, "srge": None, "uhWT": None}
    pos = 12
    while pos < len(raw) - 8:
        cid  = raw[pos: pos + 4]
        if len(cid) < 4:
            break
        size = struct.unpack("<I", raw[pos + 4: pos + 8])[0]
        key  = cid.decode("ascii", errors="replace")
        if key in chunk_info:
            chunk_info[key] = raw[pos + 8: pos + 8 + size]
        pos += 8 + size
        if size == 0:
            pos += 1

    return audio, sr, bit_depth, chunk_info


# ---------------------------------------------------------------------------
#  Wavetable chunk parsers
# ---------------------------------------------------------------------------
def parse_clm(payload) -> int | None:
    """Extract cycle size from a 'clm ' chunk payload (Serum/Deluge/Vital)."""
    if not payload:
        return None
    text = payload.decode("ascii", errors="ignore").strip()
    if text.startswith("<!>"):
        try:
            return int(text[3:].split()[0])
        except (ValueError, IndexError):
            pass
    return None


def parse_srge(payload) -> int | None:
    """Extract cycle size from a 'srge' chunk (Surge XT): uint32 at offset 0."""
    if payload and len(payload) >= 4:
        try:
            return struct.unpack("<I", payload[:4])[0]
        except struct.error:
            pass
    return None


def best_chunk_cycle_size(chunk_info: dict) -> tuple:
    """Return (cycle_size | None, source_label) from known WT chunks."""
    cs = parse_clm(chunk_info.get("clm "))
    if cs:
        return cs, "clm"
    cs = parse_srge(chunk_info.get("srge"))
    if cs:
        return cs, "srge"
    if chunk_info.get("uhWT"):
        return None, "uhWT"
    return None, ""


# ---------------------------------------------------------------------------
#  CLM chunk writer
# ---------------------------------------------------------------------------
def build_clm_chunk(cycle_size: int) -> bytes:
    marker  = f"<!>{cycle_size}".encode("ascii")
    payload = marker + b" " * (CLM_PAYLOAD_SIZE - len(marker))
    return b"clm " + struct.pack("<I", CLM_PAYLOAD_SIZE) + payload


def _encode_pcm(audio: np.ndarray, bit_depth: int) -> bytes:
    """Encode float32 audio to PCM bytes at the given bit depth (16/24/32)."""
    clipped = np.clip(audio, -1.0, 1.0)
    if bit_depth == 16:
        return (clipped * 32767).astype(np.int16).tobytes()
    elif bit_depth == 24:
        vals = (clipped * 8388607).astype(np.int32)
        buf  = bytearray()
        for v in vals:
            buf += struct.pack("<i", v)[:3]
        return bytes(buf)
    elif bit_depth == 32:
        return (clipped * 2147483647).astype(np.int32).tobytes()
    else:
        raise ValueError(f"Unsupported export bit depth: {bit_depth}")


def write_wav_with_clm(path: str, audio: np.ndarray, sr: int, cs: int,
                       bit_depth: int = 16):
    """Write a PCM WAV with an injected 'clm ' wavetable chunk."""
    buf      = io.BytesIO()
    sw       = bit_depth // 8
    pcm_data = _encode_pcm(audio, bit_depth)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(sw); w.setframerate(sr)
        w.writeframes(pcm_data)
    raw      = buf.getvalue()
    data_off = raw.find(b"data")
    if data_off == -1:
        with open(path, "wb") as f: f.write(raw)
        return
    clm     = build_clm_chunk(cs)
    new_raw = raw[:data_off] + clm + raw[data_off:]
    new_raw = new_raw[:4] + struct.pack("<I", len(new_raw) - 8) + new_raw[8:]
    with open(path, "wb") as f: f.write(new_raw)


def write_wav_plain(path: str, audio: np.ndarray, sr: int, bit_depth: int = 16):
    """Write a plain PCM mono WAV without wavetable metadata."""
    sw       = bit_depth // 8
    pcm_data = _encode_pcm(audio, bit_depth)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(sw); w.setframerate(sr)
        w.writeframes(pcm_data)


# ---------------------------------------------------------------------------
#  Audio analysis
# ---------------------------------------------------------------------------
def resample_cycle(cycle: np.ndarray, target: int) -> np.ndarray:
    """
    Resample a periodic waveform cycle using FFT zero-padding / truncation.

    This is the correct method for wavetable cycles:
    - Upscaling:   zero-pads the spectrum → no new harmonics, perfect reconstruction
    - Downscaling: truncates the spectrum → clean brick-wall anti-alias filter
    - Non-power-of-2 sources (e.g. 600 samples) are handled correctly.

    Falls back to linear interpolation only if the input is too short (<4 samples).
    """
    n = len(cycle)
    if n == target:
        return cycle.copy()
    if n < 4:
        return np.interp(
            np.linspace(0, 1, target,     endpoint=False),
            np.linspace(0, 1, n,          endpoint=False),
            cycle).astype(np.float32)
    spectrum = np.fft.rfft(cycle)
    if target > n:
        new_spec = np.zeros(target // 2 + 1, dtype=complex)
        new_spec[:len(spectrum)] = spectrum
    else:
        new_spec = spectrum[:target // 2 + 1]
    result = np.fft.irfft(new_spec, n=target)
    result *= target / n
    return result.astype(np.float32)


def detect_cycle_size(audio: np.ndarray) -> tuple:
    """Cosine-similarity based cycle size detection. Returns (best, scores)."""
    scores = {}
    for cs in CYCLE_SIZES:
        n = len(audio) // cs
        if n < 2:
            continue
        slices = [audio[i * cs:(i + 1) * cs] for i in range(min(n, 4))]
        sims   = []
        for i in range(len(slices) - 1):
            a, b = slices[i], slices[i + 1]
            norm = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
            if norm > 0:
                sims.append(abs(float(np.dot(a, b))) / norm)
        scores[cs] = float(np.mean(sims)) if sims else 0.0
    if not scores:
        return 2048, {}
    return max(scores, key=scores.get), scores


def boundary_discontinuity(cycle: np.ndarray) -> float:
    """
    Measure boundary discontinuity as |first - last| / peak_amplitude.
    0 = perfectly periodic.  >0.05 = audible click.  >0.20 = severe.
    """
    peak = max(float(np.max(np.abs(cycle))), 1e-6)
    return abs(float(cycle[0]) - float(cycle[-1])) / peak


def shift_phase(cycle: np.ndarray, offset_samples: int) -> np.ndarray:
    """Shift a cycle by offset_samples positions (circular)."""
    return np.roll(cycle, -int(offset_samples)).astype(np.float32)


def apply_snap(cycle: np.ndarray) -> np.ndarray:
    """
    Force periodicity by subtracting a linear ramp that bridges the
    start/end discontinuity. Fast and preserves harmonic content.
    """
    result = cycle.copy().astype(np.float32)
    diff   = float(result[-1]) - float(result[0])
    result -= np.linspace(0, diff, len(result), dtype=np.float32)
    return result


def apply_crossfade(cycle: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Blend the boundaries of the cycle with a crossfade over n_samples.
    Smooths start/end discontinuities without altering the middle.
    """
    result = cycle.copy().astype(np.float32)
    n    = len(cycle)
    n_cf = max(2, min(n_samples, n // 4))
    fade_out = np.linspace(1.0, 0.0, n_cf, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, n_cf, dtype=np.float32)
    start_blend = cycle[:n_cf] * fade_in  + cycle[-n_cf:] * fade_out
    end_blend   = cycle[-n_cf:] * fade_out + cycle[:n_cf] * fade_in
    result[:n_cf]  = start_blend
    result[-n_cf:] = end_blend
    return result


def classify_cycle(cycle: np.ndarray) -> tuple:
    """
    Classify via FFT harmonic analysis.
    Returns (label, fft_norm[:16]).

    Thresholds:
      sin       — fund dominant, H2 < 5 %, H3 < 5 %
      triangle  — odd-only (>85 %), H5/H3 < 0.45  (theoretical: (3/5)² = 0.36)
      square    — odd-only (>80 %), H5/H3 ≥ 0.45  (theoretical: 3/5 = 0.60)
      saw       — even harmonics present (>20 %)
      undefined — some harmonic content but pattern unclear
      complex   — broadband / noise-like
    """
    fft = np.abs(np.fft.rfft(cycle))
    if fft.max() == 0:
        return "complex", fft[:16]
    fft_n = fft / fft.max()
    fund  = float(fft[1]) if len(fft) > 1 else 1.0
    h2    = float(fft[2]) if len(fft) > 2 else 0.0
    h3    = float(fft[3]) if len(fft) > 3 else 0.0
    h5    = float(fft[5]) if len(fft) > 5 else 0.0
    total = float(np.sum(fft[1:10])) if len(fft) > 10 else 1.0
    odds  = float(sum(fft[k] for k in range(1, 10, 2) if k < len(fft)))
    evens = float(sum(fft[k] for k in range(2, 10, 2) if k < len(fft)))
    odd_r  = odds / total if total > 0 else 0.0
    even_r = evens / (odds + evens + 1e-9)

    if fund > 0 and h2 / fund < 0.05 and h3 / fund < 0.05:
        return "sin", fft_n[:16]
    if odd_r > 0.85 and h3 > 0 and h5 / h3 < 0.45:
        return "triangle", fft_n[:16]
    if odd_r > 0.80 and h3 > 0 and h5 / h3 >= 0.45:
        return "square", fft_n[:16]
    if even_r > 0.20 and total / fft.max() > 0.3:
        return "saw", fft_n[:16]
    if total / fft.max() > 0.15:
        return "undefined", fft_n[:16]
    return "complex", fft_n[:16]


# ---------------------------------------------------------------------------
#  Data model
# ---------------------------------------------------------------------------
class Bank:
    """One loaded wavetable file (or assembled waveform collection)."""
    def __init__(self, path, audio, sr, bit_depth, chunk_info):
        self.path       = path
        self.audio      = audio
        self.sr         = sr
        self.bit_depth  = bit_depth
        self.chunk_info = chunk_info
        self.cycle_size = 2048
        self.cycles: list[np.ndarray] = []

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def slice(self, cs: int):
        self.cycle_size = cs
        n = len(self.audio) // cs
        self.cycles = [self.audio[i * cs:(i + 1) * cs] for i in range(n)]


# ---------------------------------------------------------------------------
#  Application
# ---------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wavetable Analyzer & Converter  v5")
        self.configure(bg=C["bg"])
        self.geometry("1100x780")
        self.minsize(900, 640)

        # State
        self.banks:     list[Bank] = []
        self.bank_idx:  int        = 0
        self.cycle_idx: int        = 0
        self.mode:      str        = ""   # "file" | "waveforms" | "banks"

        # Tkinter vars (global settings)
        self.cs_var          = tk.IntVar(value=2048)
        self.export_size_var = tk.IntVar(value=2048)
        self.export_n_var    = tk.IntVar(value=0)
        self.export_clm_var  = tk.BooleanVar(value=True)
        self.export_sr_var   = tk.IntVar(value=44100)
        self.export_depth_var= tk.IntVar(value=16)
        self.phase_offset_var= tk.IntVar(value=0)

        self._build()

    # ── convenience ─────────────────────────────────────────────────────────
    @property
    def bank(self) -> Bank | None:
        return self.banks[self.bank_idx] if self.banks else None

    @property
    def cycles(self) -> list:
        return self.bank.cycles if self.bank else []

    # ── UI construction ──────────────────────────────────────────────────────
    def _build(self):
        # ── Part A — top toolbar ────────────────────────────────────────────
        self.toolbar = tk.Frame(self, bg=C["panel"], pady=6)
        self.toolbar.pack(fill="x", padx=0, pady=0)

        # Title
        tk.Label(self.toolbar, text="WAVETABLE ANALYZER",
                 font=("Consolas", 13, "bold"),
                 bg=C["panel"], fg=C["hot"]).pack(side="left", padx=(12, 4))
        tk.Label(self.toolbar, text="v5",
                 font=("Consolas", 11),
                 bg=C["panel"], fg=C["muted"]).pack(side="left", padx=(0, 16))

        # Mode buttons stored for highlight management
        self.mode_btns = {}
        for label, key in [("Open File", "file"),
                            ("Open Waveforms", "waveforms"),
                            ("Open Banks", "banks")]:
            b = tk.Button(self.toolbar, text=label,
                          font=("Consolas", 10),
                          bg=C["accent"], fg=C["text"],
                          activebackground=C["hot"],
                          activeforeground="#fff",
                          relief="flat", bd=0, padx=12, pady=5,
                          cursor="hand2",
                          command=lambda k=key: self._open_mode(k))
            b.pack(side="left", padx=4)
            self.mode_btns[key] = b

        # Clear — right side
        tk.Button(self.toolbar, text="Clear",
                  font=("Consolas", 10),
                  bg=C["hot"], fg="#fff",
                  activebackground="#c0304a",
                  activeforeground="#fff",
                  relief="flat", bd=0, padx=12, pady=5,
                  cursor="hand2",
                  command=self._clear).pack(side="right", padx=12)

        # ── Body: Part B (left) + Part C+D (right) ──────────────────────────
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True)

        # Part B — left panel
        self.panel_b = tk.Frame(body, bg=C["panel"], width=230)
        self.panel_b.pack(side="left", fill="y")
        self.panel_b.pack_propagate(False)
        self._build_panel_b()

        # Right column: Part C + Part D stacked
        right_col = tk.Frame(body, bg=C["bg"])
        right_col.pack(side="left", fill="both", expand=True)

        # Part C — visualisation
        self.panel_c = tk.Frame(right_col, bg=C["bg"])
        self.panel_c.pack(fill="both", expand=True, padx=10, pady=(8, 4))
        self._build_panel_c()

        # Part D — status bar (aligned with Part C, not full-width)
        self.status_var = tk.StringVar(value="Load a WAV file to get started.")
        tk.Label(right_col, textvariable=self.status_var,
                 font=("Consolas", 9),
                 bg=C["accent"], fg=C["text"],
                 anchor="w", padx=10).pack(fill="x", padx=10, pady=(0, 6))

    def _build_panel_b(self):
        p = self.panel_b

        # FILE INFO
        self._lbl_section(p, "FILE INFO")
        self.file_lbl = tk.Label(p, text="No file loaded",
                                 font=("Consolas", 9), bg=C["panel"], fg=C["text"],
                                 wraplength=210, justify="left")
        self.file_lbl.pack(anchor="w", padx=10, pady=(0, 4))

        # Bank navigator — only shown in Open Banks mode
        self.bank_nav_frame = tk.Frame(p, bg=C["panel"])
        self.bank_nav_frame.pack(fill="x", padx=10, pady=(0, 4))
        self._sbtn(self.bank_nav_frame, "◀", self._prev_bank).pack(side="left")
        self.bank_nav_lbl = tk.Label(
            self.bank_nav_frame, text="",
            font=("Consolas", 9, "bold"),
            bg=C["panel"], fg=C["hot"], padx=8)
        self.bank_nav_lbl.pack(side="left", in_=self.bank_nav_frame)
        self._sbtn(self.bank_nav_frame, "▶", self._next_bank).pack(side="left")
        self.bank_nav_frame.pack_forget()

        self._sep(p)

        # WT METADATA
        self._lbl_section(p, "WT METADATA")
        self.meta_lbl = tk.Label(p, text="—",
                                 font=("Consolas", 9), bg=C["panel"], fg=C["wave"],
                                 wraplength=210, justify="left")
        self.meta_lbl.pack(anchor="w", padx=10, pady=(0, 4))
        self._sep(p)

        # ANALYSIS CYCLE SIZE
        self._lbl_section(p, "ANALYSIS CYCLE SIZE")
        self.detect_lbl = tk.Label(p, text="—",
                                   font=("Consolas", 9), bg=C["panel"], fg=C["muted"])
        self.detect_lbl.pack(anchor="w", padx=10, pady=(0, 2))
        for cs in CYCLE_SIZES:
            tk.Radiobutton(p, text=f"{cs} samples",
                           variable=self.cs_var, value=cs,
                           command=self._on_cs_change,
                           bg=C["panel"], fg=C["text"],
                           selectcolor=C["accent"],
                           activebackground=C["panel"],
                           font=("Consolas", 9)).pack(anchor="w", padx=10)
        self._sep(p)

        # FILE INFO details
        self._lbl_section(p, "FILE INFO")
        self.info_lbl = tk.Label(p, text="—",
                                 font=("Consolas", 9), bg=C["panel"], fg=C["text"],
                                 justify="left", wraplength=210)
        self.info_lbl.pack(anchor="w", padx=10)
        self._sep(p)

        # EXPORT
        self._lbl_section(p, "EXPORT")

        tk.Label(p, text="Output cycle size:",
                 font=("Consolas", 9), bg=C["panel"], fg=C["text"]).pack(
                     anchor="w", padx=10)
        ttk.Combobox(p, textvariable=self.export_size_var,
                     values=EXPORT_SIZES, state="readonly", width=9,
                     font=("Consolas", 9)).pack(anchor="w", padx=10, pady=(2, 6))

        tk.Label(p, text="Sample rate (Hz):",
                 font=("Consolas", 9), bg=C["panel"], fg=C["text"]).pack(
                     anchor="w", padx=10)
        ttk.Combobox(p, textvariable=self.export_sr_var,
                     values=[44100, 48000, 88200, 96000], state="readonly", width=9,
                     font=("Consolas", 9)).pack(anchor="w", padx=10, pady=(2, 6))

        tk.Label(p, text="Bit depth:",
                 font=("Consolas", 9), bg=C["panel"], fg=C["text"]).pack(
                     anchor="w", padx=10)
        ttk.Combobox(p, textvariable=self.export_depth_var,
                     values=[16, 24, 32], state="readonly", width=9,
                     font=("Consolas", 9)).pack(anchor="w", padx=10, pady=(2, 8))

        self._sep(p)

        # WAVETABLE HEADER
        self._lbl_section(p, "WAVETABLE HEADER")
        tk.Checkbutton(p, text="Write 'clm' chunk",
                       variable=self.export_clm_var,
                       command=self._on_clm_toggle,
                       bg=C["panel"], fg=C["text"],
                       selectcolor=C["accent"],
                       activebackground=C["panel"],
                       font=("Consolas", 9)).pack(anchor="w", padx=10)
        self.clm_desc_lbl = tk.Label(p, text=self._clm_text(),
                                     font=("Consolas", 8), bg=C["panel"], fg=C["muted"],
                                     wraplength=210, justify="left")
        self.clm_desc_lbl.pack(anchor="w", padx=10, pady=(0, 6))
        self._sep(p)

        # Cycles to export: label + [−] N [+] on one row — just before export buttons
        n_row = tk.Frame(p, bg=C["panel"])
        n_row.pack(fill="x", padx=10, pady=(0, 2))
        tk.Label(n_row, text="Cycles to export:",
                 font=("Consolas", 9), bg=C["panel"],
                 fg=C["text"]).pack(side="left")
        self._sbtn(n_row, "−", self._dec_n).pack(side="left", padx=(6, 2))
        tk.Label(n_row, textvariable=self.export_n_var, width=3,
                 font=("Consolas", 10, "bold"),
                 bg=C["panel"], fg=C["hot"]).pack(side="left")
        self._sbtn(n_row, "+", self._inc_n).pack(side="left", padx=(2, 0))
        tk.Label(p, text="(0 = all)",
                 font=("Consolas", 8), bg=C["panel"],
                 fg=C["muted"]).pack(anchor="w", padx=10, pady=(0, 6))

        # Edit / create waveform
        self._sep(p)
        self._lbl_section(p, "EDIT WAVEFORM")
        self._btn(p, "Edit current cycle...", self._open_editor).pack(
            fill="x", padx=10, pady=2)
        self._btn(p, "New cycle from scratch", self._new_cycle).pack(
            fill="x", padx=10, pady=2)
        self._sep(p)

        # Export buttons
        for txt, cmd in [("Export current cycle",    self._exp_solo),
                         ("Export separate WAVs",    self._exp_separate),
                         ("Export unified WAV",      self._exp_unified)]:
            self._btn(p, txt, cmd).pack(fill="x", padx=10, pady=2)

        # "Export all banks" — only shown in Open Banks mode
        self.exp_all_btn = self._btn(p, "Export all banks", self._exp_all_banks)
        self.exp_all_btn.pack(fill="x", padx=10, pady=2)
        self.exp_all_btn.pack_forget()

    def _build_panel_c(self):
        p = self.panel_c

        # ── Oscilloscope + FFT ──
        vis_row = tk.Frame(p, bg=C["bg"])
        vis_row.pack(fill="both", expand=True)
        vis_row.columnconfigure(0, weight=3)
        vis_row.columnconfigure(1, weight=2)
        vis_row.rowconfigure(0, weight=1)

        wf = tk.Frame(vis_row, bg=C["panel"])
        wf.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        tk.Label(wf, text="OSCILLOSCOPE", font=("Consolas", 8),
                 bg=C["panel"], fg=C["muted"]).pack(anchor="w", padx=8, pady=(4, 0))
        self.wave_cv = tk.Canvas(wf, bg=C["panel"], highlightthickness=0)
        self.wave_cv.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.wave_cv.bind("<Configure>", lambda e: self._draw_wave())

        ff = tk.Frame(vis_row, bg=C["panel"])
        ff.grid(row=0, column=1, sticky="nsew")
        tk.Label(ff, text="FFT SPECTRUM", font=("Consolas", 8),
                 bg=C["panel"], fg=C["muted"]).pack(anchor="w", padx=8, pady=(4, 0))
        self.fft_cv = tk.Canvas(ff, bg=C["panel"], highlightthickness=0)
        self.fft_cv.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.fft_cv.bind("<Configure>", lambda e: self._draw_fft())

        # ── Cycle label + badge ──
        info_row = tk.Frame(p, bg=C["bg"])
        info_row.pack(fill="x", pady=(4, 0))
        self.cycle_nav_lbl = tk.Label(info_row, text="— / —",
                                      font=("Consolas", 11, "bold"),
                                      bg=C["bg"], fg=C["text"])
        self.cycle_nav_lbl.pack(side="left")
        self.cycle_badge = tk.Label(info_row, text="",
                                    font=("Consolas", 10, "bold"),
                                    bg=C["bg"], fg=C["hot"], padx=10)
        self.cycle_badge.pack(side="left")
        # Cycle navigation and actions — parented to info_row
        self._sbtn(info_row, "◀", self._prev_cycle).pack(side="left", padx=(12, 2))
        self._sbtn(info_row, "▶", self._next_cycle).pack(side="left")
        self._sbtn(info_row, "▶ Play", self._play_cycle).pack(side="left", padx=(16, 0))
        self._sbtn(info_row, "Delete", self._delete_cycle).pack(side="left", padx=(8, 0))
        # Phase offset control
        phase_row = tk.Frame(self.panel_c, bg=C["bg"])
        phase_row.pack(fill="x", pady=(2, 0))
        tk.Label(phase_row, text="Phase offset:",
                 font=("Consolas", 8), bg=C["bg"], fg=C["muted"]).pack(side="left")
        self.phase_offset_var = tk.IntVar(value=0)
        self._sbtn(phase_row, "−1", lambda: self._shift_cycle(-1)).pack(side="left", padx=(6, 1))
        self._sbtn(phase_row, "−10", lambda: self._shift_cycle(-10)).pack(side="left", padx=1)
        self._sbtn(phase_row, "−100", lambda: self._shift_cycle(-100)).pack(side="left", padx=1)
        tk.Label(phase_row, textvariable=self.phase_offset_var, width=6,
                 font=("Consolas", 9, "bold"), bg=C["bg"], fg=C["hot"]).pack(side="left", padx=4)
        self._sbtn(phase_row, "+100", lambda: self._shift_cycle(100)).pack(side="left", padx=1)
        self._sbtn(phase_row, "+10",  lambda: self._shift_cycle(10)).pack(side="left", padx=1)
        self._sbtn(phase_row, "+1",   lambda: self._shift_cycle(1)).pack(side="left", padx=1)
        self._sbtn(phase_row, "Reset", self._reset_phase).pack(side="left", padx=(6, 0))

        # ── ALL CYCLES thumbnails ──
        tk.Label(p, text="ALL CYCLES",
                 font=("Consolas", 8), bg=C["bg"], fg=C["muted"]).pack(
                     anchor="w", pady=(6, 2))
        thumb_outer = tk.Frame(p, bg=C["bg"], height=72)
        thumb_outer.pack(fill="x")
        thumb_outer.pack_propagate(False)
        self.thumb_cv = tk.Canvas(thumb_outer, bg=C["bg"],
                                  highlightthickness=0, height=72)
        sb = ttk.Scrollbar(thumb_outer, orient="horizontal",
                           command=self.thumb_cv.xview)
        self.thumb_cv.configure(xscrollcommand=sb.set)
        sb.pack(side="bottom", fill="x")
        self.thumb_cv.pack(fill="both", expand=True)
        self.thumb_frame = tk.Frame(self.thumb_cv, bg=C["bg"])
        self.thumb_cv.create_window((0, 0), window=self.thumb_frame, anchor="nw")
        self.thumb_frame.bind(
            "<Configure>",
            lambda e: self.thumb_cv.configure(
                scrollregion=self.thumb_cv.bbox("all")))

    # ── UI helpers ───────────────────────────────────────────────────────────
    def _btn(self, parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Consolas", 9),
                         bg=C["accent"], fg=C["text"],
                         activebackground=C["hot"],
                         activeforeground="#fff",
                         relief="flat", bd=0, padx=6, pady=4, cursor="hand2")

    def _sbtn(self, parent, text, cmd):
        """Small button anchored to parent frame."""
        return tk.Button(parent, text=text, command=cmd,
                         font=("Consolas", 9),
                         bg=C["accent"], fg=C["text"],
                         activebackground=C["hot"],
                         activeforeground="#fff",
                         relief="flat", bd=0, padx=6, pady=2, cursor="hand2")

    def _lbl_section(self, parent, text):
        tk.Label(parent, text=text, font=("Consolas", 8, "bold"),
                 bg=C["panel"], fg=C["muted"]).pack(
                     anchor="w", padx=10, pady=(8, 2))

    def _sep(self, parent):
        tk.Frame(parent, bg=C["grid"], height=1).pack(fill="x", padx=10, pady=3)

    def _clm_text(self) -> str:
        if self.export_clm_var.get():
            return (f"cycle={self.export_size_var.get()}\n"
                    f"Deluge · Serum · Vital")
        return "Plain WAV — no WT header."

    def _set_mode(self, mode: str):
        """Update mode and refresh toolbar button highlights."""
        self.mode = mode
        for key, btn in self.mode_btns.items():
            btn.configure(bg=C["hot"] if key == mode else C["accent"])

    # ── Event handlers ───────────────────────────────────────────────────────
    def _on_cs_change(self):
        b = self.bank
        if b:
            b.slice(self.cs_var.get())
            self.cycle_idx = 0
            self.export_n_var.set(0)
            self._refresh()

    def _on_clm_toggle(self):
        self.clm_desc_lbl.config(text=self._clm_text())

    def _inc_n(self):
        v = self.export_n_var.get()
        if self.cycles:
            self.export_n_var.set(min(v + 1, len(self.cycles)))

    def _dec_n(self):
        self.export_n_var.set(max(0, self.export_n_var.get() - 1))

    # ── Loading ──────────────────────────────────────────────────────────────
    def _load_bank(self, path: str) -> Bank:
        audio, sr, bd, ci = read_wav(path)
        b = Bank(path, audio, sr, bd, ci)
        cs, _ = best_chunk_cycle_size(ci)
        if cs and cs in CYCLE_SIZES:
            b.slice(cs)
        else:
            best, _ = detect_cycle_size(audio)
            b.slice(best)
        return b

    def _open_mode(self, mode: str):
        if mode == "file":
            self._open_file()
        elif mode == "waveforms":
            self._open_waveforms()
        elif mode == "banks":
            self._open_banks()

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open a wavetable bank",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.banks    = [self._load_bank(path)]
            self.bank_idx = 0
            self._set_mode("file")
            self._activate(0)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _open_waveforms(self):
        paths = filedialog.askopenfilenames(
            title="Select single-cycle waveform WAV files",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not paths:
            return
        target = self.export_size_var.get()
        cycles_audio, errors = [], []
        for p in sorted(paths):
            try:
                audio, _, _, _ = read_wav(p)
                src = audio[:target] if len(audio) >= target else audio
                cycles_audio.append(resample_cycle(src, target))
            except Exception as e:
                errors.append(f"{os.path.basename(p)}: {e}")
        if errors:
            messagebox.showwarning("Some files skipped", "\n".join(errors))
        if not cycles_audio:
            messagebox.showerror("Error", "No valid files.")
            return
        b = Bank(path=sorted(paths)[0],
                 audio=np.concatenate(cycles_audio),
                 sr=44100, bit_depth=16, chunk_info={})
        b.slice(target)
        self.banks    = [b]
        self.bank_idx = 0
        self._set_mode("waveforms")
        self._activate(0)
        self.file_lbl.config(text=f"{len(cycles_audio)} waveforms assembled")

    def _open_banks(self):
        paths = filedialog.askopenfilenames(
            title="Select wavetable bank WAV files",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not paths:
            return
        loaded, errors = [], []
        for p in sorted(paths):
            try:
                loaded.append(self._load_bank(p))
            except Exception as e:
                errors.append(f"{os.path.basename(p)}: {e}")
        if errors:
            messagebox.showwarning("Some files skipped", "\n".join(errors))
        if not loaded:
            messagebox.showerror("Error", "No valid files.")
            return
        self.banks    = loaded
        self.bank_idx = 0
        self._set_mode("banks")
        self._activate(0)

    def _clear(self):
        self.banks    = []
        self.bank_idx = 0
        self.cycle_idx = 0
        self.mode = ""
        for btn in self.mode_btns.values():
            btn.configure(bg=C["accent"])
        self.file_lbl.config(text="No file loaded")
        self.meta_lbl.config(text="—")
        self.detect_lbl.config(text="—")
        self.info_lbl.config(text="—")
        self.cycle_nav_lbl.config(text="— / —")
        self.cycle_badge.config(text="")
        self.bank_nav_frame.pack_forget()
        self.exp_all_btn.pack_forget()
        self.wave_cv.delete("all")
        self.fft_cv.delete("all")
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        self.status_var.set("Cleared.")

    # ── Bank activation ──────────────────────────────────────────────────────
    def _activate(self, idx: int):
        """Switch to bank[idx] and refresh everything."""
        self.bank_idx  = idx
        self.cycle_idx = 0
        b = self.bank
        if b is None:
            return

        # Sync cycle-size radio and export params to this bank
        self.cs_var.set(b.cycle_size)
        self.export_n_var.set(0)
        self.export_sr_var.set(b.sr)
        self.export_depth_var.set(b.bit_depth if b.bit_depth in (16, 24, 32) else 16)

        # Bank navigator visibility
        if self.mode == "banks" and len(self.banks) > 1:
            self.bank_nav_lbl.config(text=f"{idx + 1} / {len(self.banks)}")
            self.bank_nav_frame.pack(fill="x", padx=10, pady=(0, 4))
            self.exp_all_btn.pack(fill="x", padx=10, pady=2)
        else:
            self.bank_nav_frame.pack_forget()
            self.exp_all_btn.pack_forget()

        self._update_panel_b()
        self._refresh()
        self.status_var.set(
            f"{b.name}  |  {len(b.audio)} samples @ {b.sr} Hz  |  "
            f"{b.bit_depth}-bit  |  {len(b.cycles)} cycles × {b.cycle_size} samp")

    def _update_panel_b(self):
        b = self.bank
        if not b:
            return
        self.file_lbl.config(text=b.name)

        # Metadata
        ci    = b.chunk_info
        parts = []
        c = parse_clm(ci.get("clm "))
        if c:
            parts.append(f"clm  → {c} samp/cycle (Serum)")
        s = parse_srge(ci.get("srge"))
        if s:
            parts.append(f"srge → {s} samp/cycle (Surge)")
        if ci.get("uhWT"):
            parts.append("uhWT → present (u-he)")
        self.meta_lbl.config(
            text="\n".join(parts) if parts else "No WT chunk found")

        # Detection label
        cs_from_chunk, src = best_chunk_cycle_size(ci)
        if cs_from_chunk:
            self.detect_lbl.config(
                text=f"From '{src}' chunk: {cs_from_chunk}")
        else:
            _, scores = detect_cycle_size(b.audio)
            best = b.cycle_size
            tip_lines = []
            for sz in CYCLE_SIZES:
                n = len(b.audio) // sz
                tip_lines.append(
                    f"{sz}: {n} cyc  conf={scores.get(sz,0):.2f}"
                    f"{'  ◀' if sz == best else ''}")
            self.detect_lbl.config(text=f"Auto-detected: {best}")
            tip = "\n".join(tip_lines)
            self.detect_lbl.bind("<Enter>",
                                 lambda e, t=tip: self.status_var.set(t))
            self.detect_lbl.bind("<Leave>",
                                 lambda e: self._restore_status())

        # File info
        cs = b.cycle_size
        avg_br = b.sr * b.bit_depth // 8  # bytes/sec
        self.info_lbl.config(text=(
            f"Total  : {len(b.audio)} samples\n"
            f"Cycles : {len(b.cycles)}\n"
            f"Cycle  : {cs} samples\n"
            f"SR     : {b.sr} Hz\n"
            f"Depth  : {b.bit_depth}-bit\n"
            f"Bitrate: {avg_br // 1000} kB/s\n"
            f"Dur.   : {cs / b.sr * 1000:.1f} ms/cycle"
        ))
        self.clm_desc_lbl.config(text=self._clm_text())

    def _restore_status(self):
        b = self.bank
        if b:
            self.status_var.set(
                f"{b.name}  |  {len(b.audio)} samples @ {b.sr} Hz  |  "
                f"{b.bit_depth}-bit  |  {len(b.cycles)} cycles × {b.cycle_size} samp")

    # ── Navigation ───────────────────────────────────────────────────────────
    def _play_cycle(self):
        """
        Play the currently displayed cycle as a looped tone for ~1 second.
        Uses winsound on Windows (built-in, no extra deps).
        Falls back to a beep if unavailable.
        """
        if not self.cycles:
            return
        import tempfile, threading
        cycle  = self.cycles[self.cycle_idx]
        sr     = self.bank.sr if self.bank else 44100
        target = self.export_size_var.get()
        c      = resample_cycle(cycle, target)
        # Build 1-second audio by tiling the cycle
        n_rep  = max(1, sr // len(c))
        audio  = np.tile(c, n_rep + 1)[:sr].astype(np.float32)
        # Write temp WAV
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        write_wav_plain(tmp_path, audio, sr)
        def _play():
            try:
                import winsound
                winsound.PlaySound(tmp_path,
                                   winsound.SND_FILENAME | winsound.SND_NODEFAULT)
            except Exception:
                pass
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        threading.Thread(target=_play, daemon=True).start()


    def _delete_cycle(self):
        """Delete the currently displayed cycle from the active bank."""
        b = self.bank
        if not b or len(b.cycles) == 0:
            return
        if len(b.cycles) == 1:
            if not messagebox.askyesno("Delete", "This is the last cycle. Delete it?"):
                return
        b.cycles.pop(self.cycle_idx)
        b.audio = np.concatenate(b.cycles) if b.cycles else np.zeros(b.cycle_size, dtype=np.float32)
        self.cycle_idx = min(self.cycle_idx, max(0, len(b.cycles) - 1))
        self._update_panel_b()
        self._refresh()
        self.status_var.set(f"Cycle deleted. {len(b.cycles)} cycles remaining.")

    def _shift_cycle(self, delta: int):
        """Shift the current cycle by delta samples (circular)."""
        b = self.bank
        if not b or not b.cycles:
            return
        cur = self.phase_offset_var.get() + delta
        b.cycles[self.cycle_idx] = shift_phase(b.cycles[self.cycle_idx], delta)
        # Keep offset display clamped to [-cs/2, cs/2] for readability
        cs = b.cycle_size
        self.phase_offset_var.set(cur % cs if cur >= 0 else -((-cur) % cs))
        self._draw_wave()
        self._draw_fft()

    def _reset_phase(self):
        """Reset phase offset display (does not undo shifts already applied)."""
        self.phase_offset_var.set(0)

    def _new_cycle(self):
        """Create a new bank with a single empty cycle and open the editor."""
        cs = self.cs_var.get()
        b  = Bank(path="new_cycle.wav",
                  audio=np.zeros(cs, dtype=np.float32),
                  sr=self.export_sr_var.get(),
                  bit_depth=self.export_depth_var.get(),
                  chunk_info={})
        b.slice(cs)
        self.banks    = [b]
        self.bank_idx = 0
        self.cycle_idx = 0
        self._set_mode("file")
        self._activate(0)
        self._open_editor()

    def _open_editor(self):
        """Open the waveform editor Toplevel window."""
        b = self.bank
        cs = self.cs_var.get()
        # Start from current cycle if available, else flat zero
        if b and b.cycles:
            init_data = self.cycles[self.cycle_idx].copy()
        else:
            init_data = np.zeros(cs, dtype=np.float32)

        ed = tk.Toplevel(self)
        ed.title("Waveform Editor")
        ed.configure(bg=C["bg"])
        ed.geometry("700x520")
        ed.resizable(True, True)

        # Working buffer — always cs samples
        buf = [resample_cycle(init_data, cs).tolist()]

        # ── Notebook tabs ────────────────────────────────────────────────────
        nb = ttk.Notebook(ed)
        nb.pack(fill="both", expand=True, padx=10, pady=(8, 4))

        tab_draw = tk.Frame(nb, bg=C["bg"])
        tab_gen  = tk.Frame(nb, bg=C["bg"])
        tab_harm = tk.Frame(nb, bg=C["bg"])
        nb.add(tab_draw, text="  Draw  ")
        nb.add(tab_gen,  text="  Generate  ")
        nb.add(tab_harm, text="  Harmonics  ")

        # ── Shared preview canvas ────────────────────────────────────────────
        preview_frame = tk.Frame(ed, bg=C["panel"])
        preview_frame.pack(fill="x", padx=10, pady=(4, 0))
        tk.Label(preview_frame, text="PREVIEW",
                 font=("Consolas", 8), bg=C["panel"], fg=C["muted"]
                 ).pack(anchor="w", padx=6, pady=(4, 0))
        pv = tk.Canvas(preview_frame, bg=C["panel"], height=80, highlightthickness=0)
        pv.pack(fill="x", padx=4, pady=(0, 4))

        def draw_preview():
            pv.delete("all")
            pw, ph = pv.winfo_width(), pv.winfo_height()
            if pw < 10: return
            # Zero line
            pv.create_line(0, ph//2, pw, ph//2, fill=C["muted"], dash=(3,3))
            data = buf[0]
            n    = len(data)
            pts  = []
            for i, v in enumerate(data):
                pts.extend([i / max(n-1,1) * pw, ph//2 - v * (ph//2 - 4)])
            if len(pts) >= 4:
                pv.create_line(*pts, fill=C["wave"], width=1.5)
        pv.bind("<Configure>", lambda e: draw_preview())

        # ── Bottom bar: apply / add / close ─────────────────────────────────
        bot = tk.Frame(ed, bg=C["bg"])
        bot.pack(fill="x", padx=10, pady=6)

        def apply_to_current():
            """Overwrite the currently displayed cycle."""
            if not (self.bank and self.bank.cycles):
                messagebox.showwarning("Editor", "No bank loaded.", parent=ed)
                return
            self.bank.cycles[self.cycle_idx] = np.array(buf[0], dtype=np.float32)
            self._refresh()
            ed.destroy()

        def add_as_new():
            """Append the edited cycle as a new cycle in the bank."""
            if not self.bank:
                messagebox.showwarning("Editor", "No bank loaded.", parent=ed)
                return
            self.bank.cycles.append(np.array(buf[0], dtype=np.float32))
            n = len(self.bank.cycles)
            self.bank.audio = np.concatenate(self.bank.cycles)
            self.cycle_idx = n - 1
            self._refresh()
            ed.destroy()

        for txt, cmd in [("Apply to current cycle", apply_to_current),
                         ("Add as new cycle",        add_as_new)]:
            tk.Button(bot, text=txt, command=cmd,
                      font=("Consolas", 9),
                      bg=C["accent"], fg=C["text"],
                      activebackground=C["hot"],
                      relief="flat", bd=0, padx=10, pady=4,
                      cursor="hand2").pack(side="left", padx=(0, 6))
        tk.Button(bot, text="Cancel", command=ed.destroy,
                  font=("Consolas", 9),
                  bg=C["grid"], fg=C["muted"],
                  activebackground=C["accent"],
                  relief="flat", bd=0, padx=10, pady=4,
                  cursor="hand2").pack(side="left")

        # ════════════════════════════════════════════════════════════════════
        # TAB 1 — Draw
        # ════════════════════════════════════════════════════════════════════
        dk = tk.Canvas(tab_draw, bg=C["panel"], highlightthickness=0)
        dk.pack(fill="both", expand=True, padx=8, pady=8)

        # Draw state
        draw_state = {"last_x": None, "last_y": None}

        def canvas_to_sample(cx, cy, cw, ch):
            """Convert canvas pixel to (sample_index, value)."""
            idx = int(cx / max(cw, 1) * cs)
            idx = max(0, min(cs - 1, idx))
            val = 1.0 - 2.0 * cy / max(ch, 1)
            val = max(-1.0, min(1.0, val))
            return idx, val

        def draw_canvas_wave():
            dk.delete("all")
            dw, dh = dk.winfo_width(), dk.winfo_height()
            if dw < 10: return
            dk.create_line(0, dh//2, dw, dh//2,
                          fill=C["muted"], dash=(3, 3))
            for yf, lbl in [(0.25, "+0.5"), (0.75, "-0.5")]:
                y = int(dh * yf)
                dk.create_line(0, y, dw, y, fill=C["grid"], dash=(2, 4))
                dk.create_text(4, y, text=lbl, font=("Consolas", 7),
                               fill=C["muted"], anchor="w")
            data = buf[0]
            n    = len(data)
            pts  = []
            for i, v in enumerate(data):
                pts.extend([i / max(n-1,1) * dw,
                            dh//2 - v * (dh//2 - 4)])
            if len(pts) >= 4:
                dk.create_line(*pts, fill=C["wave"], width=1.5)

        def on_draw_press(event):
            draw_state["last_x"] = event.x
            draw_state["last_y"] = event.y
            dw, dh = dk.winfo_width(), dk.winfo_height()
            idx, val = canvas_to_sample(event.x, event.y, dw, dh)
            buf[0][idx] = val
            draw_canvas_wave()
            draw_preview()

        def on_draw_drag(event):
            dw, dh = dk.winfo_width(), dk.winfo_height()
            lx, ly = draw_state["last_x"], draw_state["last_y"]
            if lx is None: return
            # Interpolate between last and current position
            x0, y0 = lx, ly
            x1, y1 = event.x, event.y
            steps = max(abs(x1 - x0), 1)
            for s in range(steps + 1):
                t   = s / steps
                cx  = x0 + t * (x1 - x0)
                cy  = y0 + t * (y1 - y0)
                idx, val = canvas_to_sample(cx, cy, dw, dh)
                buf[0][idx] = val
            draw_state["last_x"] = event.x
            draw_state["last_y"] = event.y
            draw_canvas_wave()
            draw_preview()

        def on_draw_release(event):
            draw_state["last_x"] = None
            draw_state["last_y"] = None

        tk.Label(tab_draw, text="Click and drag to draw the waveform",
                 font=("Consolas", 8), bg=C["bg"], fg=C["muted"]
                 ).pack(anchor="w", padx=8, pady=(4, 0))

        dk.bind("<ButtonPress-1>",   on_draw_press)
        dk.bind("<B1-Motion>",       on_draw_drag)
        dk.bind("<ButtonRelease-1>", on_draw_release)
        dk.bind("<Configure>",       lambda e: draw_canvas_wave())

        # Clear button
        def clear_draw():
            buf[0] = [0.0] * cs
            draw_canvas_wave()
            draw_preview()

        def do_snap():
            buf[0] = apply_snap(np.array(buf[0], dtype=np.float32)).tolist()
            draw_canvas_wave()
            draw_preview()

        def do_crossfade():
            n_cf = cf_var.get()
            buf[0] = apply_crossfade(np.array(buf[0], dtype=np.float32), n_cf).tolist()
            draw_canvas_wave()
            draw_preview()

        def do_normalize():
            arr = np.array(buf[0], dtype=np.float32)
            mx  = np.max(np.abs(arr))
            if mx > 1e-6:
                arr /= mx
            buf[0] = arr.tolist()
            draw_canvas_wave()
            draw_preview()

        # Periodicity tools row
        period_row = tk.Frame(tab_draw, bg=C["bg"])
        period_row.pack(anchor="w", padx=8, pady=(2, 2))
        cf_var = tk.IntVar(value=64)
        for txt, cmd in [("Clear", clear_draw),
                         ("Snap ends", do_snap),
                         ("Normalize", do_normalize)]:
            tk.Button(period_row, text=txt, command=cmd,
                      font=("Consolas", 8),
                      bg=C["accent"], fg=C["text"],
                      relief="flat", bd=0, padx=8, pady=3).pack(side="left", padx=(0, 4))
        tk.Button(period_row, text="Crossfade",
                  command=do_crossfade,
                  font=("Consolas", 8),
                  bg=C["accent"], fg=C["text"],
                  relief="flat", bd=0, padx=8, pady=3).pack(side="left", padx=(0, 4))
        tk.Label(period_row, text="N=", font=("Consolas", 8),
                 bg=C["bg"], fg=C["muted"]).pack(side="left")
        tk.Spinbox(period_row, textvariable=cf_var, from_=4, to=512, width=5,
                   font=("Consolas", 8),
                   bg=C["panel"], fg=C["text"],
                   relief="flat").pack(side="left")

        # ════════════════════════════════════════════════════════════════════
        # TAB 2 — Generate
        # ════════════════════════════════════════════════════════════════════
        gen_shape   = tk.StringVar(value="sine")
        gen_amp     = tk.DoubleVar(value=1.0)
        gen_phase   = tk.DoubleVar(value=0.0)
        gen_mix     = tk.DoubleVar(value=1.0)  # 1=new wave, 0=existing
        gen_op      = tk.StringVar(value="blend")  # blend|add|multiply|min|max

        def gen_update(*_):
            t     = np.linspace(0, 2 * np.pi, cs, endpoint=False)
            ph    = gen_phase.get() * np.pi / 180.0
            amp   = gen_amp.get()
            shape = gen_shape.get()
            if shape == "sine":
                wave_gen = np.sin(t + ph)
            elif shape == "square":
                wave_gen = np.sign(np.sin(t + ph))
            elif shape == "saw":
                wave_gen = 2 * ((t + ph) % (2 * np.pi)) / (2 * np.pi) - 1
            elif shape == "triangle":
                wave_gen = 2 * np.abs(
                    2 * ((t + ph) % (2 * np.pi)) / (2 * np.pi) - 1) - 1
            else:
                wave_gen = np.zeros(cs)
            wave_gen = wave_gen * amp
            mix      = gen_mix.get()
            # mix=1.0 → replace entirely with wave_gen
            # mix=0.0 → add wave_gen on top of existing buffer (equal blend)
            existing = np.array(buf[0], dtype=np.float32)
            op       = gen_op.get()
            if op == "blend":
                result = mix * wave_gen + (1.0 - mix) * existing
            elif op == "add":
                result = existing + wave_gen * mix
            elif op == "multiply":
                result = existing * (1.0 - mix + mix * wave_gen)
            elif op == "min":
                result = np.minimum(existing, wave_gen * mix + existing * (1-mix))
            elif op == "max":
                result = np.maximum(existing, wave_gen * mix + existing * (1-mix))
            else:
                result = wave_gen
            # Normalize if clipping
            mx = float(np.max(np.abs(result)))
            if mx > 1.0:
                result = result / mx
            buf[0] = result.tolist()
            draw_canvas_wave()
            draw_preview()

        tk.Label(tab_gen, text="Shape:", font=("Consolas", 9),
                 bg=C["bg"], fg=C["text"]).grid(row=0, column=0,
                 sticky="w", padx=12, pady=(12, 4))
        for col, (name, val) in enumerate([("Sine","sine"),("Square","square"),
                                            ("Saw","saw"),("Triangle","triangle")]):
            tk.Radiobutton(tab_gen, text=name, variable=gen_shape, value=val,
                           bg=C["bg"], fg=C["text"],
                           selectcolor=C["accent"],
                           activebackground=C["bg"],
                           font=("Consolas", 9)).grid(
                               row=0, column=col+1, padx=6, pady=(12,4))

        tk.Label(tab_gen, text="Operator:", font=("Consolas", 9),
                 bg=C["bg"], fg=C["text"]).grid(row=1, column=0,
                 sticky="w", padx=12, pady=(4, 4))
        for col, (name, val) in enumerate([("Blend","blend"),("Add","add"),
                                            ("Multiply","multiply"),
                                            ("Min","min"),("Max","max")]):
            tk.Radiobutton(tab_gen, text=name, variable=gen_op, value=val,
                           bg=C["bg"], fg=C["text"],
                           selectcolor=C["accent"],
                           activebackground=C["bg"],
                           font=("Consolas", 8)).grid(
                               row=1, column=col+1, padx=3, pady=(4,4))

        for row_i, (lbl, var, mn, mx, res) in enumerate([
            ("Amplitude",  gen_amp,   0.0, 1.0,   0.01),
            ("Phase (°)",  gen_phase, 0.0, 360.0, 1.0),
            ("Mix",        gen_mix,   0.0, 1.0,   0.01),
        ], start=2):
            tk.Label(tab_gen, text=lbl, font=("Consolas", 9),
                     bg=C["bg"], fg=C["text"]).grid(
                         row=row_i, column=0, sticky="w", padx=12, pady=4)
            sl = tk.Scale(tab_gen, variable=var, from_=mn, to=mx,
                          resolution=res, orient="horizontal",
                          bg=C["bg"], fg=C["text"],
                          troughcolor=C["accent"],
                          highlightthickness=0, length=280,
                          command=lambda v: None)
            sl.grid(row=row_i, column=1, columnspan=4, padx=8, pady=4, sticky="w")

        tk.Button(tab_gen, text="Apply shape",
                  command=gen_update,
                  font=("Consolas", 9),
                  bg=C["hot"], fg="#fff",
                  relief="flat", bd=0, padx=10, pady=4).grid(
                      row=5, column=0, columnspan=6, pady=12)

        # ════════════════════════════════════════════════════════════════════
        # TAB 3 — Harmonics
        # ════════════════════════════════════════════════════════════════════
        harm_vars = [tk.DoubleVar(value=(1.0 if i == 0 else 0.0))
                     for i in range(8)]

        def harm_update(*_):
            t    = np.linspace(0, 2 * np.pi, cs, endpoint=False)
            wave_h = np.zeros(cs)
            for i, v in enumerate(harm_vars):
                amp = v.get()
                if abs(amp) > 1e-6:
                    wave_h += amp * np.sin((i + 1) * t)
            mx = np.max(np.abs(wave_h))
            if mx > 0:
                wave_h /= mx
            buf[0] = wave_h.tolist()
            draw_canvas_wave()
            draw_preview()

        tk.Label(tab_harm, text="Harmonic amplitudes (H1=fundamental)",
                 font=("Consolas", 8), bg=C["bg"], fg=C["muted"]
                 ).grid(row=0, column=0, columnspan=2,
                        sticky="w", padx=12, pady=(10, 4))
        for i, hv in enumerate(harm_vars):
            lbl = f"H{i+1} {'(fund)' if i==0 else '       '}"
            tk.Label(tab_harm, text=lbl, font=("Consolas", 9),
                     bg=C["bg"], fg=C["text"]).grid(
                         row=i+1, column=0, sticky="w", padx=12, pady=3)
            sl = tk.Scale(tab_harm, variable=hv,
                          from_=0.0, to=1.0, resolution=0.01,
                          orient="horizontal", bg=C["bg"], fg=C["text"],
                          troughcolor=C["accent"],
                          highlightthickness=0, length=280,
                          command=lambda v: harm_update())
            sl.grid(row=i+1, column=1, padx=8, pady=3, sticky="w")

        tk.Button(tab_harm, text="Apply harmonics",
                  command=harm_update,
                  font=("Consolas", 9),
                  bg=C["hot"], fg="#fff",
                  relief="flat", bd=0, padx=10, pady=4).grid(
                      row=9, column=0, columnspan=2, pady=12)

        # Initial draw
        draw_canvas_wave()
        draw_preview()

    def _prev_bank(self):
        if self.banks:
            self._activate((self.bank_idx - 1) % len(self.banks))

    def _next_bank(self):
        if self.banks:
            self._activate((self.bank_idx + 1) % len(self.banks))

    def _prev_cycle(self):
        if self.cycles:
            self.cycle_idx = (self.cycle_idx - 1) % len(self.cycles)
            self._refresh()

    def _next_cycle(self):
        if self.cycles:
            self.cycle_idx = (self.cycle_idx + 1) % len(self.cycles)
            self._refresh()

    def _goto_cycle(self, idx: int):
        self.cycle_idx = idx
        self._refresh()

    # ── Display ──────────────────────────────────────────────────────────────
    def _refresh(self):
        if not self.cycles:
            return
        n     = len(self.cycles)
        label, _ = classify_cycle(self.cycles[self.cycle_idx])
        self.cycle_nav_lbl.config(
            text=f"Cycle  {self.cycle_idx + 1}  /  {n}")
        self.cycle_badge.config(
            text=label.upper(),
            fg=LABEL_COLORS.get(label, C["muted"]))
        self._draw_wave()
        self._draw_fft()
        self._build_thumbs()

    def _draw_wave(self):
        cv = self.wave_cv
        cv.delete("all")
        if not self.cycles:
            return
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 10 or h < 10:
            return
        # Layout margins for axes labels
        lpad, rpad, tpad, bpad = 32, 8, 6, 18
        dw = w - lpad - rpad   # drawable width
        dh = h - tpad - bpad   # drawable height
        # Grid + Y axis labels (-1, -0.5, 0, +0.5, +1)
        for val, label in [(-1.0, "-1"), (-0.5, "-.5"), (0.0, "0"),
                           (0.5, "+.5"), (1.0, "+1")]:
            y = tpad + (1.0 - (val + 1) / 2) * dh
            cv.create_line(lpad, y, w - rpad, y,
                           fill=C["grid"] if val != 0 else C["muted"],
                           dash=(4, 4) if val != 0 else ())
            cv.create_text(lpad - 3, y, text=label,
                           font=("Consolas", 7), fill=C["muted"], anchor="e")
        # X axis — sample ticks
        s      = self.cycles[self.cycle_idx]
        n_samp = len(s)
        n_ticks = min(8, n_samp)
        step    = n_samp // n_ticks
        for i in range(0, n_samp + 1, step):
            if i > n_samp:
                break
            x = lpad + (i / max(n_samp - 1, 1)) * dw
            cv.create_line(x, h - bpad, x, h - bpad + 3, fill=C["muted"])
            cv.create_text(x, h - 2, text=str(i),
                           font=("Consolas", 7), fill=C["muted"], anchor="s")
        # Waveform
        pts = []
        for i, v in enumerate(s):
            x = lpad + (i / max(n_samp - 1, 1)) * dw
            y = tpad + (1.0 - (float(v) + 1) / 2) * dh
            pts.extend([x, y])
        if len(pts) >= 4:
            cv.create_line(*pts, fill=C["wave"], width=1.5, smooth=False)
        # Phase continuity indicator: vertical markers at start and end
        disc = boundary_discontinuity(s)
        marker_color = "#c0392b" if disc > 0.20 else ("#e67e22" if disc > 0.05 else "#2ecc71")
        marker_h = min(20, int(dh * 0.25))
        # Start marker
        cv.create_line(lpad, tpad, lpad, tpad + dh, fill=marker_color, width=2)
        # End marker
        cv.create_line(w - rpad, tpad, w - rpad, tpad + dh, fill=marker_color, width=2)
        # Disc score label
        if disc > 0.01:
            cv.create_text(lpad + 4, tpad + 4,
                           text=f"disc={disc:.2f}",
                           font=("Consolas", 7), fill=marker_color, anchor="nw")

    def _draw_fft(self):
        cv = self.fft_cv
        cv.delete("all")
        if not self.cycles:
            return
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 10 or h < 10:
            return
        _, fft = classify_cycle(self.cycles[self.cycle_idx])
        n      = min(len(fft), 12)
        # Margins for Y axis
        lpad, bpad, tpad = 26, 18, 6
        dh    = h - tpad - bpad
        slot  = (w - lpad) / max(n, 1)
        bw    = max(4, int(slot * 0.65))
        lbls  = ["F","2","3","4","5","6","7","8","9","10","11","12"]
        # Y axis labels (0, 0.5, 1.0)
        for val, label in [(0.0, "0"), (0.5, ".5"), (1.0, "1")]:
            y = tpad + (1.0 - val) * dh
            cv.create_line(lpad, y, w, y, fill=C["grid"], dash=(3, 3))
            cv.create_text(lpad - 3, y, text=label,
                           font=("Consolas", 7), fill=C["muted"], anchor="e")
        # Bars
        for i in range(n):
            bh = int(float(fft[i]) * dh)
            x  = int(lpad + i * slot + (slot - bw) / 2)
            cv.create_rectangle(x, tpad + dh - bh, x + bw, tpad + dh,
                                fill=C["hot"] if i == 0 else C["fft"],
                                outline="")
            cv.create_text(x + bw // 2, h - 4, text=lbls[i],
                           font=("Consolas", 7), fill=C["muted"])

    def _build_thumbs(self):
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        for i, cyc in enumerate(self.cycles):
            label, _ = classify_cycle(cyc)
            disc      = boundary_discontinuity(cyc)
            # Border: active=hot, phase-warning=amber, phase-bad=red, normal=panel
            if i == self.cycle_idx:
                border = C["hot"]
            elif disc > 0.20:
                border = "#c0392b"   # red — severe discontinuity
            elif disc > 0.05:
                border = "#e67e22"   # amber — audible click
            else:
                border = C["panel"]
            frm = tk.Frame(self.thumb_frame, bg=border, padx=1, pady=1)
            frm.pack(side="left", padx=2)
            th = tk.Canvas(frm, width=48, height=44,
                           bg=C["panel"], highlightthickness=0, cursor="hand2")
            th.pack()
            color = LABEL_COLORS.get(label, C["muted"])
            pts   = []
            for j, v in enumerate(cyc):
                pts.extend([(j / max(len(cyc) - 1, 1)) * 48,
                            22 - float(v) * 18])
            if len(pts) >= 4:
                th.create_line(*pts, fill=color, width=1)
            idx = i
            th.bind("<Button-1>", lambda e, idx=idx: self._goto_cycle(idx))
            # Show discontinuity score under thumbnail
            disc_txt = f"{disc:.2f}" if disc > 0.01 else ""
            disc_col = "#c0392b" if disc > 0.20 else ("#e67e22" if disc > 0.05 else color)
            lbl_txt  = f"{label[:4].upper()} {disc_txt}".strip()
            tk.Label(frm, text=lbl_txt,
                     font=("Consolas", 7), bg=border, fg=disc_col).pack()

    # ── Export ───────────────────────────────────────────────────────────────
    def _prep_cycles(self, b: Bank | None = None):
        src = b or self.bank
        if not src or not src.cycles:
            messagebox.showwarning("Export", "No cycles loaded.")
            return None, None
        target = self.export_size_var.get()
        n      = self.export_n_var.get()
        cycs   = src.cycles[:n] if 0 < n <= len(src.cycles) else src.cycles
        return [resample_cycle(c, target) for c in cycs], src

    def _write(self, path: str, audio: np.ndarray, sr: int, cs: int):
        depth = self.export_depth_var.get()
        if self.export_clm_var.get():
            write_wav_with_clm(path, audio, sr, cs, depth)
        else:
            write_wav_plain(path, audio, sr, depth)

    def _exp_solo(self):
        if not self.cycles:
            messagebox.showwarning("Export", "No cycles loaded.")
            return
        b      = self.bank
        target = self.export_size_var.get()
        label, _ = classify_cycle(self.cycles[self.cycle_idx])
        suf    = "_clm" if self.export_clm_var.get() else ""
        name   = (f"{os.path.splitext(b.name)[0]}"
                  f"_cycle{self.cycle_idx + 1:02d}_{label}_{target}{suf}.wav")
        path = filedialog.asksaveasfilename(
            title="Export current cycle", initialfile=name,
            defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        out_sr = self.export_sr_var.get()
        self._write(path, resample_cycle(self.cycles[self.cycle_idx], target),
                    out_sr, target)
        self.status_var.set(f"Exported: {os.path.basename(path)}")

    def _exp_separate(self):
        cycs, b = self._prep_cycles()
        if not cycs:
            return
        folder = filedialog.askdirectory(title="Choose export folder")
        if not folder:
            return
        target = self.export_size_var.get()
        suf    = "_clm" if self.export_clm_var.get() else ""
        base   = os.path.splitext(b.name)[0]
        for i, c in enumerate(cycs):
            label, _ = classify_cycle(b.cycles[i])
            fname = f"{base}_cycle{i + 1:02d}_{label}_{target}{suf}.wav"
            out_sr = self.export_sr_var.get()
            self._write(os.path.join(folder, fname), c, out_sr, target)
        msg = f"{len(cycs)} files → {folder}"
        self.status_var.set(msg)
        messagebox.showinfo("Export complete", msg)

    def _exp_unified(self, b: Bank | None = None):
        cycs, src = self._prep_cycles(b)
        if not cycs:
            return
        target  = self.export_size_var.get()
        suf     = "_clm" if self.export_clm_var.get() else ""
        default = f"{os.path.splitext(src.name)[0]}_{len(cycs)}x{target}{suf}.wav"
        path = filedialog.asksaveasfilename(
            title="Save unified wavetable WAV", initialfile=default,
            defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if not path:
            return
        out_sr = self.export_sr_var.get()
        self._write(path, np.concatenate(cycs), out_sr, target)
        msg = f"{os.path.basename(path)}  |  {len(cycs)} × {target} samp"
        self.status_var.set(msg)
        messagebox.showinfo("Export complete", msg)

    def _exp_all_banks(self):
        if not self.banks:
            return
        folder = filedialog.askdirectory(title="Export all banks to folder")
        if not folder:
            return
        target = self.export_size_var.get()
        suf    = "_clm" if self.export_clm_var.get() else ""
        ok, errors = 0, []
        for bank in self.banks:
            cycs, _ = self._prep_cycles(bank)
            if not cycs:
                continue
            fname = f"{os.path.splitext(bank.name)[0]}_{len(cycs)}x{target}{suf}.wav"
            try:
                self._write(os.path.join(folder, fname),
                            np.concatenate(cycs), bank.sr, target)
                ok += 1
            except Exception as e:
                errors.append(f"{bank.name}: {e}")
        if errors:
            messagebox.showwarning("Some exports failed", "\n".join(errors))
        msg = f"{ok}/{len(self.banks)} banks exported → {folder}"
        self.status_var.set(msg)
        messagebox.showinfo("Export all complete", msg)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    App().mainloop()
