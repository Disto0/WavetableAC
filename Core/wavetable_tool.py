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


def build_heatmap(cycles: list, n_harmonics: int = 16) -> 'np.ndarray':
    """
    Build a 2D heatmap array [n_cycles × n_harmonics].
    Each cell = amplitude of harmonic H(j+1) in cycle i.
    Values normalized per-harmonic across cycles to [0,1].
    """
    if not cycles:
        return np.zeros((0, n_harmonics), dtype=np.float32)
    data = []
    for c in cycles:
        fft  = np.abs(np.fft.rfft(c))
        amps = np.array([float(fft[i+1]) if i+1 < len(fft) else 0.0
                         for i in range(n_harmonics)], dtype=np.float32)
        data.append(amps)
    arr = np.array(data, dtype=np.float32)
    for h in range(n_harmonics):
        col = arr[:, h]
        mn, mx = float(col.min()), float(col.max())
        if mx > mn:
            arr[:, h] = (col - mn) / (mx - mn)
    return arr


def build_morph_coherence_path(cycles: list, n_steps: int = 200) -> 'np.ndarray':
    """
    Compute spectral coherence score along the full bank morph path.
    Returns array of n_steps floats in [0,1].
    """
    if len(cycles) < 2:
        return np.ones(n_steps, dtype=np.float32)
    mean_fft = np.mean(
        [np.abs(np.fft.rfft(c))[:17] for c in cycles], axis=0).astype(np.float32)
    result = []
    n = len(cycles)
    for i in range(n_steps):
        pos   = i / (n_steps - 1) * (n - 1)
        idx_a = int(pos)
        idx_b = min(idx_a + 1, n - 1)
        t_m   = pos - idx_a
        ca, cb = cycles[idx_a], cycles[idx_b]
        sz = max(len(ca), len(cb))
        if len(ca) != sz:
            ca = np.interp(np.linspace(0,1,sz,endpoint=False),
                           np.linspace(0,1,len(ca),endpoint=False), ca)
        if len(cb) != sz:
            cb = np.interp(np.linspace(0,1,sz,endpoint=False),
                           np.linspace(0,1,len(cb),endpoint=False), cb)
        m_fft = np.abs(np.fft.rfft(((1-t_m)*ca + t_m*cb).astype(np.float32)))[:17]
        norm  = float(np.linalg.norm(m_fft)) * float(np.linalg.norm(mean_fft))
        result.append(float(np.dot(m_fft, mean_fft) / norm) if norm > 0 else 0.0)
    return np.array(result, dtype=np.float32)


def spectral_coherence(cycles: list, n_harmonics: int = 16) -> dict:
    """
    Analyze spectral coherence across all cycles.
    Returns global score [0,1] and per-cycle scores.
    1.0 = perfectly coherent, <0.85 = problematic.
    """
    if not cycles:
        return {"global": 0.0, "per_cycle": [], "harm_std": np.zeros(n_harmonics),
                "profiles": np.zeros((0, n_harmonics)), "mean_profile": np.zeros(n_harmonics)}
    profiles = []
    for c in cycles:
        fft  = np.abs(np.fft.rfft(c))
        amps = np.array([float(fft[i+1]) if i+1 < len(fft) else 0.0
                         for i in range(n_harmonics)], dtype=np.float32)
        mx   = float(amps.max())
        profiles.append(amps/mx if mx > 0 else amps)
    P    = np.array(profiles)
    mean = P.mean(axis=0)
    sims = []
    for p in P:
        norm = float(np.linalg.norm(p)) * float(np.linalg.norm(mean))
        sims.append(float(np.dot(p, mean)/norm) if norm > 0 else 0.0)
    return {"global": float(np.mean(sims)), "per_cycle": sims,
            "harm_std": P.std(axis=0), "profiles": P, "mean_profile": mean}


def extract_harmonics(cycle: np.ndarray, n: int = 16) -> np.ndarray:
    """Extract n harmonic amplitudes from cycle, normalized to [0,1]. H[0]=fundamental."""
    fft  = np.abs(np.fft.rfft(cycle))
    amps = np.array([float(fft[i+1]) if i+1 < len(fft) else 0.0
                     for i in range(n)], dtype=np.float32)
    mx   = amps.max()
    return amps / mx if mx > 0 else amps


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

# ---------------------------------------------------------------------------
#  Frequency detection & cycle scanner helpers
# ---------------------------------------------------------------------------
def detect_fundamental(audio: np.ndarray, sr: int,
                        f_min: float = 20.0,
                        f_max: float = 4000.0) -> float:
    """
    Detect fundamental frequency using NSDF (Normalized Square Difference).
    Avoids the low-lag plateau issue of raw autocorrelation.
    Returns frequency in Hz.
    """
    lag_min = max(2, int(sr / f_max))
    lag_max = min(len(audio) // 2, int(sr / f_min))
    if lag_min >= lag_max:
        return 440.0
    n     = len(audio)
    fft_a = np.fft.rfft(audio - float(audio.mean()), n=2 * n)
    acorr = np.fft.irfft(fft_a * np.conj(fft_a))[:n].real
    energy = float(acorr[0])
    if energy < 1e-10:
        return 440.0
    nsdf    = acorr / energy
    segment = nsdf[lag_min:lag_max]
    # Find first downward zero-crossing (valley), then best peak after it
    first_valley = 0
    for i in range(1, len(segment)):
        if segment[i - 1] > 0 >= segment[i]:
            first_valley = i
            break
    if first_valley > 0 and first_valley < len(segment):
        sub = segment[first_valley:]
        best_lag = lag_min + first_valley + int(np.argmax(sub))
    else:
        best_lag = lag_min + int(np.argmax(segment))
    return float(sr / max(best_lag, 1))


def freq_to_note(freq: float) -> str:
    """Convert frequency to nearest note name + cents offset string."""
    if freq <= 0:
        return "—"
    names  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    midi_f = 69 + 12 * np.log2(max(freq, 1.0) / 440.0)
    midi_n = int(round(midi_f))
    cents  = (midi_f - midi_n) * 100
    name   = names[midi_n % 12] + str(midi_n // 12 - 1)
    sign   = "+" if cents >= 0 else ""
    return f"{name} ({sign}{cents:.0f}¢)"


def find_zero_crossing_start(audio: np.ndarray, period: float) -> int:
    """Find first positive-slope zero crossing within two periods."""
    limit = min(int(period * 2), len(audio) - 1)
    for i in range(1, limit):
        if audio[i - 1] <= 0 < audio[i]:
            return i
    return 0


def extract_cycles_from_audio(audio: np.ndarray, sr: int,
                               freq: float) -> list:
    """
    Extract all complete cycles from audio at the given frequency.
    Returns list of dicts: {index, start, end, audio, stability}.
    """
    period    = sr / freq
    start_off = find_zero_crossing_start(audio, period)
    cycles    = []
    first     = None
    pos       = float(start_off)
    while pos + period <= len(audio):
        s = int(round(pos))
        e = min(int(round(pos + period)), len(audio))
        if e - s >= 4:
            cyc = audio[s:e].copy()
            if first is None:
                first = cyc
            ref = np.interp(np.linspace(0, 1, len(cyc), endpoint=False),
                            np.linspace(0, 1, len(first), endpoint=False),
                            first)
            na, nr = np.linalg.norm(cyc), np.linalg.norm(ref)
            stab = float(np.dot(cyc, ref) / (na * nr + 1e-10))                    if na * nr > 0 else 0.0
            cycles.append({"index": len(cycles), "start": s, "end": e,
                           "audio": cyc, "stability": stab})
        pos += period
    return cycles


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
        self.geometry("1280x860")
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
        self.morph_var        = None
        self.global_morph_var  = None
        self._view_btns:       dict = {}
        self._show_overlay_var       = None
        self._show_legend_var        = None
        self._harmonic_filter: set   = set()
        # Zoom state
        self._zoom_start: int = 0
        self._zoom_end:   int = -1
        # Playback state
        self._loop_running:  bool = False
        self._loop_thread          = None
        self._morph_cached         = None
        # Multi-cycle selection for overlay (set of indices)
        self._selected_cycles: set = set()
        # View mode: "waveform" | "fft" | "heatmap" | "harmonic_lines"
        self._view_mode:       str = "waveform"
        # Global morph position (0..n_cycles-1)
        self._global_morph_pos: float = 0.0
        # Undo stack
        self._undo_stack: list = []
        self._max_undo:   int  = 30

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

        # Part B — scrollable left panel
        b_outer = tk.Frame(body, bg=C["panel"], width=246)
        b_outer.pack(side="left", fill="y")
        b_outer.pack_propagate(False)
        b_cv = tk.Canvas(b_outer, bg=C["panel"], highlightthickness=0, width=228)
        b_sb = ttk.Scrollbar(b_outer, orient="vertical", command=b_cv.yview)
        b_cv.configure(yscrollcommand=b_sb.set)
        b_sb.pack(side="right", fill="y")
        b_cv.pack(side="left", fill="both", expand=True)
        self.panel_b = tk.Frame(b_cv, bg=C["panel"], width=228)
        b_cv.create_window((0, 0), window=self.panel_b, anchor="nw")
        self.panel_b.bind("<Configure>",
            lambda e, c=b_cv: c.configure(scrollregion=c.bbox("all")))
        def _b_wheel(event, c=b_cv):
            c.yview_scroll(int(-1*(event.delta/120)), "units")
        b_cv.bind("<MouseWheel>",        _b_wheel)
        self.panel_b.bind("<MouseWheel>", _b_wheel)
        self._build_panel_b()

        # Right column: Part C + Part D stacked
        right_col = tk.Frame(body, bg=C["bg"])
        right_col.pack(side="left", fill="both", expand=True)

        # Part C — visualisation
        self.panel_c = tk.Frame(right_col, bg=C["bg"])
        self.panel_c.pack(fill="both", expand=True, padx=10, pady=(8, 4))
        self._build_panel_c()

        # Keyboard shortcuts
        self.bind("<Control-z>", self._undo)
        self.bind("<Control-Z>", self._undo)

        # Keyboard shortcuts
        self.bind('<Control-z>', self._undo)
        self.bind('<Control-Z>', self._undo)

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
        self._btn(p, "Create new bank", self._create_bank).pack(
            fill="x", padx=10, pady=2)
        self._btn(p, "Add empty cycle", self._add_empty_cycle).pack(
            fill="x", padx=10, pady=2)
        self._btn(p, "Scan WAV for cycles...", self._open_scanner).pack(
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
        self.wave_cv.bind("<MouseWheel>",   lambda e: self._zoom_scroll(e.delta))
        self.wave_cv.bind("<Button-4>",     lambda e: self._zoom_scroll(120))
        self.wave_cv.bind("<Button-5>",     lambda e: self._zoom_scroll(-120))
        self.wave_cv.bind("<B2-Motion>",    self._on_pan_wave)  # middle-click drag
        self.wave_cv.bind("<B3-Motion>",    self._on_pan_wave)  # right-click drag

        ff = tk.Frame(vis_row, bg=C["panel"])
        ff.grid(row=0, column=1, sticky="nsew")
        fft_hdr = tk.Frame(ff, bg=C["panel"])
        fft_hdr.pack(fill="x")
        tk.Label(fft_hdr, text="FFT SPECTRUM",
                 font=("Consolas", 8), bg=C["panel"], fg=C["muted"]).pack(
                     side="left", padx=8, pady=(4, 0))
        self.fft_filter_lbl = tk.Label(fft_hdr, text="",
                                       font=("Consolas", 7), bg=C["panel"],
                                       fg=C["hot"])
        self.fft_filter_lbl.pack(side="right", padx=6, pady=(4, 0))
        self.fft_cv = tk.Canvas(ff, bg=C["panel"], highlightthickness=0)
        self.fft_cv.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.fft_cv.bind("<Configure>", lambda e: self._draw_fft())
        self.fft_cv.bind("<Button-1>",  self._on_fft_click)

        # ── View mode tabs ──
        tab_row = tk.Frame(p, bg=C["bg"])
        tab_row.pack(fill="x", pady=(2, 0))
        self._view_btns = {}
        for lbl, mode in [("Waveform","waveform"),("FFT","fft"),
                          ("Heatmap","heatmap"),("Lines","harmonic_lines")]:
            b = tk.Button(tab_row, text=lbl,
                          font=("Consolas", 8),
                          bg=C["hot"] if mode=="waveform" else C["accent"],
                          fg=C["text"],
                          activebackground=C["hot"],
                          relief="flat", bd=0, padx=8, pady=3,
                          cursor="hand2",
                          command=lambda m=mode: self._set_view_mode(m))
            b.pack(side="left", padx=2)
            self._view_btns[mode] = b
        # View checkboxes
        self._show_overlay_var = tk.BooleanVar(value=False)
        tk.Checkbutton(tab_row, text="Overlay",
                       variable=self._show_overlay_var,
                       command=self._refresh_view,
                       bg=C["bg"], fg=C["text"],
                       selectcolor=C["accent"],
                       activebackground=C["bg"],
                       font=("Consolas", 8)).pack(side="left", padx=(8,2))
        self._show_legend_var = tk.BooleanVar(value=True)
        tk.Checkbutton(tab_row, text="Legend",
                       variable=self._show_legend_var,
                       command=self._refresh_view,
                       bg=C["bg"], fg=C["text"],
                       selectcolor=C["accent"],
                       activebackground=C["bg"],
                       font=("Consolas", 8)).pack(side="left", padx=2)

        # ── Global morph slider ──
        gmorph_row = tk.Frame(p, bg=C["bg"])
        gmorph_row.pack(fill="x", pady=(2, 0))
        tk.Label(gmorph_row, text="Global morph:",
                 font=("Consolas", 8), bg=C["bg"], fg=C["muted"]).pack(side="left")
        self.global_morph_var = tk.DoubleVar(value=0.0)
        self.global_morph_scale = tk.Scale(
            gmorph_row, variable=self.global_morph_var,
            from_=0.0, to=1.0, resolution=0.001,
            orient="horizontal", length=220,
            bg=C["bg"], fg=C["text"], troughcolor=C["accent"],
            highlightthickness=0, showvalue=False,
            command=self._on_global_morph)
        self.global_morph_scale.pack(side="left", padx=4)
        self.global_morph_lbl = tk.Label(gmorph_row, text="pos: 0.00",
                                         font=("Consolas", 8), bg=C["bg"], fg=C["muted"])
        self.global_morph_lbl.pack(side="left", padx=4)

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
        self._sbtn(info_row, "← Move", self._cycle_move_left).pack(side="left", padx=(8,1))
        self._sbtn(info_row, "→ Move", self._cycle_move_right).pack(side="left", padx=1)
        # Zoom controls
        tk.Label(info_row, text="  Zoom:",
                 font=("Consolas", 8), bg=C["bg"], fg=C["muted"]).pack(side="left", padx=(8,2))
        self._sbtn(info_row, "+", self._zoom_in).pack(side="left", padx=1)
        self._sbtn(info_row, "−", self._zoom_out).pack(side="left", padx=1)
        self._sbtn(info_row, "Fit", self._zoom_reset).pack(side="left", padx=1)
        # Phase offset control
        phase_row = tk.Frame(self.panel_c, bg=C["bg"])
        phase_row.pack(fill="x", pady=(2, 0))
        tk.Label(phase_row, text="Phase:",
                 font=("Consolas", 8), bg=C["bg"], fg=C["muted"]).pack(side="left")
        self.phase_offset_var = tk.IntVar(value=0)
        self.phase_slider_var = tk.IntVar(value=0)
        self._sbtn(phase_row, "−100", lambda: self._shift_cycle(-100)).pack(side="left", padx=(4,1))
        self._sbtn(phase_row, "−10",  lambda: self._shift_cycle(-10)).pack(side="left", padx=1)
        self._sbtn(phase_row, "−1",   lambda: self._shift_cycle(-1)).pack(side="left", padx=1)
        self.phase_sl = tk.Scale(phase_row, variable=self.phase_slider_var,
                                 from_=-1024, to=1024, resolution=1,
                                 orient="horizontal", length=120,
                                 bg=C["bg"], fg=C["text"], troughcolor=C["accent"],
                                 highlightthickness=0, showvalue=False,
                                 command=self._on_phase_slider)
        self.phase_sl.pack(side="left", padx=2)
        tk.Label(phase_row, textvariable=self.phase_offset_var, width=5,
                 font=("Consolas", 8, "bold"), bg=C["bg"], fg=C["hot"]).pack(side="left")
        self._sbtn(phase_row, "+1",   lambda: self._shift_cycle(1)).pack(side="left", padx=1)
        self._sbtn(phase_row, "+10",  lambda: self._shift_cycle(10)).pack(side="left", padx=1)
        self._sbtn(phase_row, "+100", lambda: self._shift_cycle(100)).pack(side="left", padx=1)
        self._sbtn(phase_row, "Reset", self._reset_phase).pack(side="left", padx=(4, 0))

        # Playback controls
        play_row = tk.Frame(self.panel_c, bg=C["bg"])
        play_row.pack(fill="x", pady=(2, 0))
        tk.Label(play_row, text="Play:",
                 font=("Consolas", 8), bg=C["bg"], fg=C["muted"]).pack(side="left")
        self._sbtn(play_row, "▶ Once",  self._play_cycle).pack(side="left", padx=(4,2))
        self._sbtn(play_row, "↺ Loop",  self._play_loop).pack(side="left", padx=2)
        self._sbtn(play_row, "■ Stop",  self._stop_play).pack(side="left", padx=2)
        tk.Label(play_row, text="  Freq:",
                 font=("Consolas", 8), bg=C["bg"], fg=C["muted"]).pack(side="left")
        self.play_freq_var = tk.DoubleVar(value=440.0)
        tk.Scale(play_row, variable=self.play_freq_var,
                 from_=55.0, to=1760.0, resolution=1.0,
                 orient="horizontal", length=120,
                 bg=C["bg"], fg=C["text"], troughcolor=C["accent"],
                 highlightthickness=0, showvalue=False).pack(side="left", padx=2)
        self.play_freq_lbl = tk.Label(play_row, text="440 Hz",
                                      font=("Consolas", 8), bg=C["bg"], fg=C["text"])
        self.play_freq_lbl.pack(side="left")
        self.play_freq_var.trace_add("write", self._on_freq_change)

        # ── Morph row ──
        morph_row = tk.Frame(p, bg=C["bg"])
        morph_row.pack(fill="x", pady=(4, 0))
        tk.Label(morph_row, text="Morph:",
                 font=("Consolas", 8), bg=C["bg"], fg=C["muted"]).pack(side="left")
        self.morph_var = tk.DoubleVar(value=0.0)
        tk.Scale(morph_row, variable=self.morph_var,
                 from_=0.0, to=1.0, resolution=0.01,
                 orient="horizontal", length=160,
                 bg=C["bg"], fg=C["text"], troughcolor=C["accent"],
                 highlightthickness=0, showvalue=False,
                 command=self._on_morph).pack(side="left", padx=4)
        self.morph_lbl = tk.Label(morph_row, text="C1←0.00→C2",
                                  font=("Consolas", 8), bg=C["bg"], fg=C["muted"])
        self.morph_lbl.pack(side="left", padx=4)
        self._sbtn(morph_row, "Bake", self._bake_morph).pack(side="left", padx=(8,2))
        self.morph_coh_lbl = tk.Label(morph_row, text="",
                                      font=("Consolas", 8), bg=C["bg"], fg=C["muted"])
        self.morph_coh_lbl.pack(side="right", padx=8)

        # ── Spectral coherence canvas (bank + morph gradient) ──
        coh_frame = tk.Frame(p, bg=C["panel"])
        coh_frame.pack(fill="x", pady=(3, 0))
        coh_hdr = tk.Frame(coh_frame, bg=C["panel"])
        coh_hdr.pack(fill="x")
        tk.Label(coh_hdr, text="SPECTRAL COHERENCE",
                 font=("Consolas", 7), bg=C["panel"], fg=C["muted"]).pack(
                     side="left", padx=6, pady=(2,0))
        self.coh_global_lbl = tk.Label(coh_hdr, text="global: —",
                                       font=("Consolas", 7), bg=C["panel"], fg=C["muted"])
        self.coh_global_lbl.pack(side="right", padx=6, pady=(2,0))
        self.coh_cv = tk.Canvas(coh_frame, bg=C["panel"], height=48,
                                highlightthickness=0)
        self.coh_cv.pack(fill="x", padx=4, pady=(0, 3))
        self.coh_cv.bind("<Configure>", lambda e: self._draw_coherence())

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
        """Play current cycle once (respects morph position and freq slider)."""
        import tempfile, threading
        audio, sr = self._get_play_audio()
        if audio is None:
            return
        # Trim to 1 second for "once" play
        audio = audio[:sr]
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
                try: os.unlink(tmp_path)
                except Exception: pass
        threading.Thread(target=_play, daemon=True).start()


    def _cycle_move_left(self):
        """Move current cycle one position to the left in the bank."""
        self._push_undo()
        b = self.bank
        if not b or len(b.cycles) < 2 or self.cycle_idx == 0:
            return
        i = self.cycle_idx
        b.cycles[i-1], b.cycles[i] = b.cycles[i], b.cycles[i-1]
        b.audio = np.concatenate(b.cycles)
        self.cycle_idx = i - 1
        self._refresh()

    def _cycle_move_right(self):
        """Move current cycle one position to the right in the bank."""
        self._push_undo()
        b = self.bank
        if not b or len(b.cycles) < 2 or self.cycle_idx >= len(b.cycles)-1:
            return
        i = self.cycle_idx
        b.cycles[i], b.cycles[i+1] = b.cycles[i+1], b.cycles[i]
        b.audio = np.concatenate(b.cycles)
        self.cycle_idx = i + 1
        self._refresh()

    def _delete_cycle(self):
        """Delete the currently displayed cycle from the active bank."""
        self._push_undo()
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
        self._push_undo()
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

    def _create_bank(self):
        """Create a brand-new empty bank and open the editor on its first cycle."""
        cs = self.cs_var.get()
        b  = Bank(path="new_bank.wav",
                  audio=np.zeros(cs, dtype=np.float32),
                  sr=self.export_sr_var.get(),
                  bit_depth=self.export_depth_var.get(),
                  chunk_info={})
        b.slice(cs)
        self.banks     = [b]
        self.bank_idx  = 0
        self.cycle_idx = 0
        self._set_mode("file")
        self._activate(0)
        self._open_editor()

    def _add_empty_cycle(self):
        """Add a new empty cycle to the current bank and open the editor."""
        cs = self.cs_var.get()
        if not self.bank:
            self._create_bank()
            return
        empty = np.zeros(cs, dtype=np.float32)
        self.bank.cycles.append(empty)
        self.bank.audio = np.concatenate(self.bank.cycles)
        self.cycle_idx  = len(self.bank.cycles) - 1
        self._refresh()
        self._open_editor()

    def _new_cycle(self):
        """Legacy alias kept for compatibility."""
        self._create_bank()

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
        ed.geometry("820x640")
        ed.resizable(True, True)

        # Working buffer — always cs samples
        buf = [resample_cycle(init_data, cs).tolist()]

        # ── Notebook tabs ────────────────────────────────────────────────────
        nb = ttk.Notebook(ed)
        nb.pack(fill="both", expand=True, padx=10, pady=(8, 4))

        tab_draw = tk.Frame(nb, bg=C["bg"])
        tab_gen  = tk.Frame(nb, bg=C["bg"])
        tab_harm  = tk.Frame(nb, bg=C["bg"])
        tab_layer = tk.Frame(nb, bg=C["bg"])
        nb.add(tab_draw,  text="  Draw  ")
        nb.add(tab_gen,   text="  Generate  ")
        nb.add(tab_harm,  text="  Harmonics  ")
        nb.add(tab_layer, text="  Layer  ")

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
            wh = (ph * 2) // 3   # top 2/3 waveform
            fh = ph - wh - 2     # bottom 1/3 FFT
            # Waveform
            pv.create_line(0, wh//2, pw, wh//2, fill=C["muted"], dash=(3,3))
            data = buf[0]
            n    = len(data)
            pts  = []
            for i, v in enumerate(data):
                pts.extend([i/max(n-1,1)*pw, wh//2 - float(v)*(wh//2-3)])
            if len(pts) >= 4:
                pv.create_line(*pts, fill=C["wave"], width=1.5)
            # FFT bars
            fft_vals = extract_harmonics(np.array(data, dtype=np.float32), 16)
            slot = pw / 16
            bw   = max(2, int(slot * 0.7))
            pv.create_line(0, wh+1, pw, wh+1, fill=C["grid"])
            for i, amp in enumerate(fft_vals):
                bh = int(float(amp) * fh)
                x  = int(i * slot + (slot-bw)/2)
                pv.create_rectangle(x, ph-bh, x+bw, ph,
                                    fill=C["hot"] if i==0 else C["fft"],
                                    outline="")
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

        # Draw canvas layout margins (mirrors oscilloscope style)
        lpad_d, bpad_d, tpad_d = 28, 14, 6

        # Draw state
        draw_state = {"last_x": None, "last_y": None}

        def canvas_to_sample(cx, cy, cw, ch):
            """Convert canvas pixel to (sample_index, value) using draw margins."""
            draw_w = cw - lpad_d
            draw_h = ch - tpad_d - bpad_d
            idx = int((cx - lpad_d) / max(draw_w, 1) * cs)
            idx = max(0, min(cs - 1, idx))
            val = 1.0 - 2.0 * (cy - tpad_d) / max(draw_h, 1)
            val = max(-1.0, min(1.0, val))
            return idx, val

        def draw_canvas_wave():
            dk.delete("all")
            dw, dh = dk.winfo_width(), dk.winfo_height()
            if dw < 10: return
            # Grid lines
            for yf, lbl in [(0.0, "+1"), (0.25, "+0.5"), (0.5, "0"),
                            (0.75, "-0.5"), (1.0, "-1")]:
                y = int(tpad_d + yf * (dh - tpad_d - bpad_d))
                dk.create_line(lpad_d, y, dw, y,
                               fill=C["grid"] if yf != 0.5 else C["muted"],
                               dash=(3, 3) if yf != 0.5 else ())
                dk.create_text(lpad_d - 3, y, text=lbl,
                               font=("Consolas", 7), fill=C["muted"], anchor="e")
            # Waveform as polygon fill (area between waveform and zero line)
            data   = buf[0]
            n      = len(data)
            zero_y = tpad_d + 0.5 * (dh - tpad_d - bpad_d)
            pts_top = []
            for i, v in enumerate(data):
                x = lpad_d + (i / max(n-1, 1)) * (dw - lpad_d)
                y = tpad_d + (0.5 - v * 0.5) * (dh - tpad_d - bpad_d)
                pts_top.extend([x, y])
            # Polygon: waveform top + reverse along zero line
            if len(pts_top) >= 4:
                poly_pts = pts_top + [dw, zero_y, lpad_d, zero_y]
                dk.create_polygon(*poly_pts,
                                  fill="#1a3a5c", outline="")  # subtle fill
                dk.create_line(*pts_top, fill=C["wave"], width=1.5)

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

        def do_invert_v():
            arr = np.array(buf[0], dtype=np.float32)
            buf[0] = (-arr).tolist()
            draw_canvas_wave(); draw_preview()

        def do_invert_h():
            buf[0] = buf[0][::-1]
            draw_canvas_wave(); draw_preview()

        invert_row = tk.Frame(tab_draw, bg=C["bg"])
        invert_row.pack(anchor="w", padx=8, pady=(2, 0))
        for txt, cmd in [("Invert V", do_invert_v), ("Mirror H", do_invert_h)]:
            tk.Button(invert_row, text=txt, command=cmd,
                      font=("Consolas", 8), bg=C["accent"], fg=C["text"],
                      relief="flat", bd=0, padx=8, pady=3).pack(side="left", padx=(0,4))

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
            elif op == "subtract":
                result = existing - wave_gen * mix
            elif op == "multiply":
                result = existing * (1.0 - mix + mix * wave_gen)
            elif op == "divide":
                denom = 1.0 - mix + mix * (np.abs(wave_gen) + 1e-6)
                result = existing / denom
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
                                            ("Sub","subtract"),("Mul","multiply"),
                                            ("Div","divide"),
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
                      row=5, column=0, columnspan=8, pady=12)

        # ════════════════════════════════════════════════════════════════════
        # TAB 3 — Harmonics
        # ════════════════════════════════════════════════════════════════════
        # Preload harmonic values from the current cycle buffer
        _existing_harmonics = extract_harmonics(np.array(buf[0], dtype=np.float32), 16)
        harm_vars = [tk.DoubleVar(value=float(_existing_harmonics[i]))
                     for i in range(16)]

        def harm_update(*_):
            t_h    = np.linspace(0, 2 * np.pi, cs, endpoint=False)
            wave_h = np.zeros(cs, dtype=np.float64)
            for i, v in enumerate(harm_vars):
                amp = float(v.get())
                if abs(amp) > 1e-6:
                    wave_h += amp * np.sin((i + 1) * t_h)
            mx = float(np.max(np.abs(wave_h)))
            if mx > 1e-10:
                wave_h /= mx
            buf[0] = wave_h.astype(np.float32).tolist()
            draw_canvas_wave()
            draw_preview()

        # Scrollable frame for 16 harmonics
        tk.Label(tab_harm, text="Harmonic amplitudes — H1=fundamental (matches FFT labels)",
                 font=("Consolas", 8), bg=C["bg"], fg=C["muted"]
                 ).pack(anchor="w", padx=12, pady=(8, 2))
        harm_scroll_outer = tk.Frame(tab_harm, bg=C["bg"])
        harm_scroll_outer.pack(fill="both", expand=True, padx=8)
        harm_cv = tk.Canvas(harm_scroll_outer, bg=C["bg"], highlightthickness=0)
        harm_sb = ttk.Scrollbar(harm_scroll_outer, orient="vertical",
                                command=harm_cv.yview)
        harm_cv.configure(yscrollcommand=harm_sb.set)
        harm_sb.pack(side="right", fill="y")
        harm_cv.pack(side="left", fill="both", expand=True)
        harm_inner = tk.Frame(harm_cv, bg=C["bg"])
        harm_cv.create_window((0,0), window=harm_inner, anchor="nw")
        harm_inner.bind("<Configure>",
            lambda e: harm_cv.configure(scrollregion=harm_cv.bbox("all")))

        for i, hv in enumerate(harm_vars):
            row_f = tk.Frame(harm_inner, bg=C["bg"])
            row_f.pack(fill="x", pady=1)
            tk.Label(row_f, text=f"H{i+1:2d}", width=4,
                     font=("Consolas", 9), bg=C["bg"],
                     fg=C["hot"] if i==0 else C["text"]).pack(side="left", padx=(8,2))
            sl = tk.Scale(row_f, variable=hv, from_=0.0, to=1.0,
                          resolution=0.01, orient="horizontal",
                          bg=C["bg"], fg=C["text"],
                          troughcolor=C["accent"],
                          highlightthickness=0, length=240,
                          showvalue=False,
                          command=lambda v: None)
            sl.pack(side="left")
            tk.Label(row_f, textvariable=hv, width=4,
                     font=("Consolas", 8), bg=C["bg"],
                     fg=C["muted"]).pack(side="left", padx=4)

        tk.Button(tab_harm, text="Apply harmonics", command=harm_update,
                  font=("Consolas", 9), bg=C["hot"], fg="#fff",
                  relief="flat", bd=0, padx=10, pady=4).pack(pady=6)

        # ════════════════════════════════════════════════════════════════════
        # TAB 4 — Layer (blend cycles from bank or external WAV)
        # ════════════════════════════════════════════════════════════════════
        layer_src_audio = [None]  # holds the source cycle to blend

        lay_mix_var = tk.DoubleVar(value=0.5)
        lay_op_var  = tk.StringVar(value="blend")

        # Source selection
        src_frame = tk.Frame(tab_layer, bg=C["bg"])
        src_frame.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(src_frame, text="Source:",
                 font=("Consolas", 9), bg=C["bg"], fg=C["text"]).pack(
                     side="left")

        def load_layer_from_bank():
            """Pick a cycle from the current bank as blend source."""
            bc = self.bank
            if not bc or not bc.cycles:
                messagebox.showinfo("Layer", "No bank loaded.", parent=ed)
                return
            # Simple dialog: show cycle list
            pick = tk.Toplevel(ed)
            pick.title("Pick source cycle")
            pick.configure(bg=C["bg"])
            pick.geometry("300x320")
            tk.Label(pick, text="Select cycle to use as blend source:",
                     font=("Consolas", 9), bg=C["bg"], fg=C["text"],
                     wraplength=280).pack(padx=12, pady=(10, 4))
            lb = tk.Listbox(pick, font=("Consolas", 9),
                            bg=C["panel"], fg=C["text"],
                            selectbackground=C["accent"],
                            height=12)
            lb.pack(fill="both", expand=True, padx=12, pady=4)
            for i, cyc in enumerate(bc.cycles):
                lbl, _ = classify_cycle(cyc)
                lb.insert(tk.END, f"Cycle {i+1:02d}  {lbl.upper()}")
            def confirm():
                sel = lb.curselection()
                if not sel: return
                idx = sel[0]
                layer_src_audio[0] = resample_cycle(bc.cycles[idx], cs)
                src_lbl.config(text=f"Bank cycle {idx+1}")
                lay_preview()
                pick.destroy()
            tk.Button(pick, text="Use this cycle", command=confirm,
                      font=("Consolas", 9),
                      bg=C["hot"], fg="#fff",
                      relief="flat", bd=0, padx=10, pady=4).pack(pady=8)

        def load_layer_from_file():
            """Load external WAV as blend source."""
            path = filedialog.askopenfilename(
                parent=ed, title="Open source WAV",
                filetypes=[("WAV files", "*.wav")])
            if not path: return
            try:
                audio, _, _, _ = read_wav(path)
                src_cycle = audio[:cs] if len(audio) >= cs else audio
                layer_src_audio[0] = resample_cycle(src_cycle, cs)
                src_lbl.config(text=os.path.basename(path))
                lay_preview()
            except Exception as ex:
                messagebox.showerror("Error", str(ex), parent=ed)

        tk.Button(src_frame, text="From bank...", command=load_layer_from_bank,
                  font=("Consolas", 9),
                  bg=C["accent"], fg=C["text"],
                  relief="flat", bd=0, padx=8, pady=3).pack(side="left", padx=(8,4))
        tk.Button(src_frame, text="From WAV...", command=load_layer_from_file,
                  font=("Consolas", 9),
                  bg=C["accent"], fg=C["text"],
                  relief="flat", bd=0, padx=8, pady=3).pack(side="left", padx=4)
        src_lbl = tk.Label(src_frame, text="None",
                           font=("Consolas", 8), bg=C["bg"], fg=C["muted"])
        src_lbl.pack(side="left", padx=8)

        # Operator selector
        op_row = tk.Frame(tab_layer, bg=C["bg"])
        op_row.pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(op_row, text="Operator:",
                 font=("Consolas", 9), bg=C["bg"], fg=C["text"]).pack(side="left")
        for name, val in [("Blend","blend"),("Add","add"),("Sub","subtract"),
                          ("Mul","multiply"),("Min","min"),("Max","max")]:
            tk.Radiobutton(op_row, text=name, variable=lay_op_var, value=val,
                           bg=C["bg"], fg=C["text"],
                           selectcolor=C["accent"],
                           activebackground=C["bg"],
                           font=("Consolas", 8)).pack(side="left", padx=3)

        # Mix slider
        mix_row = tk.Frame(tab_layer, bg=C["bg"])
        mix_row.pack(fill="x", padx=8, pady=4)
        tk.Label(mix_row, text="Mix (0=base, 1=source):",
                 font=("Consolas", 9), bg=C["bg"], fg=C["text"]).pack(side="left")
        tk.Scale(mix_row, variable=lay_mix_var, from_=0.0, to=1.0,
                 resolution=0.01, orient="horizontal",
                 bg=C["bg"], fg=C["text"],
                 troughcolor=C["accent"],
                 highlightthickness=0, length=220).pack(side="left", padx=8)

        # Layer preview canvas
        lay_cv = tk.Canvas(tab_layer, bg=C["panel"], height=120,
                           highlightthickness=0)
        lay_cv.pack(fill="x", padx=8, pady=4)

        def lay_preview(*_):
            lay_cv.delete("all")
            w2, h2 = lay_cv.winfo_width(), lay_cv.winfo_height()
            if w2 < 10: return
            # Draw base (gray)
            base_arr = np.array(buf[0], dtype=np.float32)
            for col, arr, lw in [(C["muted"], base_arr, 1),
                                  (C["wave"],
                                   _compute_layer(base_arr), 2)]:
                pts = []
                for i, v in enumerate(arr):
                    pts.extend([i/max(len(arr)-1,1)*w2,
                                h2//2 - float(v)*(h2//2-4)])
                if len(pts) >= 4:
                    lay_cv.create_line(*pts, fill=col, width=lw)

        def _compute_layer(base):
            src = layer_src_audio[0]
            if src is None: return base
            mix = lay_mix_var.get()
            op  = lay_op_var.get()
            if op == "blend":    r = (1-mix)*base + mix*src
            elif op == "add":    r = base + src*mix
            elif op == "subtract": r = base - src*mix
            elif op == "multiply": r = base*(1-mix+mix*src)
            elif op == "min":    r = np.minimum(base, src*mix+base*(1-mix))
            elif op == "max":    r = np.maximum(base, src*mix+base*(1-mix))
            else:                r = base
            mx = float(np.max(np.abs(r)))
            return r/mx if mx > 1.0 else r

        lay_cv.bind("<Configure>", lambda e: lay_preview())

        def apply_layer():
            src = layer_src_audio[0]
            if src is None:
                messagebox.showinfo("Layer", "Load a source first.", parent=ed)
                return
            base    = np.array(buf[0], dtype=np.float32)
            result  = _compute_layer(base)
            mx      = float(np.max(np.abs(result)))
            if mx > 1.0: result /= mx
            buf[0]  = result.tolist()
            draw_canvas_wave()
            draw_preview()

        tk.Button(tab_layer, text="Apply layer",
                  command=apply_layer,
                  font=("Consolas", 9),
                  bg=C["hot"], fg="#fff",
                  relief="flat", bd=0, padx=10, pady=4).pack(pady=8)

        # Initial draw
        draw_canvas_wave()
        draw_preview()

    def _open_scanner(self):
        """
        Cycle Scanner — separate window.
        Load a WAV, detect fundamental frequency, display all extracted cycles,
        let user pick one (or average several), then inject into current bank.
        """
        sc = tk.Toplevel(self)
        sc.title("Cycle Scanner")
        sc.configure(bg=C["bg"])
        sc.geometry("900x680")
        sc.resizable(True, True)

        # State
        scan_audio   = [None]
        scan_sr      = [44100]
        scan_cycles  = [[]]    # list of cycle dicts
        scan_freq    = [0.0]
        selected_idx = [set()]

        # ── Top: file + freq controls ─────────────────────────────────────
        top = tk.Frame(sc, bg=C["panel"], pady=6)
        top.pack(fill="x", padx=0)

        tk.Label(top, text="Source WAV:",
                 font=("Consolas", 9), bg=C["panel"], fg=C["text"]).pack(
                     side="left", padx=(10, 4))
        file_lbl = tk.Label(top, text="No file",
                            font=("Consolas", 9), bg=C["panel"], fg=C["wave"],
                            width=24)
        file_lbl.pack(side="left")

        def open_source():
            path = filedialog.askopenfilename(
                parent=sc, title="Open source WAV",
                filetypes=[("WAV files", "*.wav")])
            if not path: return
            try:
                audio, sr, _, _ = read_wav(path)
                scan_audio[0] = audio
                scan_sr[0]    = sr
                file_lbl.config(text=os.path.basename(path))
                run_detection()
            except Exception as ex:
                messagebox.showerror("Error", str(ex), parent=sc)

        tk.Button(top, text="Open WAV...", command=open_source,
                  font=("Consolas", 9),
                  bg=C["accent"], fg=C["text"],
                  relief="flat", bd=0, padx=8, pady=3).pack(
                      side="left", padx=8)

        tk.Label(top, text="Freq (Hz):",
                 font=("Consolas", 9), bg=C["panel"], fg=C["text"]).pack(
                     side="left", padx=(8, 2))
        freq_var = tk.DoubleVar(value=440.0)
        freq_entry = tk.Entry(top, textvariable=freq_var, width=8,
                              font=("Consolas", 9),
                              bg=C["panel"], fg=C["text"],
                              relief="flat")
        freq_entry.pack(side="left")
        freq_entry.bind("<Return>",
                        lambda e: run_detection(manual_freq=freq_var.get()))
        note_lbl = tk.Label(top, text="",
                            font=("Consolas", 8), bg=C["panel"], fg=C["muted"])
        note_lbl.pack(side="left", padx=4)

        tk.Button(top, text="Re-scan", command=lambda: run_detection(
                    manual_freq=freq_var.get()),
                  font=("Consolas", 9),
                  bg=C["accent"], fg=C["text"],
                  relief="flat", bd=0, padx=8, pady=3).pack(
                      side="left", padx=8)

        # ── Overview canvas: full waveform with cycle markers ─────────────
        tk.Label(sc, text="Full sample — click a cycle marker to select/deselect",
                 font=("Consolas", 8), bg=C["bg"], fg=C["muted"]).pack(
                     anchor="w", padx=10, pady=(6, 1))
        overview_cv = tk.Canvas(sc, bg=C["panel"], height=100,
                                highlightthickness=0)
        overview_cv.pack(fill="x", padx=10, pady=(0, 4))

        # ── Detail canvas: selected cycle zoomed in ───────────────────────
        tk.Label(sc, text="Selected cycle preview",
                 font=("Consolas", 8), bg=C["bg"], fg=C["muted"]).pack(
                     anchor="w", padx=10)
        detail_cv = tk.Canvas(sc, bg=C["panel"], height=180,
                              highlightthickness=0)
        detail_cv.pack(fill="x", padx=10, pady=(0, 4))

        # ── Info + controls ───────────────────────────────────────────────
        info_row = tk.Frame(sc, bg=C["bg"])
        info_row.pack(fill="x", padx=10, pady=4)
        info_lbl = tk.Label(info_row, text="",
                            font=("Consolas", 9), bg=C["bg"], fg=C["text"])
        info_lbl.pack(side="left")

        bot = tk.Frame(sc, bg=C["bg"])
        bot.pack(fill="x", padx=10, pady=6)

        tk.Label(bot, text="Export cycle size:",
                 font=("Consolas", 9), bg=C["bg"], fg=C["text"]).pack(side="left")
        out_cs_var = tk.IntVar(value=self.export_size_var.get())
        ttk.Combobox(bot, textvariable=out_cs_var,
                     values=EXPORT_SIZES, state="readonly", width=6,
                     font=("Consolas", 9)).pack(side="left", padx=(4, 16))

        def add_selected():
            """Add selected cycles (or average) to the current bank."""
            sel = sorted(selected_idx[0])
            if not sel:
                messagebox.showinfo("Scanner",
                                    "Select at least one cycle first.", parent=sc)
                return
            target = out_cs_var.get()
            cycles_to_add = [scan_cycles[0][i]["audio"] for i in sel]
            if len(cycles_to_add) == 1:
                final = resample_cycle(cycles_to_add[0], target)
            else:
                # Average selected cycles (all resampled to target first)
                resampled = [resample_cycle(c, target) for c in cycles_to_add]
                final     = np.mean(resampled, axis=0).astype(np.float32)
            # Ensure bank exists
            if not self.bank:
                b = Bank("scanned.wav", final.copy(), scan_sr[0], 16, {})
                b.slice(target)
                self.banks    = [b]
                self.bank_idx = 0
                self._set_mode("file")
            else:
                self.bank.cycles.append(final)
                self.bank.audio = np.concatenate(self.bank.cycles)
            self.cycle_idx = len(self.bank.cycles) - 1
            self._activate(self.bank_idx)
            messagebox.showinfo(
                "Scanner",
                f"Added {'average of ' if len(sel)>1 else ''}"
                f"{len(sel)} cycle(s) → {target} samples",
                parent=sc)

        def select_best():
            if not scan_cycles[0]: return
            best = max(scan_cycles[0], key=lambda c: c["stability"])
            selected_idx[0] = {best["index"]}
            draw_overview(); draw_detail()
            freq_info = f"{scan_freq[0]:.1f} Hz  |  {len(scan_cycles[0])} cycles  |  period: {scan_sr[0]/scan_freq[0]:.1f} samp"
            info_lbl.config(text=freq_info + f"  |  Best: cycle {best['index']+1} (stab {best['stability']:.3f})")

        tk.Button(bot, text="Select best", command=select_best,
                  font=("Consolas", 9), bg=C["accent"], fg=C["text"],
                  relief="flat", bd=0, padx=8, pady=4).pack(side="left", padx=(0,8))

        tk.Button(bot, text="Add to bank",
                  command=add_selected,
                  font=("Consolas", 9),
                  bg=C["hot"], fg="#fff",
                  relief="flat", bd=0, padx=10, pady=4).pack(side="left")

        # ── Drawing helpers ───────────────────────────────────────────────
        def draw_overview():
            overview_cv.delete("all")
            audio = scan_audio[0]
            if audio is None: return
            w, h = overview_cv.winfo_width(), overview_cv.winfo_height()
            if w < 10: return
            n = len(audio)
            # Waveform (downsampled for speed)
            step = max(1, n // w)
            pts  = []
            for i in range(0, n, step):
                x = int(i / n * w)
                y = int(h//2 - float(audio[i]) * (h//2 - 4))
                pts.extend([x, y])
            if len(pts) >= 4:
                overview_cv.create_line(*pts, fill=C["muted"], width=1)
            # Cycle markers
            for cyc in scan_cycles[0]:
                x  = int(cyc["start"] / n * w)
                x2 = int(cyc["end"]   / n * w)
                is_sel = cyc["index"] in selected_idx[0]
                color  = "#00bcd4" if is_sel else C["wave"]
                overview_cv.create_line(x, 0, x, h,
                                        fill=color, width=1,
                                        dash=() if is_sel else (3,3))
                # Stability indicator: height of small rect
                sh = int(cyc["stability"] * (h//4))
                overview_cv.create_rectangle(x, h-sh, x2, h,
                                             fill=color, outline="",
                                             stipple="gray50" if not is_sel else "")

        def draw_detail():
            detail_cv.delete("all")
            sel = sorted(selected_idx[0])
            if not sel or not scan_cycles[0]: return
            w, h = detail_cv.winfo_width(), detail_cv.winfo_height()
            if w < 10: return
            # Show average if multiple selected, else single cycle
            cyc_list = [scan_cycles[0][i]["audio"] for i in sel]
            if len(cyc_list) > 1:
                max_len = max(len(c) for c in cyc_list)
                resampled = [resample_cycle(c, max_len) for c in cyc_list]
                data = np.mean(resampled, axis=0)
            else:
                data = cyc_list[0]
            # Draw zero line
            detail_cv.create_line(0, h//2, w, h//2,
                                  fill=C["muted"], dash=(3,3))
            pts = []
            for i, v in enumerate(data):
                pts.extend([i / max(len(data)-1, 1) * w,
                            h//2 - float(v) * (h//2 - 6)])
            if len(pts) >= 4:
                detail_cv.create_line(*pts, fill=C["wave"], width=1.5)
            # Info
            disc = boundary_discontinuity(data)
            disc_col = "#c0392b" if disc > 0.20 else (
                       "#e67e22" if disc > 0.05 else "#2ecc71")
            periodic_ok = disc < 0.05
            periodic_txt = "periodic OK" if periodic_ok else (
                           "slight discontinuity" if disc < 0.20 else "NOT periodic")
            detail_cv.create_text(8, 8,
                text=f"{len(sel)} cycle(s)  |  disc={disc:.3f}  |  {periodic_txt}",
                font=("Consolas", 8), fill=disc_col, anchor="nw")

        def on_overview_click(event):
            audio = scan_audio[0]
            if audio is None: return
            w  = overview_cv.winfo_width()
            n  = len(audio)
            sample_pos = int(event.x / w * n)
            # Find nearest cycle
            best_i, best_d = 0, n
            for cyc in scan_cycles[0]:
                d = abs(cyc["start"] - sample_pos)
                if d < best_d:
                    best_d, best_i = d, cyc["index"]
            # Toggle selection
            if best_i in selected_idx[0]:
                selected_idx[0].discard(best_i)
            else:
                selected_idx[0].add(best_i)
            draw_overview()
            draw_detail()
            # Update info — keep base freq info, append selection details
            sel = sorted(selected_idx[0])
            freq_info = f"{scan_freq[0]:.1f} Hz  |  {len(scan_cycles[0])} cycles  |  period: {scan_sr[0]/scan_freq[0]:.1f} samp"
            if sel:
                stabs = [scan_cycles[0][i]["stability"] for i in sel]
                sel_info = f"  |  Selected: {[s+1 for s in sel]}  |  avg stability: {np.mean(stabs):.3f}"
                info_lbl.config(text=freq_info + sel_info)

        overview_cv.bind("<Button-1>", on_overview_click)
        overview_cv.bind("<Configure>", lambda e: draw_overview())
        detail_cv.bind("<Configure>", lambda e: draw_detail())

        def run_detection(manual_freq=None):
            audio = scan_audio[0]
            if audio is None: return
            sr    = scan_sr[0]
            if manual_freq and manual_freq > 0:
                freq = float(manual_freq)
            else:
                freq = detect_fundamental(audio, sr)
            scan_freq[0] = freq
            freq_var.set(round(freq, 2))
            note_lbl.config(text=freq_to_note(freq))
            cycles = extract_cycles_from_audio(audio, sr, freq)
            scan_cycles[0] = cycles
            selected_idx[0].clear()
            # Auto-select the most stable cycle (not necessarily the first)
            if cycles:
                best = max(cycles, key=lambda c: c["stability"])
                selected_idx[0].add(best["index"])
                # Show best cycle info in detail
                info_lbl.config(
                    text=f"Detected: {freq:.1f} Hz  |  {len(cycles)} cycles  |  "
                         f"period: {sr/freq:.1f} samples  |  "
                         f"Best: cycle {best['index']+1} (stability {best['stability']:.3f})")
            info_lbl.config(
                text=f"Detected: {freq:.1f} Hz  |  "
                     f"{len(cycles)} cycles  |  "
                     f"period: {sr/freq:.1f} samples")
            draw_overview()
            draw_detail()

    def spectral_coherence_bank(self):
        """Return spectral_coherence result for current bank cycles."""
        return spectral_coherence(self.cycles) if self.cycles else {}

    def _on_morph(self, val=None):
        """Interpolate between cycle[idx] and cycle[idx+1] for live preview."""
        b = self.bank
        if not b or len(b.cycles) < 2:
            if self.morph_lbl: self.morph_lbl.config(text="Need ≥2 cycles")
            return
        t     = float(self.morph_var.get())
        idx_a = self.cycle_idx % len(b.cycles)
        idx_b = (self.cycle_idx + 1) % len(b.cycles)
        ca, cb = b.cycles[idx_a], b.cycles[idx_b]
        n = max(len(ca), len(cb))
        if len(ca) != n: ca = resample_cycle(ca, n)
        if len(cb) != n: cb = resample_cycle(cb, n)
        morphed = ((1-t)*ca + t*cb).astype(np.float32)
        self.morph_lbl.config(text=f"C{idx_a+1}←{t:.2f}→C{idx_b+1}")
        # Draw morphed waveform (purple)
        cv = self.wave_cv
        cv.delete("all")
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 10: return
        lp,rp,tp,bp = 32,8,6,18; dw=w-lp-rp; dh=h-tp-bp
        for vv,ll in [(-1.,"-1"),(-.5,"-.5"),(0.,"0"),(.5,"+.5"),(1.,"+1")]:
            y = tp+(1.-(vv+1)/2)*dh
            cv.create_line(lp,y,w-rp,y,fill=C["grid"] if vv!=0 else C["muted"],
                           dash=(4,4) if vv!=0 else ())
            cv.create_text(lp-3,y,text=ll,font=("Consolas",7),fill=C["muted"],anchor="e")
        pts=[]
        for i,v in enumerate(morphed):
            pts.extend([lp+i/max(len(morphed)-1,1)*dw, tp+(1.-(float(v)+1)/2)*dh])
        if len(pts)>=4: cv.create_line(*pts, fill="#ce93d8", width=1.5)
        # Cache morphed cycle for play
        self._morph_cached = morphed
        # Compute morph coherence vs bank mean
        if len(self.cycles) >= 2:
            result   = spectral_coherence(self.cycles)
            mean_p   = result["mean_profile"]
            m_prof   = extract_harmonics(morphed, 16)
            norm_a   = float(np.linalg.norm(m_prof))
            norm_b   = float(np.linalg.norm(mean_p))
            m_score  = float(np.dot(m_prof, mean_p)/(norm_a*norm_b+1e-10)) if norm_a*norm_b > 0 else 0.0
            self._draw_coherence(morph_score=m_score)
        # Draw morphed FFT
        _,fft_m = classify_cycle(morphed)
        fc=self.fft_cv; fc.delete("all")
        fw,fh=fc.winfo_width(),fc.winfo_height()
        if fw>10:
            lp2,bp2,tp2=26,18,6; dh2=fh-tp2-bp2
            slot=(fw-lp2)/max(len(fft_m),1); bw2=max(4,int(slot*.65))
            for vv,ll in [(0.,"0"),(.5,".5"),(1.,"1")]:
                yy=tp2+(1.-vv)*dh2
                fc.create_line(lp2,yy,fw,yy,fill=C["grid"],dash=(3,3))
                fc.create_text(lp2-3,yy,text=ll,font=("Consolas",7),fill=C["muted"],anchor="e")
            lbls=["F","2","3","4","5","6","7","8","9","10","11","12"]
            for ii in range(min(len(fft_m),12)):
                bh2=int(float(fft_m[ii])*dh2); xx=int(lp2+ii*slot+(slot-bw2)/2)
                fc.create_rectangle(xx,tp2+dh2-bh2,xx+bw2,tp2+dh2,
                                    fill="#ce93d8" if ii==0 else C["fft"],outline="")
                fc.create_text(xx+bw2//2,fh-4,text=lbls[ii],font=("Consolas",7),fill=C["muted"])

    def _draw_coherence(self, morph_score: float = None):
        """
        Draw two rows in the coherence canvas:
          Top row (h/2)  : per-cycle bars, current cycle highlighted
          Bottom row     : morph gradient — horizontal blend from cyan to color
                           at current morph position, with score label
        """
        cv = self.coh_cv
        cv.delete("all")
        if not self.cycles: return
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 10: return
        row_h = h // 2
        # ── Top row: per-cycle bars ──────────────────────────────────────────
        if len(self.cycles) >= 2:
            result = spectral_coherence(self.cycles)
            scores = result["per_cycle"]
            glob   = result["global"]
            n      = len(scores)
            gc = "#2ecc71" if glob>0.95 else ("#e67e22" if glob>0.85 else "#c0392b")
            for i, score in enumerate(scores):
                x  = int(i * w / n)
                bw = max(2, int(w/n) - 2)
                bh = int(score * (row_h - 4))
                is_cur = (i == self.cycle_idx)
                col = "#00bcd4" if is_cur else (
                      "#2ecc71" if score>0.95 else
                      "#e67e22" if score>0.85 else "#c0392b")
                cv.create_rectangle(x, row_h-bh, x+bw, row_h, fill=col, outline="")
                if is_cur:
                    cv.create_text(x+bw//2, row_h-bh-2, text=f"{score:.2f}",
                                   font=("Consolas",6), fill="#00bcd4", anchor="s")
            self.coh_global_lbl.config(text=f"global: {glob:.3f}", fg=gc)
             # ── Bottom row: full-bank morph coherence gradient ────────────────────────
        if len(self.cycles) >= 2:
            path = build_morph_coherence_path(self.cycles, n_steps=max(w,2))
            fh2  = h - row_h - 1
            for x_px in range(w):
                score_px = float(path[min(x_px, len(path)-1)])
                r   = int(min(255, max(0, int((1-score_px)*2*255))))
                g   = int(min(255, max(0, int(score_px*2*255))))
                col = f"#{r:02x}{g:02x}40"
                bh2 = max(1, int(score_px * fh2))
                cv.create_line(x_px, h-bh2, x_px, h, fill=col)
            # White cursor = global morph position
            gpos = float(self.global_morph_var.get()) if hasattr(self,'global_morph_var') else 0.0
            xm   = int(gpos * (w-1))
            cv.create_line(xm, row_h+1, xm, h, fill="#ffffff", width=2)
            # Purple cursor = local morph (between current cycle pair)
            t_loc = float(self.morph_var.get()) if self.morph_var else 0.0
            n_c   = len(self.cycles)
            x_loc = int((self.cycle_idx + t_loc) / max(n_c-1,1) * (w-1))
            cv.create_line(x_loc, row_h+1, x_loc, h, fill="#ce93d8", width=1)
            if morph_score is not None:
                mc = "#2ecc71" if morph_score>0.95 else ("#e67e22" if morph_score>0.85 else "#c0392b")
                self.morph_coh_lbl.config(text=f"Morph:{morph_score:.3f}", fg=mc)
        else:
            cv.create_text(4, row_h+2, text="← load ≥2 cycles for morph path",
                           font=("Consolas",6), fill=C["muted"], anchor="nw")
            self.morph_coh_lbl.config(text="", fg=C["muted"])
    def _set_view_mode(self, mode: str):
        """Switch the main view mode and update tab button highlights."""
        self._view_mode = mode
        for m, btn in self._view_btns.items():
            btn.config(bg=C["hot"] if m == mode else C["accent"])
        self._refresh_view()

    def _refresh_view(self):
        """Redraw left canvas per view mode. FFT (right) is always redrawn."""
        mode = self._view_mode
        if mode == "waveform":
            self._draw_wave()
        elif mode == "fft":
            self.wave_cv.delete("all")
            self.wave_cv.create_text(10, 10, text="FFT always shown →",
                font=("Consolas", 8), fill=C["muted"], anchor="nw")
        elif mode == "heatmap":
            self._draw_heatmap()
        elif mode == "harmonic_lines":
            self._draw_harmonic_lines()
        self._draw_fft()
        if self._show_overlay_var.get() and self._selected_cycles:
            self._draw_overlay()

    def _toggle_cycle_selection(self, idx: int):
        """Toggle idx in multi-selection. Auto-enables overlay when non-empty."""
        if idx in self._selected_cycles:
            self._selected_cycles.discard(idx)
        else:
            self._selected_cycles.add(idx)
        if self._selected_cycles:
            self._show_overlay_var.set(True)
        self._build_thumbs()
        self._refresh_view()

    def _draw_overlay(self):
        """Draw selected cycles as colored overlays on the wave canvas."""
        if not self._selected_cycles or not self.cycles:
            return
        cv = self.wave_cv
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 10: return
        lpad, rpad, tpad, bpad = 32, 8, 6, 18
        dw, dh = w-lpad-rpad, h-tpad-bpad
        # Color palette for overlays
        palette = ["#ff6b6b","#ffd93d","#6bcb77","#4d96ff",
                   "#c77dff","#f4845f","#48cae4","#e9c46a"]
        legend_y = tpad + 4
        for i, sel_idx in enumerate(sorted(self._selected_cycles)):
            if sel_idx >= len(self.cycles): continue
            cyc   = self.cycles[sel_idx]
            color = palette[i % len(palette)]
            # Apply zoom
            zs = max(0, self._zoom_start)
            ze = len(cyc) if self._zoom_end < 0 else min(self._zoom_end, len(cyc))
            view = cyc[zs:ze]
            pts  = []
            for j, v in enumerate(view):
                x = lpad + j/max(len(view)-1,1)*dw
                y = tpad + (1.-(float(v)+1)/2)*dh
                pts.extend([x, y])
            if len(pts) >= 4:
                cv.create_line(*pts, fill=color, width=1, dash=(4,2))
            # Legend on waveform canvas
            if self._show_legend_var.get():
                label, _ = classify_cycle(cyc)
                cv.create_rectangle(lpad+4, legend_y, lpad+14, legend_y+8,
                                    fill=color, outline="")
                cv.create_text(lpad+18, legend_y+4,
                               text=f"C{sel_idx+1} {label}",
                               font=("Consolas",7), fill=color, anchor="w")
                legend_y += 12
            # FFT overlay — thin colored bars on FFT canvas
            _, fft_sel = classify_cycle(cyc)
            fc = self.fft_cv
            fw, fh = fc.winfo_width(), fc.winfo_height()
            if fw < 10: continue
            n_fo   = min(len(fft_sel), 12)
            lp_f   = 26; bp_f = 18; tp_f = 6
            dh_f   = fh - tp_f - bp_f
            slot_f = (fw - lp_f) / max(n_fo, 1)
            bw_f   = max(2, int(slot_f * 0.25))
            for ii in range(n_fo):
                if self._harmonic_filter and ii not in self._harmonic_filter:
                    continue
                bh_f = int(float(fft_sel[ii]) * dh_f)
                xf   = int(lp_f + ii*slot_f + (slot_f-bw_f)/2) + i*3
                fc.create_rectangle(xf, tp_f+dh_f-bh_f, xf+bw_f, tp_f+dh_f,
                                    fill=color, outline="")

    def _draw_heatmap(self):
        """Draw spectral heatmap: cycles (Y) × harmonics (X), color = amplitude."""
        cv = self.wave_cv
        cv.delete("all")
        if not self.cycles:
            return
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 10: return
        n_harm = 16
        hm     = build_heatmap(self.cycles, n_harm)
        n_cyc  = len(hm)
        if n_cyc == 0: return
        cell_w = (w - 36) / n_harm
        cell_h = (h - 16) / n_cyc
        # Draw cells
        for ci in range(n_cyc):
            for hi in range(n_harm):
                val = float(hm[ci, hi])
                # Color: black → blue → cyan → green → yellow → red
                if val < 0.25:
                    r,g,b = 0, 0, int(val*4*255)
                elif val < 0.5:
                    r,g,b = 0, int((val-0.25)*4*255), 255
                elif val < 0.75:
                    r,g,b = 0, 255, int((1-(val-0.5)*4)*255)
                else:
                    r,g,b = int((val-0.75)*4*255), 255, 0
                col = f"#{r:02x}{g:02x}{b:02x}"
                x1 = 36 + hi * cell_w
                y1 = ci * cell_h
                cv.create_rectangle(x1, y1, x1+cell_w-1, y1+cell_h-1,
                                    fill=col, outline="")
            # Y axis: cycle label
            y_mid = ci * cell_h + cell_h/2
            is_cur = (ci == self.cycle_idx)
            cv.create_text(2, y_mid, text=f"C{ci+1}",
                           font=("Consolas", 7),
                           fill="#00bcd4" if is_cur else C["muted"],
                           anchor="w")
        # X axis: harmonic labels
        lbls = ["F","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16"]
        for hi in range(n_harm):
            x_mid = 36 + hi * cell_w + cell_w/2
            cv.create_text(x_mid, h-4, text=lbls[hi],
                           font=("Consolas", 7), fill=C["muted"])

    def _draw_harmonic_lines(self):
        """Draw harmonic evolution curves: one line per harmonic across cycles."""
        cv = self.wave_cv
        cv.delete("all")
        if not self.cycles or len(self.cycles) < 2:
            return
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 10: return
        n_harm = 12
        lpad, rpad, tpad, bpad = 36, 8, 6, 18
        dw, dh = w-lpad-rpad, h-tpad-bpad
        palette = ["#ff6b6b","#ffd93d","#6bcb77","#4d96ff",
                   "#c77dff","#f4845f","#48cae4","#e9c46a",
                   "#ff9f1c","#cbf3f0","#ffbfd3","#a8dadc"]
        # Extract harmonic profiles
        profiles = []
        for c in self.cycles:
            fft  = np.abs(np.fft.rfft(c))
            amps = [float(fft[i+1]) if i+1 < len(fft) else 0.0
                    for i in range(n_harm)]
            profiles.append(amps)
        profiles = np.array(profiles)
        # Normalize each harmonic independently
        for hi in range(n_harm):
            col = profiles[:,hi]
            mn, mx = col.min(), col.max()
            if mx > mn:
                profiles[:,hi] = (col-mn)/(mx-mn)
        # Grid
        for yf in [0.25,0.5,0.75,1.0]:
            y = tpad + (1-yf)*dh
            cv.create_line(lpad, y, w-rpad, y, fill=C["grid"], dash=(2,4))
        # Draw one line per harmonic
        n_cyc = len(self.cycles)
        legend_y = tpad + 2
        lbls = ["H1(F)","H2","H3","H4","H5","H6","H7","H8",
                "H9","H10","H11","H12"]
        for hi in range(n_harm):
            if self._harmonic_filter and hi not in self._harmonic_filter:
                continue
            color = palette[hi % len(palette)]
            pts   = []
            for ci in range(n_cyc):
                x = lpad + ci/max(n_cyc-1,1)*dw
                y = tpad + (1-profiles[ci,hi])*dh
                pts.extend([x,y])
            if len(pts) >= 4:
                cv.create_line(*pts, fill=color, width=1.5, smooth=True)
            # Legend
            if self._show_legend_var.get():
                cv.create_rectangle(lpad+2, legend_y, lpad+10, legend_y+6,
                                    fill=color, outline="")
                cv.create_text(lpad+14, legend_y+3,
                               text=lbls[hi],
                               font=("Consolas",6), fill=color, anchor="w")
                legend_y += 9
        # X axis: cycle numbers
        for ci in range(n_cyc):
            x = lpad + ci/max(n_cyc-1,1)*dw
            cv.create_text(x, h-4, text=str(ci+1),
                           font=("Consolas",7), fill=C["muted"])

    def _on_global_morph(self, val=None):
        """Global morph slider — sets position across the full bank."""
        if not self.cycles or len(self.cycles) < 2:
            return
        t     = float(self.global_morph_var.get())
        n     = len(self.cycles)
        pos   = t * (n - 1)
        idx_a = int(pos)
        idx_b = min(idx_a+1, n-1)
        t_loc = pos - idx_a
        ca, cb = self.cycles[idx_a], self.cycles[idx_b]
        sz = max(len(ca), len(cb))
        if len(ca) != sz: ca = resample_cycle(ca, sz)
        if len(cb) != sz: cb = resample_cycle(cb, sz)
        morphed = ((1-t_loc)*ca + t_loc*cb).astype(np.float32)
        self._morph_cached = morphed
        self.global_morph_lbl.config(
            text=f"pos:{pos:.2f} C{idx_a+1}↔C{idx_b+1}")
        # Sync local morph slider
        if self.morph_var is not None:
            self.cycle_idx = idx_a
            self.morph_var.set(t_loc)
        self._draw_coherence()
        # Refresh waveform with morphed
        self._on_morph()

    def _on_pan_wave(self, event):
        """Pan the zoom window left/right by dragging on the oscilloscope."""
        dx = event.x - getattr(self, '_pan_last_x', event.x)
        self._pan_last_x = event.x
        if not self.cycles: return
        n = len(self.cycles[self.cycle_idx])
        zs = self._zoom_start
        ze = n if self._zoom_end < 0 else self._zoom_end
        span = ze - zs
        w = self.wave_cv.winfo_width()
        samp_per_px = span / max(w, 1)
        delta = int(-dx * samp_per_px)
        new_zs = max(0, zs + delta)
        new_ze = new_zs + span
        if new_ze > n:
            new_ze = n
            new_zs = max(0, n - span)
        self._zoom_start = new_zs
        self._zoom_end   = new_ze
        self._draw_wave()

    def _on_fft_click(self, event):
        """Click on FFT bar to toggle harmonic in filter (for Lines mode)."""
        w = self.fft_cv.winfo_width()
        if w < 10 or not self.cycles: return
        _, fft = classify_cycle(self.cycles[self.cycle_idx])
        n      = min(len(fft), 12)
        lp2    = 26
        slot   = (w - lp2) / max(n, 1)
        h_idx  = int((event.x - lp2) / max(slot, 1))
        if 0 <= h_idx < n:
            if h_idx in self._harmonic_filter:
                self._harmonic_filter.discard(h_idx)
            else:
                self._harmonic_filter.add(h_idx)
            if self._harmonic_filter:
                self.fft_filter_lbl.config(
                    text="H:" + ",".join(str(i+1) for i in sorted(self._harmonic_filter)))
            else:
                self.fft_filter_lbl.config(text="")
            self._draw_fft()
            if self._view_mode == "harmonic_lines":
                self._draw_harmonic_lines()

    def _bake_morph(self):
        """Add the current morphed waveform as a new cycle in the bank."""
        if self._morph_cached is None:
            self.status_var.set("Move the morph slider first.")
            return
        if not self.bank:
            return
        self._push_undo()
        target = self.export_size_var.get()
        baked  = resample_cycle(self._morph_cached, target)
        self.bank.cycles.append(baked)
        self.bank.audio = np.concatenate(self.bank.cycles)
        self.cycle_idx  = len(self.bank.cycles) - 1
        self._morph_cached = None
        self.morph_var.set(0.0)
        self._refresh()
        self.status_var.set(f"Morph baked → cycle {self.cycle_idx+1}")

    def _get_play_audio(self) -> tuple:
        """Return (audio_array, sr) for playback, using morph cache if active."""
        b = self.bank
        if b is None:
            return None, 44100
        sr     = b.sr
        freq   = float(self.play_freq_var.get()) if hasattr(self, 'play_freq_var') else 440.0
        # Use morphed cycle if slider is active
        if self._morph_cached is not None and float(self.morph_var.get()) > 0:
            cycle = self._morph_cached
        elif self.cycles:
            cycle = self.cycles[self.cycle_idx]
        else:
            return None, sr
        # Resample cycle to match the desired frequency
        # target_samples = sr / freq
        target_samples = max(8, int(round(sr / freq)))
        c = resample_cycle(cycle, target_samples)
        # Tile to ~2 seconds
        n_rep = max(1, int(sr * 2 / len(c)))
        audio = np.tile(c, n_rep + 1)[:sr * 2].astype(np.float32)
        return audio, sr

    def _play_loop(self):
        """Start looped playback. Uses winsound on Windows."""
        if self._loop_running:
            return
        self._loop_running = True
        import tempfile, threading
        audio, sr = self._get_play_audio()
        if audio is None:
            self._loop_running = False
            return
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        write_wav_plain(tmp_path, audio, sr)
        def _loop():
            try:
                import winsound
                while self._loop_running:
                    winsound.PlaySound(tmp_path,
                                       winsound.SND_FILENAME | winsound.SND_NODEFAULT)
            except Exception:
                pass
            finally:
                try: os.unlink(tmp_path)
                except Exception: pass
        self._loop_thread = threading.Thread(target=_loop, daemon=True)
        self._loop_thread.start()
        self.status_var.set("Looping — press ■ Stop to end.")

    def _stop_play(self):
        """Stop looped playback."""
        self._loop_running = False
        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
        self.status_var.set("Playback stopped.")

    def _on_freq_change(self, *_):
        """Update frequency label when slider moves."""
        try:
            freq = float(self.play_freq_var.get())
            import math
            note = freq_to_note(freq)
            self.play_freq_lbl.config(text=f"{freq:.0f}Hz {note}")
        except Exception:
            pass

    def _on_phase_slider(self, val):
        """Phase slider moved — apply shift relative to last slider position."""
        new_val = int(float(val))
        old_val = getattr(self, '_phase_slider_last', 0)
        delta   = new_val - old_val
        if delta != 0 and self.bank and self.bank.cycles:
            self.bank.cycles[self.cycle_idx] = shift_phase(
                self.bank.cycles[self.cycle_idx], delta)
            cur = self.phase_offset_var.get() + delta
            cs  = self.bank.cycle_size
            self.phase_offset_var.set(cur % cs if cur >= 0 else -((-cur) % cs))
            self._phase_slider_last = new_val
            self._draw_wave()
            self._draw_fft()

    def _push_undo(self):
        b = self.bank
        if not b: return
        self._undo_stack.append(
            (self.bank_idx, self.cycle_idx, [c.copy() for c in b.cycles]))
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    def _undo(self, event=None):
        if not self._undo_stack:
            self.status_var.set("Nothing to undo."); return
        bidx, cidx, snap = self._undo_stack.pop()
        if 0 <= bidx < len(self.banks):
            b = self.banks[bidx]
            b.cycles = snap
            b.audio  = np.concatenate(snap) if snap else np.zeros(b.cycle_size, dtype=np.float32)
            self.bank_idx  = bidx
            self.cycle_idx = min(cidx, max(0, len(snap)-1))
            self._refresh()
            self.status_var.set(f"Undo — {len(self._undo_stack)} step(s) left.")

    def _zoom_in(self):
        if not self.cycles: return
        n    = len(self.cycles[self.cycle_idx])
        z_s  = self._zoom_start
        z_e  = n if self._zoom_end < 0 else self._zoom_end
        mid  = (z_s + z_e) // 2
        half = max(8, (z_e - z_s) // 4)
        self._zoom_start = max(0, mid - half)
        self._zoom_end   = min(n, mid + half)
        self._draw_wave()

    def _zoom_out(self):
        if not self.cycles: return
        n   = len(self.cycles[self.cycle_idx])
        z_s = self._zoom_start
        z_e = n if self._zoom_end < 0 else self._zoom_end
        mid = (z_s + z_e) // 2
        half = (z_e - z_s)
        self._zoom_start = max(0, mid - half)
        self._zoom_end   = min(n, mid + half)
        if self._zoom_start == 0 and self._zoom_end >= n:
            self._zoom_end = -1
        self._draw_wave()

    def _zoom_reset(self):
        self._zoom_start = 0
        self._zoom_end   = -1
        self._draw_wave()

    def _zoom_scroll(self, delta: int):
        """Zoom in/out via mouse wheel on the oscilloscope."""
        if delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()

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
        # Reset zoom when cycle changes
        self._zoom_start = 0
        self._zoom_end   = -1
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
        self._draw_coherence()
        if self.morph_var is not None:
            self.morph_var.set(0.0)
            if len(self.cycles) >= 2:
                ib = (self.cycle_idx+1)%len(self.cycles)
                self.morph_lbl.config(
                    text=f"C{self.cycle_idx+1}←0.00→C{ib+1}")

    def _draw_wave(self):
        cv = self.wave_cv
        cv.delete("all")
        if not self.cycles:
            return
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 10 or h < 10:
            return
        lpad, rpad, tpad, bpad = 32, 8, 6, 18
        dw = w - lpad - rpad
        dh = h - tpad - bpad
        # Y axis grid + labels
        for val, label in [(-1.0, "-1"), (-0.5, "-.5"), (0.0, "0"),
                           (0.5, "+.5"), (1.0, "+1")]:
            y = tpad + (1.0 - (val + 1) / 2) * dh
            cv.create_line(lpad, y, w - rpad, y,
                           fill=C["grid"] if val != 0 else C["muted"],
                           dash=(4, 4) if val != 0 else ())
            cv.create_text(lpad - 3, y, text=label,
                           font=("Consolas", 7), fill=C["muted"], anchor="e")
        # Get current cycle
        s       = self.cycles[self.cycle_idx]
        n_total = len(s)
        # Apply zoom
        z_start = max(0, self._zoom_start)
        z_end   = n_total if self._zoom_end < 0 else min(self._zoom_end, n_total)
        z_end   = max(z_start + 4, z_end)
        s_view  = s[z_start:z_end]
        n_view  = len(s_view)
        # X axis ticks
        n_ticks = min(8, n_view)
        step    = max(1, n_view // max(n_ticks, 1))
        for i in range(0, n_view, step):
            x = lpad + (i / max(n_view - 1, 1)) * dw
            cv.create_line(x, h - bpad, x, h - bpad + 3, fill=C["muted"])
            cv.create_text(x, h - 2, text=str(z_start + i),
                           font=("Consolas", 7), fill=C["muted"], anchor="s")
        # Waveform line
        pts = []
        for i, v in enumerate(s_view):
            x = lpad + (i / max(n_view - 1, 1)) * dw
            y = tpad + (1.0 - (float(v) + 1) / 2) * dh
            pts.extend([x, y])
        if len(pts) >= 4:
            cv.create_line(*pts, fill=C["wave"], width=1.5, smooth=False)
        # Phase continuity markers
        disc = boundary_discontinuity(s)
        mc   = "#c0392b" if disc > 0.20 else ("#e67e22" if disc > 0.05 else "#2ecc71")
        cv.create_line(lpad, tpad, lpad, tpad + dh, fill=mc, width=2)
        cv.create_line(w - rpad, tpad, w - rpad, tpad + dh, fill=mc, width=2)
        if disc > 0.01:
            cv.create_text(lpad + 4, tpad + 4, text=f"disc={disc:.2f}",
                           font=("Consolas", 7), fill=mc, anchor="nw")
        # Draw overlay if enabled
        if getattr(self, '_show_overlay_var', None) and self._show_overlay_var.get():
            self._draw_overlay()


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
            bh       = int(float(fft[i]) * dh)
            x        = int(lpad + i * slot + (slot - bw) / 2)
            filtered = bool(self._harmonic_filter) and i not in self._harmonic_filter
            bar_col  = C["grid"] if filtered else (C["hot"] if i==0 else C["fft"])
            cv.create_rectangle(x, tpad + dh - bh, x + bw, tpad + dh,
                                fill=bar_col, outline="")
            cv.create_text(x + bw // 2, h - 4, text=lbls[i],
                           font=("Consolas", 7),
                           fill=C["muted"] if filtered else C["text"])

    def _build_thumbs(self):
        for w in self.thumb_frame.winfo_children():
            w.destroy()
        for i, cyc in enumerate(self.cycles):
            label, _ = classify_cycle(cyc)
            disc      = boundary_discontinuity(cyc)
            # Border: active=hot, phase-warning=amber, phase-bad=red, normal=panel
            is_selected = i in self._selected_cycles
            if i == self.cycle_idx:
                border = "#00bcd4"   # cyan — active cycle
            elif is_selected:
                border = "#ffd93d"   # yellow — selected for overlay
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
            th.bind("<Control-Button-1>",
                    lambda e, idx=idx: self._toggle_cycle_selection(idx))
            th.bind("<Shift-Button-1>",
                    lambda e, idx=idx: self._toggle_cycle_selection(idx))
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




if __name__ == "__main__":
    App().mainloop()
