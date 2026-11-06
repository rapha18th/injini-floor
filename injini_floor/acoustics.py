"""Injini acoustics: enrol a machine from healthy clips, score new clips, explain the score.

The recipe is the one Injini validated on DCASE Task 2: a frozen AudioSet-pretrained
embedder (EfficientAT mn10_as, 960-d) and a whitened distance in embedding space.
Here the distance is Mahalanobis under a Ledoit-Wolf shrinkage covariance, which stays
well-conditioned with a few dozen windows from two or three phone clips.

Scores are reported in sigma: how far a clip's median window distance sits above the
held-out distances of the machine's own healthy windows.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.covariance import LedoitWolf

from . import features as F

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, os.pardir, "models", "injini_mn10_as_fp32.onnx")

WIN = 200  # frames per window (2 s at 100 frames/s)
HOP = 100  # 1 s hop
N_GROUPS = 8  # occlusion groups of 16 mel bands
THRESHOLD_SIGMA = 3.0
MIN_SECONDS = 4.0

_session: ort.InferenceSession | None = None


def session() -> ort.InferenceSession:
    global _session
    if _session is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        _session = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    return _session


# ── audio in ────────────────────────────────────────────────────────────────


def decode(data: bytes) -> np.ndarray:
    """Any container libsndfile reads (OGG Opus voice notes, WAV, FLAC) to 32 kHz mono."""
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != F.SR:
        g = np.gcd(int(sr), F.SR)
        y = resample_poly(y, F.SR // g, int(sr) // g).astype(np.float32)
    return y


@dataclass
class Quality:
    ok: bool
    seconds: float
    rms_dbfs: float
    clipped: float
    reason: str = ""


def check_quality(y: np.ndarray) -> Quality:
    seconds = len(y) / F.SR
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    dbfs = 20 * np.log10(rms + 1e-12)
    clipped = float(np.mean(np.abs(y) > 0.99))
    if seconds < MIN_SECONDS:
        return Quality(False, seconds, dbfs, clipped, f"Only {seconds:.1f} s. Hold the phone still for 10 seconds.")
    if dbfs < -55:
        return Quality(False, seconds, dbfs, clipped, "Too quiet to hear the machine. Hold the phone closer.")
    if clipped > 0.01:
        return Quality(False, seconds, dbfs, clipped, "The recording is clipping. Move the phone back a little.")
    return Quality(True, seconds, dbfs, clipped)


# ── features and embeddings ─────────────────────────────────────────────────


def logmel(y: np.ndarray) -> np.ndarray:
    lm = F.logmel(y)
    if lm.shape[1] < WIN:
        lm = np.pad(lm, ((0, 0), (0, WIN - lm.shape[1])), mode="edge")
    return lm


def windows(lm: np.ndarray) -> np.ndarray:
    starts = range(0, lm.shape[1] - WIN + 1, HOP)
    return np.stack([lm[:, s : s + WIN] for s in starts])[:, None, :, :].astype(np.float32)


def embed(lm: np.ndarray) -> np.ndarray:
    x = windows(lm)
    out = []
    for i in range(0, len(x), 16):
        out.append(session().run(None, {"logmel": x[i : i + 16]})[0])
    return np.concatenate(out).astype(np.float64)


def power_spectrum_db(y: np.ndarray) -> np.ndarray:
    """Mean log power per FFT bin (513 bins, 31.25 Hz apart)."""
    p = F._stft_power(np.asarray(y, dtype=np.float64))
    return (10 * np.log10(p.mean(axis=1) + 1e-10)).astype(np.float32)


def mel_band_edges_hz() -> np.ndarray:
    mel = lambda f: 1127.0 * np.log1p(f / 700.0)
    imel = lambda m: 700.0 * (np.exp(m / 1127.0) - 1)
    return imel(np.linspace(mel(F.FMIN), mel(F.FMAX), F.N_MELS + 2))


def group_ranges_hz() -> list[tuple[float, float]]:
    edges = mel_band_edges_hz()
    per = F.N_MELS // N_GROUPS
    return [(float(edges[g * per]), float(edges[(g + 1) * per + 1])) for g in range(N_GROUPS)]


# ── the baseline ────────────────────────────────────────────────────────────


@dataclass
class Baseline:
    asset: str
    clip_embeddings: list[np.ndarray]
    band_mean: np.ndarray  # (128,) mean log-mel per band over healthy frames
    spec_mean: np.ndarray  # (513,) mean dB spectrum over healthy clips
    ref_lm: np.ndarray | None = None  # (128, WIN) real healthy frames, the occlusion filler
    created: float = field(default_factory=time.time)
    folds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)  # (mean, precision) per fold
    c_mean: float = 0.0
    c_std: float = 1.0

    @property
    def n_clips(self) -> int:
        return len(self.clip_embeddings)

    @property
    def n_windows(self) -> int:
        return int(sum(len(e) for e in self.clip_embeddings))

    def fit(self) -> Baseline:
        """Leave-one-clip-out models, calibrated on the clips each one never saw.

        New recordings are scored by the same fold models, so a healthy recording and the
        held-out healthy clips are judged on equal terms. The scale is the spread of
        clip-level medians, bootstrapped from the held-out windows.
        """
        parts = _parts(self.clip_embeddings)
        self.folds, held = [], []
        for i, part in enumerate(parts):
            mu, prec = _gaussian(np.concatenate([p for j, p in enumerate(parts) if j != i]))
            self.folds.append((mu, prec))
            held.append(_maha(part, mu, prec))
        h = np.concatenate(held)
        w = max(1, int(np.median([len(e) for e in self.clip_embeddings])))
        boot = np.median(np.random.default_rng(0).choice(h, (4000, w)), axis=1)
        self.c_mean = float(boot.mean())
        self.c_std = float(max(boot.std(), 0.02 * self.c_mean, 1e-6))
        return self

    def distances(self, E: np.ndarray) -> np.ndarray:
        return np.mean([_maha(E, mu, prec) for mu, prec in self.folds], axis=0)

    def sigma(self, E: np.ndarray) -> float:
        return float((np.median(self.distances(E)) - self.c_mean) / self.c_std)

    def window_sigmas(self, E: np.ndarray) -> np.ndarray:
        return (self.distances(E) - self.c_mean) / self.c_std

    # Stored in a Switch room document, so the machine's memory lives in the room.
    def to_json(self) -> str:
        X = np.concatenate(self.clip_embeddings).astype(np.float16)
        return json.dumps(
            {
                "format": "injini-baseline/1",
                "asset": self.asset,
                "created": self.created,
                "clip_sizes": [len(e) for e in self.clip_embeddings],
                "embeddings_f16_b64": base64.b64encode(X.tobytes()).decode(),
                "dim": int(X.shape[1]),
                "band_mean": np.round(self.band_mean, 4).tolist(),
                "spec_mean": np.round(self.spec_mean, 2).tolist(),
                "ref_lm_f16_b64": base64.b64encode(self.ref_lm.astype(np.float16).tobytes()).decode(),
                "threshold_sigma": THRESHOLD_SIGMA,
                "model": "EfficientAT mn10_as (frozen), leave-one-clip-out Ledoit-Wolf Mahalanobis",
            }
        )

    @classmethod
    def from_json(cls, text: str) -> Baseline:
        d = json.loads(text)
        X = np.frombuffer(base64.b64decode(d["embeddings_f16_b64"]), dtype=np.float16)
        X = X.reshape(-1, d["dim"]).astype(np.float64)
        clips, i = [], 0
        for n in d["clip_sizes"]:
            clips.append(X[i : i + n])
            i += n
        return cls(
            asset=d["asset"],
            clip_embeddings=clips,
            band_mean=np.asarray(d["band_mean"], dtype=np.float32),
            spec_mean=np.asarray(d["spec_mean"], dtype=np.float32),
            ref_lm=np.frombuffer(base64.b64decode(d["ref_lm_f16_b64"]), dtype=np.float16)
            .reshape(F.N_MELS, WIN)
            .astype(np.float32),
            created=d.get("created", time.time()),
        ).fit()


def _gaussian(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lw = LedoitWolf().fit(X)
    return lw.location_, lw.precision_


def _maha(E: np.ndarray, mu: np.ndarray, prec: np.ndarray) -> np.ndarray:
    d = E - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, prec, d), 0))


def _parts(clips: list[np.ndarray]) -> list[np.ndarray]:
    """Clips are the folds; a single clip is split into three blocks of windows."""
    if len(clips) >= 2:
        return clips
    X = clips[0]
    return [p for p in np.array_split(X, min(3, len(X))) if len(p)]


@dataclass
class Clip:
    y: np.ndarray
    lm: np.ndarray
    E: np.ndarray
    spec: np.ndarray
    quality: Quality


def analyse(data: bytes) -> Clip:
    y = decode(data)
    q = check_quality(y)
    lm = logmel(y)
    return Clip(y=y, lm=lm, E=embed(lm) if q.ok else np.zeros((0, 960)), spec=power_spectrum_db(y), quality=q)


def enrol(asset: str, clips: list[Clip]) -> Baseline:
    good = [c for c in clips if c.quality.ok]
    if not good:
        raise ValueError("No usable healthy clip to enrol from.")
    band_mean = np.mean(np.concatenate([c.lm for c in good], axis=1), axis=1)
    spec_mean = np.mean(np.stack([c.spec for c in good]), axis=0)
    ref_lm = good[0].lm[:, :WIN].copy()
    return Baseline(asset, [c.E for c in good], band_mean, spec_mean, ref_lm=ref_lm).fit()


# ── scoring and explanation ─────────────────────────────────────────────────


@dataclass
class Verdict:
    asset: str
    sigma: float
    anomalous: bool
    contributions: list[float]  # sigma removed by restoring each band group to baseline
    top_group: int
    top_range_hz: tuple[float, float]
    peak_hz: float
    peak_db: float
    hint: str
    window_sigmas: list[float]
    seconds: float

    def line(self) -> str:
        state = "OUT OF BASELINE" if self.anomalous else "within baseline"
        lo, hi = self.top_range_hz
        return (
            f"{self.asset}: {self.sigma:.1f}σ, {state}. "
            f"Strongest change {lo:,.0f}–{hi:,.0f} Hz"
            + (f", peak +{self.peak_db:.0f} dB at {self.peak_hz:,.0f} Hz" if self.peak_db > 3 else "")
            + f". {self.hint}"
        )


def score(base: Baseline, clip: Clip, explain: bool = True) -> Verdict:
    sig = base.sigma(clip.E)
    wsig = base.window_sigmas(clip.E).tolist()
    ranges = group_ranges_hz()
    contrib = [0.0] * N_GROUPS
    if explain:
        per = F.N_MELS // N_GROUPS
        for g in range(N_GROUPS):
            lm2 = clip.lm.copy()
            rows = slice(g * per, (g + 1) * per)
            reps = int(np.ceil(clip.lm.shape[1] / WIN))
            lm2[rows, :] = np.tile(base.ref_lm[rows, :], (1, reps))[:, : clip.lm.shape[1]]
            contrib[g] = sig - base.sigma(embed(lm2))
    top = int(np.argmax(contrib))
    lo, hi = ranges[top]
    diff = _smooth(clip.spec - base.spec_mean, 3)
    hz = np.arange(len(diff)) * F.SR / F.N_FFT
    band = (hz >= lo) & (hz <= hi)
    k = int(np.argmax(np.where(band, diff, -np.inf)))
    peak_hz, peak_db = float(hz[k]), float(diff[k])
    tonal = _prominence(clip.spec, k) > 6.0
    return Verdict(
        asset=base.asset,
        sigma=sig,
        anomalous=sig >= THRESHOLD_SIGMA,
        contributions=contrib,
        top_group=top,
        top_range_hz=(lo, hi),
        peak_hz=peak_hz,
        peak_db=peak_db,
        hint=_hint(sig, lo, hi, peak_hz, tonal) if sig >= THRESHOLD_SIGMA else "Sounds like its healthy self.",
        window_sigmas=wsig,
        seconds=clip.quality.seconds,
    )


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    return np.convolve(x, np.ones(k) / k, mode="same")


def _prominence(spec_db: np.ndarray, k: int, width: int = 12) -> float:
    lo, hi = max(0, k - width), min(len(spec_db), k + width + 1)
    neighbours = np.concatenate([spec_db[lo : max(lo, k - 2)], spec_db[k + 3 : hi]])
    return float(spec_db[k] - np.median(neighbours)) if len(neighbours) else 0.0


def _hint(sig: float, lo: float, hi: float, peak_hz: float, tonal: bool) -> str:
    centre = (lo + hi) / 2
    if centre < 300:
        return "Hint: low-frequency rise. Check for looseness, imbalance or a worn mounting."
    if tonal and centre < 6000:
        return f"Hint: a new tone near {peak_hz:,.0f} Hz. Check for rubbing, a blade or belt fault, or gear mesh."
    if centre < 2000:
        return "Hint: broadband rise in the mid range. Check airflow, guards and loose panels."
    if centre < 6000:
        return "Hint: high-frequency energy, typical of bearing wear or poor lubrication."
    return "Hint: hiss above 6 kHz. Check for friction, a dry bearing or a leak."
