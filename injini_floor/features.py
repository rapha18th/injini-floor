"""Log-mel front end for Injini.

The embedder Injini ships is an EfficientAT MobileNetV3 pretrained on AudioSet.
Those weights are only valid against EfficientAT's own mel spec, so this module
reproduces that spec exactly:

    sample rate 32 kHz, pre-emphasis 0.97, STFT n_fft=1024 / hop=320 /
    win_length=800 (symmetric Hann, zero-padded to n_fft, centered/reflect),
    128 Kaldi-style mel bands 0-15000 Hz, log(mel + 1e-5), then (x + 4.5) / 5.

Two implementations of that one spec:

  * ``reference_logmel`` uses EfficientAT's ``AugmentMelSTFT`` in eval mode. This
    is the ground truth used for training and evaluation on Kaggle.
  * ``logmel`` is a pure-NumPy reimplementation with no torch / torchaudio at
    run time. It loads a precomputed Kaldi mel matrix (``models/mel_kaldi_128x513.npy``,
    written by ``dump_mel_matrix``) so the fiddly Kaldi filterbank never has to
    be re-derived. This is the version ported to Kotlin for the phone, and
    ``tests/test_feature_parity.py`` holds the two within 1e-3.

Capture on the phone is 16 kHz (matches SiloSense, matches the Android
UNPROCESSED source, keeps knock detection honestly out of scope). ``load_audio``
resamples whatever it is given to 32 kHz with the same naive linear interp
SiloSense used.
"""
from __future__ import annotations

import os

import numpy as np
import soundfile as sf

SR = 32_000
N_FFT = 1024
HOP = 320
WIN_LENGTH = 800
N_MELS = 128
FMIN = 0.0
FMAX = 15_000.0  # AugmentMelSTFT eval: sr // 2 - fmax_aug_range // 2 = 16000 - 1000
PREEMPH = 0.97
LOG_OFFSET = 1e-5
NORM_ADD = 4.5
NORM_DIV = 5.0

CLIP_SECONDS = 10.0
CLIP_SAMPLES = int(SR * CLIP_SECONDS)
N_FRAMES = 1 + CLIP_SAMPLES // HOP  # 1001, the fixed length used for the on-device ONNX graph

_HERE = os.path.dirname(os.path.abspath(__file__))
MEL_MATRIX_PATH = os.path.join(_HERE, os.pardir, "models", "mel_kaldi_128x513.npy")

# Symmetric (periodic=False) Hann of win_length, zero-padded to n_fft and centered,
# matching torch.stft(win_length=800, n_fft=1024, window=hann_window(800, periodic=False)).
_hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WIN_LENGTH) / (WIN_LENGTH - 1))
_WINDOW = np.zeros(N_FFT, dtype=np.float64)
_pad_left = (N_FFT - WIN_LENGTH) // 2
_WINDOW[_pad_left:_pad_left + WIN_LENGTH] = _hann

_MEL: np.ndarray | None = None


def dump_mel_matrix(path: str = MEL_MATRIX_PATH) -> np.ndarray:
    """Compute EfficientAT's Kaldi mel filterbank once (needs torchaudio) and cache it.

    Returns a (128, 513) float32 matrix: get_mel_banks(...) padded with one zero
    column, exactly as AugmentMelSTFT does before ``mel_basis @ power``.
    """
    import torch
    import torchaudio

    mel_basis, _ = torchaudio.compliance.kaldi.get_mel_banks(
        N_MELS, N_FFT, SR, FMIN, FMAX,
        vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0,
    )
    mel_basis = torch.nn.functional.pad(mel_basis, (0, 1), mode="constant", value=0)
    mat = mel_basis.cpu().numpy().astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, mat)
    return mat


def mel_matrix() -> np.ndarray:
    global _MEL
    if _MEL is None:
        if not os.path.exists(MEL_MATRIX_PATH):
            dump_mel_matrix(MEL_MATRIX_PATH)
        _MEL = np.load(MEL_MATRIX_PATH).astype(np.float64)
    return _MEL


def resample_linear(y: np.ndarray, orig_sr: int, target_sr: int = SR) -> np.ndarray:
    if orig_sr == target_sr or len(y) == 0:
        return y.astype(np.float32)
    duration = len(y) / orig_sr
    n_target = max(1, int(round(duration * target_sr)))
    x_orig = np.linspace(0.0, duration, num=len(y), endpoint=False)
    x_target = np.linspace(0.0, duration, num=n_target, endpoint=False)
    return np.interp(x_target, x_orig, y).astype(np.float32)


def load_audio(path: str, offset: float = 0.0, duration: float | None = None) -> np.ndarray:
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        f.seek(int(offset * sr))
        n = int(duration * sr) if duration is not None else -1
        y = f.read(frames=n, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return resample_linear(np.asarray(y, dtype=np.float32), sr, SR)


def fixed_length(y: np.ndarray, n: int = CLIP_SAMPLES) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)), mode="constant")


def _preemphasis(y: np.ndarray) -> np.ndarray:
    # torch: conv1d(x, [[[-.97, 1]]]) -> out[t] = -0.97*x[t] + x[t+1], length N-1.
    return (y[1:] - PREEMPH * y[:-1]).astype(np.float64)


def _stft_power(y: np.ndarray) -> np.ndarray:
    pad = N_FFT // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (len(yp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[:, None] + HOP * np.arange(n_frames)[None, :]
    frames = yp[idx] * _WINDOW[:, None]
    spec = np.fft.rfft(frames, n=N_FFT, axis=0)
    return (spec.real ** 2 + spec.imag ** 2)  # (513, n_frames)


def logmel(y: np.ndarray, fixed: bool = False) -> np.ndarray:
    """Pure-NumPy log-mel matching ``reference_logmel``. (128, T) float32.

    ``fixed=True`` pins the output to ``N_FRAMES`` columns for the on-device graph.
    """
    if fixed:
        y = fixed_length(y)
    power = _stft_power(_preemphasis(np.asarray(y, dtype=np.float64)))
    mel = mel_matrix() @ power
    mel = np.log(mel + LOG_OFFSET)
    mel = (mel + NORM_ADD) / NORM_DIV
    out = mel.astype(np.float32)
    if fixed:
        if out.shape[1] >= N_FRAMES:
            out = out[:, :N_FRAMES]
        else:
            out = np.pad(out, ((0, 0), (0, N_FRAMES - out.shape[1])), mode="edge")
    return out


def reference_logmel(y: np.ndarray) -> np.ndarray:
    """EfficientAT AugmentMelSTFT in eval mode. Ground truth. Needs torch/torchaudio."""
    import sys

    import torch

    vendor = os.path.join(_HERE, os.pardir, "vendor_efficientat")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from models.preprocess import AugmentMelSTFT

    mel = AugmentMelSTFT(
        n_mels=N_MELS, sr=SR, win_length=WIN_LENGTH, hopsize=HOP, n_fft=N_FFT,
        freqm=0, timem=0, fmin=FMIN, fmax=FMAX, fmin_aug_range=1, fmax_aug_range=1,
    )
    mel.eval()
    with torch.no_grad():
        spec = mel(torch.from_numpy(np.asarray(y, dtype=np.float32))[None, :])
    return spec.squeeze(0).cpu().numpy()


if __name__ == "__main__":
    mat = dump_mel_matrix()
    print(f"wrote {MEL_MATRIX_PATH}  shape={mat.shape}  sum={mat.sum():.3f}")
