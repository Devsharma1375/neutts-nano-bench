"""Signal-level checks that need no ML models."""
from __future__ import annotations

import numpy as np

SR = 24_000


def log_mel(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(y=np.asarray(audio, np.float32), sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    return 10.0 * np.log10(np.maximum(mel, 1e-10))


def stream_fidelity_db(streamed: np.ndarray, reference: np.ndarray, floor_db: float = -60.0) -> float:
    """Mean absolute log-mel difference (dB) between streamed audio and the same tokens
    decoded in one pass.

    Both come from identical tokens, so any difference is caused purely by chunked
    decoding (lookback, chunk size, cross-fades, watermark placement). 0 = identical;
    differences under ~1 dB are generally inaudible. Frames that are silent in the
    reference are ignored so silence can't dominate the average.
    """
    n = min(len(streamed), len(reference))
    if n < 2048:
        return float("nan")
    a, b = log_mel(streamed[:n]), log_mel(reference[:n])
    mask = b.max(axis=0) > floor_db
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(a[:, mask] - b[:, mask])))


def longest_silence_s(audio: np.ndarray, sr: int = SR, threshold_db: float = -45.0, frame_s: float = 0.02) -> float:
    """Longest run of frames below ``threshold_db`` (relative to full scale), excluding
    leading and trailing silence."""
    hop = int(sr * frame_s)
    if len(audio) < hop:
        return 0.0
    frames = audio[: len(audio) // hop * hop].reshape(-1, hop)
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-12)
    voiced = np.flatnonzero(rms_db > threshold_db)
    if voiced.size == 0:
        return len(audio) / sr
    inner = rms_db[voiced[0] : voiced[-1] + 1] <= threshold_db
    best = run = 0
    for silent in inner:
        run = run + 1 if silent else 0
        best = max(best, run)
    return best * frame_s


def to_pcm16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
