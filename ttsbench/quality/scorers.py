"""Model-based quality scores. All run after timing, in a separate step, because they
compete for the same CPU and would distort the latency numbers.

* Intelligibility: Whisper (faster-whisper, INT8) transcribes the output, scored as word
  and character error rate against the input text after normalisation.
* Speaker similarity: cosine similarity of ECAPA-TDNN speaker embeddings (SpeechBrain,
  trained on VoxCeleb) between the output and the reference recording.
* Naturalness (optional): UTMOS predicted MOS. Off by default, because it loads code
  from GitHub via torch.hub with ``trust_repo=True``.
* Watermark: Perth detection confidence on the final audio.

Absolute values depend on the scoring models. Reports therefore focus on *differences
from the baseline configuration*, which cancel most scorer bias.
"""
from __future__ import annotations

import numpy as np

from ..engines.components import CACHE_DIR
from .text_norm import normalize


def load_audio(path, sr: int) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio.astype(np.float32)


class AsrScorer:
    def __init__(self, model: str = "base.en", threads: int = 0) -> None:
        from faster_whisper import WhisperModel

        self.model_name = model
        self.model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)

    def transcribe(self, path) -> str:
        # Decode ourselves and pass an array: faster-whisper's own file decoder goes through
        # PyAV, whose API has broken between releases (seen with faster-whisper 1.2.1).
        audio = load_audio(path, 16_000)
        segments, _ = self.model.transcribe(audio, language="en", beam_size=5, vad_filter=False,
                                            condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()

    def score(self, path, reference_text: str) -> dict:
        import jiwer

        hyp = self.transcribe(path)
        ref_n, hyp_n = normalize(reference_text), normalize(hyp)
        if not ref_n:
            return {"hyp": hyp}
        return {
            "hyp": hyp,
            "wer": float(jiwer.wer(ref_n, hyp_n)) if hyp_n else 1.0,
            "cer": float(jiwer.cer(ref_n, hyp_n)) if hyp_n else 1.0,
        }


class SpeakerScorer:
    SR = 16_000

    def __init__(self) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(CACHE_DIR / "spkrec-ecapa"),
            run_opts={"device": "cpu"},
        )
        self._ref_cache: dict[str, np.ndarray] = {}

    def embed(self, audio: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            emb = self.model.encode_batch(torch.from_numpy(audio).unsqueeze(0)).squeeze().numpy()
        return emb / (np.linalg.norm(emb) + 1e-9)

    def similarity(self, path, reference_wav) -> float:
        key = str(reference_wav)
        if key not in self._ref_cache:
            self._ref_cache[key] = self.embed(load_audio(reference_wav, self.SR))
        return float(np.dot(self.embed(load_audio(path, self.SR)), self._ref_cache[key]))


class MosScorer:
    SR = 16_000

    def __init__(self) -> None:
        import torch

        self.model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

    def score(self, path) -> float:
        import torch

        audio = torch.from_numpy(load_audio(path, self.SR)).unsqueeze(0)
        with torch.no_grad():
            return float(self.model(audio, self.SR).item())


class WatermarkScorer:
    def __init__(self) -> None:
        from ..engines.components import load_perth

        self.wm = load_perth()

    def score(self, path) -> float:
        return float(self.wm.get_watermark(load_audio(path, 24_000), sample_rate=24_000))
