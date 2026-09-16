"""Adapt Demucs's pretrained ensemble to the existing audio samplers."""

import math

import torch
from torch import nn


DEMUCS_MODELS = {
    "[Demucs] htdemucs_ft · 4-stem [stem_1=drums]": "htdemucs_ft",
}


class DemucsModel(nn.Module):
    sources = ("drums", "bass", "other", "vocals")
    num_stems = 4

    def __init__(self, model, apply_model):
        super().__init__()
        if len(model.sources) != 4 or set(model.sources) != set(self.sources):
            raise ValueError("Expected a Demucs model with drums, bass, other, and vocals")
        self.model = model
        self._apply_model = apply_model
        self._source_indices = [model.sources.index(source) for source in self.sources]
        self.samplerate = model.samplerate
        self.stereo = model.audio_channels == 2
        self.max_chunk_size = float(model.max_allowed_segment)
        if not math.isfinite(self.max_chunk_size) or self.max_chunk_size <= 0:
            raise ValueError("Demucs model does not specify a valid maximum segment length")

    def normalize_audio(self, audio):
        """Normalize the whole clip before chunking, as in Demucs's API."""
        if audio.shape[-1] == 0:
            raise ValueError("Audio must contain at least one sample")
        reference = audio.mean(dim=0)
        mean = reference.mean()
        std = reference.std(unbiased=reference.numel() > 1) + 1e-8
        return (audio - mean) / std, (mean, std)

    def denormalize_audio(self, stems, normalization):
        mean, std = normalization
        return stems * std.to(stems) + mean.to(stems)

    def forward(self, audio):
        # The ComfyUI sampler already handles overlapping chunks and batches.
        # Disable Demucs's extra splitting and random shifts here.
        stems = self._apply_model(
            self.model, audio, shifts=0, split=False, device=audio.device,
        )
        return stems[:, self._source_indices]


def load_demucs_model(name):
    if name not in DEMUCS_MODELS.values():
        raise ValueError(f"Unsupported Demucs model: {name}")
    try:
        from demucs.apply import apply_model
        from demucs.hf import get_hf_model
    except ImportError as exc:
        raise ImportError(
            "Demucs support requires demucs>=4.1. Install this node's updated "
            'requirements or run: pip install "demucs>=4.1"'
        ) from exc

    # Use the official Hugging Face safetensors loader directly. Its downloads
    # use the user's normal HF cache; no legacy pickle fallback is needed.
    return DemucsModel(get_hf_model(name), apply_model).eval()
