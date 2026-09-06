"""Checkpoint-compatible BS-RoFormer inference, sharing the Mel model's layers.

Adapted from lucidrains/BS-RoFormer 0.4.1:
https://github.com/lucidrains/BS-RoFormer

MIT License

Copyright (c) 2023 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from functools import partial

import torch
from torch import nn
from einops import rearrange, pack, unpack
from rotary_embedding_torch import RotaryEmbedding

from .mel_band_roformer import BandSplit, MaskEstimator, RMSNorm, Transformer


class BSRoformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        freqs_per_bands,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        dim_head=64,
        heads=8,
        attn_dropout=0.,
        ff_dropout=0.,
        flash_attn=True,
        dim_freqs_in=1025,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn=None,
        mask_estimator_depth=2,
    ):
        super().__init__()
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems

        expected_freqs = stft_n_fft // 2 + 1
        if (len(freqs_per_bands) < 2 or any(f <= 0 for f in freqs_per_bands)
                or sum(freqs_per_bands) != expected_freqs
                or dim_freqs_in != expected_freqs):
            raise ValueError(f"BS-RoFormer bands must cover all {expected_freqs} STFT bins")
        if mask_estimator_depth < 1:
            raise ValueError("mask_estimator_depth must be at least 1")

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            # BS checkpoints normalize once after all axial transformer blocks.
            norm_output=False,
        )
        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed,
                            **transformer_kwargs),
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed,
                            **transformer_kwargs),
            ])
            for _ in range(depth)
        ])
        self.final_norm = RMSNorm(dim)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )
        self.stft_window_fn = partial(
            torch.hann_window if stft_window_fn is None else stft_window_fn,
            stft_win_length,
        )
        dim_inputs = tuple(2 * f * self.audio_channels for f in freqs_per_bands)
        self.band_split = BandSplit(dim=dim, dim_inputs=dim_inputs)
        self.mask_estimators = nn.ModuleList([
            # Mel's MLP depth counts hidden layers; BS counts all linear layers.
            MaskEstimator(dim=dim, dim_inputs=dim_inputs, depth=mask_estimator_depth - 1)
            for _ in range(num_stems)
        ])

    def forward(self, raw_audio):
        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")
        if raw_audio.ndim != 3 or raw_audio.shape[1] != self.audio_channels:
            raise ValueError(f"Expected audio with {self.audio_channels} channel(s)")

        batch = raw_audio.shape[0]
        window = self.stft_window_fn(device=raw_audio.device)
        raw_audio = rearrange(raw_audio, "b s t -> (b s) t")
        spectrum = torch.stft(raw_audio, **self.stft_kwargs, window=window, return_complex=True)
        spectrum = torch.view_as_real(spectrum)
        spectrum = rearrange(spectrum, "(b s) f t c -> b (f s) t c", b=batch, s=self.audio_channels)

        x = self.band_split(rearrange(spectrum, "b f t c -> b t (f c)"))
        for time_transformer, freq_transformer in self.layers:
            x = rearrange(x, "b t f d -> b f t d")
            x, shape = pack([x], "* t d")
            x = time_transformer(x)
            x, = unpack(x, shape, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, shape = pack([x], "* f d")
            x = freq_transformer(x)
            x, = unpack(x, shape, "* f d")
        x = self.final_norm(x)

        mask = torch.stack([estimator(x) for estimator in self.mask_estimators], dim=1)
        mask = rearrange(mask, "b n t (f c) -> b n f t c", c=2)
        spectrum = torch.view_as_complex(spectrum.unsqueeze(1)) * torch.view_as_complex(mask)
        spectrum = rearrange(spectrum, "b n (f s) t -> (b n s) f t", s=self.audio_channels)
        audio = torch.istft(spectrum, **self.stft_kwargs, window=window, return_complex=False)
        audio = rearrange(audio, "(b n s) t -> b n s t", b=batch, n=self.num_stems, s=self.audio_channels)
        return audio[:, 0] if self.num_stems == 1 else audio
