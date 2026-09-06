import json
from pathlib import Path

import pytest
import torch


def test_legacy_checkpoint_and_inference_parity(bs_model_class):
    # Shapes and output recorded from the unmodified BS-RoFormer 0.4.1 wheel.
    # Deterministic weights keep this fixture small and avoid an external package
    # dependency or checkpoint download when running the regression test.
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "bs_roformer_0_4_1.json").read_text()
    )
    state = {}
    for key, shape in fixture["state_shapes"].items():
        if key in fixture["rotary_freqs"]:
            state[key] = torch.tensor(fixture["rotary_freqs"][key])
            continue
        indices = torch.arange(torch.Size(shape).numel(), dtype=torch.float32)
        values = torch.sin(indices / 11 + sum(key.encode()) / 100)
        state[key] = (1 + values * 0.1 if key.endswith(".gamma") else values * 0.2).reshape(shape)

    model = bs_model_class(**fixture["config"]).eval()
    model.load_state_dict(state, strict=True)
    audio = torch.sin(torch.arange(128, dtype=torch.float32) / 17).reshape(2, 2, 32)
    with torch.inference_mode():
        actual = model(audio)
    torch.testing.assert_close(actual, torch.tensor(fixture["output"]), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("num_stems", [1, 2, 4])
def test_bs_masks_preserve_audio_channels_and_stems(bs_model_class, stereo, num_stems):
    model = bs_model_class(
        dim=8, depth=1, freqs_per_bands=(2, 3, 4), stereo=stereo,
        num_stems=num_stems, dim_head=4, heads=2, dim_freqs_in=9,
        stft_n_fft=16, stft_win_length=16, stft_hop_length=4,
    ).eval()
    # Set each stem's complex mask to a known real gain. This checks STFT
    # reconstruction and channel ordering independently of the learned weights.
    with torch.no_grad():
        for stem, estimator in enumerate(model.mask_estimators, start=1):
            for band in estimator.to_freqs:
                linear = band[0][-1]
                linear.weight.zero_()
                linear.bias.zero_()
                # GLU's zero gate halves the real/imaginary part before it.
                linear.bias[:linear.out_features // 2:2] = 2 * stem

    audio = torch.randn(2, 2 if stereo else 1, 131)
    with torch.inference_mode():
        actual = model(audio)
    expected = audio[..., :128]
    if num_stems > 1:
        expected = torch.stack([expected * stem for stem in range(1, num_stems + 1)], dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_bs_loader_uses_bundled_checkpoint_architecture(nodes_module, bs_model_class, monkeypatch):
    model = bs_model_class(
        dim=8, depth=1, freqs_per_bands=(2, 3, 1020), stereo=True,
        time_transformer_depth=1, freq_transformer_depth=1,
    ).eval()
    state = model.state_dict()
    monkeypatch.setattr(nodes_module, "BSRoformer", bs_model_class)
    monkeypatch.setattr(nodes_module, "load_torch_file", lambda *_args, **_kwargs: state)
    loaded, _ = nodes_module.MelBandRoFormerModelLoader().loadmodel("legacy-bs.ckpt")
    assert isinstance(loaded, bs_model_class)
    assert not loaded.training
    for key, value in state.items():
        torch.testing.assert_close(loaded.state_dict()[key], value)

    # A broken checkpoint must still fail strict loading.
    del state["final_norm.gamma"]
    with pytest.raises(RuntimeError, match="final_norm.gamma"):
        nodes_module.MelBandRoFormerModelLoader().loadmodel("legacy-bs.ckpt")
