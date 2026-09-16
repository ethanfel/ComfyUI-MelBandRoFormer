import json
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("skip_connection", [False, True])
def test_mel_inference_matches_upstream(mel_model_class, skip_connection):
    # CPU output sampled from the upstream revision used when Aname's models
    # were released. Deterministic weights avoid downloading external models.
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "mel_skip_connections.json").read_text()
    )
    options = {"skip_connection": True} if skip_connection else {}
    model = mel_model_class(**fixture["config"], **options).eval()
    state = model.state_dict()
    for key, value in state.items():
        if key.endswith(".freqs"):
            continue
        indices = torch.arange(value.numel(), dtype=torch.float32)
        values = torch.sin(indices / 11 + sum(key.encode()) / 100)
        state[key] = (1 + values * 0.1 if key.endswith(".gamma") else values * 0.2).reshape(value.shape)
    model.load_state_dict(state, strict=True)
    audio = torch.sin(torch.arange(518, dtype=torch.float32) / 17).reshape(1, 2, 259)
    with torch.inference_mode():
        actual = model(audio)
    expected = torch.tensor(fixture["snapshots"][str(skip_connection).lower()])
    assert actual.shape == (1, 4, 2, 256)
    torch.testing.assert_close(actual[..., ::32], expected, rtol=1e-5, atol=1e-6)


@pytest.fixture(scope="module")
def aname_state(mel_model_class):
    # Published STFT/band/stem layout with small transformer dimensions, so the
    # real loader and strict state loading can be exercised with little memory.
    model = mel_model_class(
        dim=8, depth=3, stereo=True, num_stems=4,
        time_transformer_depth=1, freq_transformer_depth=1,
        dim_freqs_in=2049, stft_n_fft=4096, stft_win_length=4096,
        stft_hop_length=882, mask_estimator_depth=2, skip_connection=True,
    ).eval()
    # Constant complex masks give each output a distinct, known gain.
    with torch.no_grad():
        for stem, estimator in enumerate(model.mask_estimators, start=1):
            for band in estimator.to_freqs:
                linear = band[0][-1]
                linear.weight.zero_()
                linear.bias.zero_()
                linear.bias[:linear.out_features // 2:2] = 2 * stem
    return model.state_dict()


@pytest.mark.parametrize("size", ["large", "XL"])
@pytest.mark.parametrize("selection", ["registry", "legacy", "local", "subfolder"])
@pytest.mark.parametrize("loader_name", ["MelBandRoFormerModelLoader", "MelBandRoFormerModelLoaderLatest"])
def test_aname_loader_configuration(
    nodes_module, mel_model_class, aname_state, monkeypatch, size, selection, loader_name,
):
    canonical = f"4-stem {size} · Aname-Tommy [stem_1=drums]"
    filename = f"mel_band_roformer_4stems_{size.lower()}_ver1.ckpt"
    name = {
        "registry": canonical,
        "legacy": f"4-stem {size} · Aname-Tommy [stem_1=vox only]",
        "local": filename,
        "subfolder": f"custom/{filename}",
    }[selection]
    monkeypatch.setattr(nodes_module.folder_paths, "get_filename_list", lambda _name: [name])
    monkeypatch.setattr(nodes_module, "download_hf_model", lambda _repo, file: f"/models/{file}")
    monkeypatch.setattr(nodes_module, "load_torch_file", lambda *_args, **_kwargs: aname_state)
    monkeypatch.setattr(nodes_module, "MelBandRoformer", mel_model_class)

    loader = getattr(nodes_module, loader_name)
    assert loader.VALIDATE_INPUTS(name) is True
    assert loader.VALIDATE_INPUTS("missing.ckpt") is not True
    choices = loader.INPUT_TYPES()["required"]["model_name"][0]
    assert canonical in choices
    assert not any("vox only" in choice for choice in choices)

    model, _ = loader().loadmodel(name)
    assert isinstance(model, mel_model_class)
    assert model.stft_kwargs == {
        "n_fft": 4096, "hop_length": 882, "win_length": 4096, "normalized": False,
    }
    assert model.skip_connection is True
    assert model.num_stems == 4
    assert not model.training


def test_aname_drum_output_and_four_stem_order(nodes_module, mel_model_class, aname_state, monkeypatch):
    monkeypatch.setattr(nodes_module, "load_torch_file", lambda *_args, **_kwargs: aname_state)
    monkeypatch.setattr(nodes_module, "MelBandRoformer", mel_model_class)
    model, _ = nodes_module.MelBandRoFormerModelLoader().loadmodel(
        "mel_band_roformer_4stems_large_ver1.ckpt"
    )
    waveform = torch.randn(2, 2, 5000)
    audio = {"waveform": waveform, "sample_rate": 44100}
    # One padded chunk per clip checks the real model, overlap reconstruction,
    # batch handling, and all four output positions without a full-size model.
    stems = nodes_module.MelBandRoFormerSampler4Stem().process4(model, audio, chunk_size=1.0)
    assert len(stems) == 4
    for index, stem in enumerate(stems, start=1):
        assert stem["sample_rate"] == 44100
        torch.testing.assert_close(stem["waveform"], waveform * index, rtol=1e-5, atol=3e-6)

    drums, bass = nodes_module.MelBandRoFormerSampler().process(model, audio, chunk_size=1.0)
    torch.testing.assert_close(drums["waveform"], stems[0]["waveform"])
    torch.testing.assert_close(bass["waveform"], stems[1]["waveform"])
