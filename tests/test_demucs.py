import sys
import types

import pytest
import torch


class _DemucsBag(torch.nn.Module):
    # Deliberately different from the public output order.
    sources = ["vocals", "drums", "other", "bass"]
    samplerate = 44100
    audio_channels = 2
    max_allowed_segment = 0.01


def _fake_demucs(nodes_module, calls):
    gains = {"drums": 2, "bass": 3, "other": 4, "vocals": 5}

    def apply_model(model, audio, *, shifts, split, device):
        assert shifts == 0 and split is False
        assert device == audio.device
        assert audio.shape[-1] <= int(model.max_allowed_segment * model.samplerate)
        calls.append(audio.shape)
        return torch.stack([audio * gains[source] for source in model.sources], dim=1)

    return nodes_module.DemucsModel(_DemucsBag(), apply_model).eval()


@pytest.mark.parametrize("loader_name", ["MelBandRoFormerModelLoader", "MelBandRoFormerModelLoaderLatest"])
def test_demucs_available_and_loaded_without_roformer_path(nodes_module, monkeypatch, loader_name):
    seen = []
    model = _fake_demucs(nodes_module, [])

    def load(name):
        seen.append(name)
        return model

    def reject_roformer_load(*_args, **_kwargs):
        pytest.fail("Demucs must not pass through RoFormer checkpoint loading")

    monkeypatch.setattr(nodes_module, "load_demucs_model", load)
    monkeypatch.setattr(nodes_module, "load_torch_file", reject_roformer_load)
    loader = getattr(nodes_module, loader_name)
    display_name = next(iter(nodes_module.DEMUCS_MODELS))
    assert display_name in loader.INPUT_TYPES()["required"]["model_name"][0]
    assert loader.VALIDATE_INPUTS(display_name) is True
    loaded, recommended = loader().loadmodel(display_name)
    assert loaded is model
    assert recommended == model.max_chunk_size
    assert seen == ["htdemucs_ft"]


@pytest.mark.parametrize("four_stems", [False, True])
@pytest.mark.parametrize("intensity", [1.0, 0.4])
def test_demucs_samplers_normalize_once_and_preserve_stems_and_batches(nodes_module, four_stems, intensity):
    calls = []
    model = _fake_demucs(nodes_module, calls)
    waveform = torch.randn(2, 2, 1800) * 0.2 + torch.tensor([1.0, -2.0]).reshape(2, 1, 1)
    audio = {"waveform": waveform, "sample_rate": 44100}
    if four_stems:
        sampler = nodes_module.MelBandRoFormerSampler4Stem().process4
    else:
        sampler = nodes_module.MelBandRoFormerSampler().process
    # Requested chunks exceed the model limit. Multiple overlapping chunks and
    # distinct clip means expose accidental normalization of individual chunks.
    stems = sampler(model, audio, chunk_size=8.0, overlap=3, batch_size=2, intensity=intensity)
    assert len(stems) == (4 if four_stems else 2)
    mean = waveform.mean(dim=1).mean(dim=-1).reshape(2, 1, 1)
    for stem, gain in zip(stems, (2, 3, 4, 5)):
        expected = ((waveform - mean) * gain + mean) * intensity + waveform * (1 - intensity)
        assert stem["sample_rate"] == 44100
        torch.testing.assert_close(stem["waveform"], expected, rtol=1e-5, atol=2e-6)
    assert len(calls) > 2
    assert any(shape[0] == 2 for shape in calls)


@pytest.mark.parametrize("length", [1, 1000])
def test_demucs_silence_is_finite(nodes_module, length):
    model = _fake_demucs(nodes_module, [])
    audio = {"waveform": torch.zeros(1, 2, length), "sample_rate": 44100}
    stems = nodes_module.MelBandRoFormerSampler4Stem().process4(model, audio)
    for stem in stems:
        assert torch.isfinite(stem["waveform"]).all()
        torch.testing.assert_close(stem["waveform"], audio["waveform"])


@pytest.mark.parametrize("four_stems", [False, True])
def test_demucs_restores_input_sample_rate(nodes_module, monkeypatch, four_stems):
    calls = []

    def resample(waveform, *, orig_freq, new_freq):
        calls.append((orig_freq, new_freq))
        length = round(waveform.shape[-1] * new_freq / orig_freq)
        return torch.nn.functional.interpolate(waveform, size=length, mode="linear", align_corners=False)

    monkeypatch.setattr(nodes_module.TAF, "resample", resample)
    model = _fake_demucs(nodes_module, [])
    audio = {"waveform": torch.zeros(1, 1, 500), "sample_rate": 22050}
    sampler = (nodes_module.MelBandRoFormerSampler4Stem().process4 if four_stems
               else nodes_module.MelBandRoFormerSampler().process)
    stems = sampler(model, audio)
    assert (22050, 44100) in calls and (44100, 22050) in calls
    for stem in stems:
        assert stem["sample_rate"] == 22050
        assert stem["waveform"].shape == (1, 2, 500)


def test_demucs_factory_uses_huggingface_loader(nodes_module, monkeypatch):
    calls = []
    hf = types.ModuleType("demucs.hf")
    apply = types.ModuleType("demucs.apply")
    hf.get_hf_model = lambda name: calls.append(name) or _DemucsBag()
    apply.apply_model = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "demucs", types.ModuleType("demucs"))
    monkeypatch.setitem(sys.modules, "demucs.hf", hf)
    monkeypatch.setitem(sys.modules, "demucs.apply", apply)
    model = nodes_module.load_demucs_model("htdemucs_ft")
    assert calls == ["htdemucs_ft"]
    assert not model.training and not model.model.training


def test_demucs_missing_dependency_has_install_hint(nodes_module, monkeypatch):
    monkeypatch.setitem(sys.modules, "demucs.apply", None)
    with pytest.raises(ImportError, match="demucs>=4.1"):
        nodes_module.load_demucs_model("htdemucs_ft")
