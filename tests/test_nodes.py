import inspect
import sys
import tomllib
import types
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _base_state_dict(band_widths, channels=2, mask_last_linear=2):
    divisor = 2 * channels
    state = {
        f"band_split.to_features.{index}.1.weight": torch.empty(256, divisor * width)
        for index, width in enumerate(band_widths)
    }
    state.update({
        "layers.0.0.layers.0.0.weight": torch.empty(1),
        "layers.0.1.layers.0.0.weight": torch.empty(1),
        "layers.11.0.layers.0.0.weight": torch.empty(1),
        "mask_estimators.0.to_freqs.0.0.0.weight": torch.empty(1),
        f"mask_estimators.0.to_freqs.0.0.{mask_last_linear}.weight": torch.empty(1),
    })
    return state


def test_node_pin_contracts(nodes_module):
    for node_name, node_class in nodes_module.NODE_CLASS_MAPPINGS.items():
        inputs = node_class.INPUT_TYPES()
        function = getattr(node_class, node_class.FUNCTION)
        parameters = set(inspect.signature(function).parameters) - {"self"}
        declared = set(inputs.get("required", {})) | set(inputs.get("optional", {}))

        assert declared <= parameters, node_name
        assert len(node_class.RETURN_TYPES) == len(node_class.RETURN_NAMES), node_name


def test_loader_has_no_pickle_acknowledgement_pin(nodes_module):
    for loader in (
        nodes_module.MelBandRoFormerModelLoader,
        nodes_module.MelBandRoFormerModelLoaderLatest,
    ):
        assert set(loader.INPUT_TYPES()["required"]) == {"model_name"}


def test_registry_additions_and_dead_entry_removal(nodes_module):
    assert nodes_module.MODEL_REGISTRY["Crowd · aufr33/viperx ⭐ (SDR 8.71)"] == (
        "cdjmix1991/mel-band-roformer-crowd",
        "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt",
    )
    assert nodes_module.MODEL_REGISTRY["Vocals fullness · Aname-Tommy"] == (
        "Aname-Tommy/MelBandRoformers",
        "FullnessVocalModel.ckpt",
    )
    assert not any("Karaoke · aufr33" in name for name in nodes_module.MODEL_REGISTRY)


def test_bs_config_preserves_numeric_band_order(nodes_module):
    widths = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 959)
    state = _base_state_dict(widths, channels=2)
    config = nodes_module.infer_bs_roformer_config(state)

    assert config["stereo"] is True
    assert config["freqs_per_bands"] == widths
    assert config["mask_estimator_depth"] == 2
    assert "sample_rate" not in config
    assert config["stft_hop_length"] == 512


def test_bs_mono_detection_uses_all_bands(nodes_module):
    # The first mono band input has width 4 and is divisible by four. Looking
    # only at that tensor used to incorrectly classify this checkpoint as stereo.
    state = _base_state_dict((2, 1023), channels=1)
    config = nodes_module.infer_bs_roformer_config(state)

    assert config["stereo"] is False
    assert config["freqs_per_bands"] == (2, 1023)


def test_bs_invalid_band_layout_fails_clearly(nodes_module):
    state = _base_state_dict((2, 1000), channels=2)
    with pytest.raises(ValueError, match="Unsupported BS-RoFormer band layout"):
        nodes_module.infer_bs_roformer_config(state)


def test_mel_and_bs_mask_depth_semantics_differ(nodes_module):
    state = _base_state_dict((2, 3, 1020), channels=2, mask_last_linear=4)
    assert nodes_module.infer_melband_config(state)["mask_estimator_depth"] == 2

    state = _base_state_dict((2, 3, 1020), channels=2, mask_last_linear=2)
    assert nodes_module.infer_bs_roformer_config(state)["mask_estimator_depth"] == 2


def test_loader_enforces_weights_only_safe_loading(nodes_module, monkeypatch):
    seen = {}

    class DummyModel:
        def eval(self):
            return self

        def load_state_dict(self, state, strict):
            seen["state"] = state
            seen["strict"] = strict

    def fake_load(path, **kwargs):
        seen["path"] = path
        seen["load_kwargs"] = kwargs
        return {"weights": torch.empty(1)}

    config = {
        "dim": 1,
        "depth": 1,
        "num_stems": 1,
        "time_transformer_depth": 1,
        "freq_transformer_depth": 1,
    }
    monkeypatch.setattr(nodes_module, "load_torch_file", fake_load)
    monkeypatch.setattr(nodes_module, "infer_config", lambda _state: ("melband", config))
    monkeypatch.setattr(nodes_module, "MelBandRoformer", lambda **_config: DummyModel())

    nodes_module.MelBandRoFormerModelLoader().loadmodel("local.ckpt")

    assert seen["load_kwargs"] == {"safe_load": True}
    assert seen["strict"] is True


def test_bs_revive_models_keep_training_hop_size(nodes_module):
    revive_overrides = {
        name: override
        for name, override in nodes_module._MODEL_CONFIG_OVERRIDES.items()
        if "revive" in name.lower()
    }
    assert len(revive_overrides) == 3
    assert all(override == {"stft_hop_length": 441} for override in revive_overrides.values())


class _IdentityOneStem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_estimators = [object()]

    def forward(self, waveform):
        return waveform


class _IdentityTwoStem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_estimators = [object(), object()]

    def forward(self, waveform):
        return torch.stack((waveform, waveform * 2), dim=1)


class _IdentityMonoStem(_IdentityOneStem):
    stereo = False

    def forward(self, waveform):
        assert waveform.shape[1] == 1
        return waveform


def test_two_stem_sampler_preserves_audio_batch(nodes_module):
    waveform = torch.randn(2, 2, 64)
    stem_1, stem_2 = nodes_module.MelBandRoFormerSampler().process(
        _IdentityOneStem(),
        {"waveform": waveform, "sample_rate": 44100},
        chunk_size=1.0,
    )

    assert stem_1["waveform"].shape == waveform.shape
    assert stem_2["waveform"].shape == waveform.shape
    torch.testing.assert_close(stem_1["waveform"], waveform)
    torch.testing.assert_close(stem_2["waveform"], torch.zeros_like(waveform))


def test_four_stem_sampler_preserves_batch_and_silence_shape(nodes_module):
    waveform = torch.randn(2, 2, 64)
    stems = nodes_module.MelBandRoFormerSampler4Stem().process4(
        _IdentityTwoStem(),
        {"waveform": waveform, "sample_rate": 44100},
        chunk_size=1.0,
    )

    assert all(stem["waveform"].shape == waveform.shape for stem in stems)
    torch.testing.assert_close(stems[0]["waveform"], waveform)
    torch.testing.assert_close(stems[1]["waveform"], waveform * 2)
    torch.testing.assert_close(stems[2]["waveform"], torch.zeros_like(waveform))
    torch.testing.assert_close(stems[3]["waveform"], torch.zeros_like(waveform))


def test_sampler_respects_mono_checkpoint(nodes_module):
    waveform = torch.stack((torch.ones(64), torch.zeros(64))).unsqueeze(0)
    stem_1, stem_2 = nodes_module.MelBandRoFormerSampler().process(
        _IdentityMonoStem(),
        {"waveform": waveform, "sample_rate": 44100},
        chunk_size=1.0,
    )

    expected = torch.full((1, 1, 64), 0.5)
    torch.testing.assert_close(stem_1["waveform"], expected)
    torch.testing.assert_close(stem_2["waveform"], torch.zeros_like(expected))


def test_spectrogram_preserves_and_broadcasts_audio_batches(nodes_module, monkeypatch):
    monkeypatch.setattr(nodes_module, "_db_spectrogram", lambda *_args: np.zeros((4, 4), dtype=np.float32))
    monkeypatch.setattr(
        nodes_module,
        "_render_figure",
        lambda *_args, **_kwargs: np.zeros((8, 12, 3), dtype=np.uint8),
    )
    audio_a = {"waveform": torch.randn(2, 2, 64), "sample_rate": 44100}
    audio_b = {"waveform": torch.randn(1, 2, 64), "sample_rate": 44100}

    (images,) = nodes_module.MelBandRoFormerSpectrogram().compare(
        audio_a, audio_b, "A", "B", "stacked", 512, 128
    )

    assert images.shape == (2, 8, 12, 3)


def test_lufs_normalize_preserves_audio_batch(nodes_module, monkeypatch):
    pyloudnorm = types.ModuleType("pyloudnorm")

    class Meter:
        def __init__(self, _sample_rate):
            pass

        def integrated_loudness(self, _waveform):
            return -20.0

    pyloudnorm.Meter = Meter
    monkeypatch.setitem(sys.modules, "pyloudnorm", pyloudnorm)
    waveform = torch.full((2, 2, 64), 0.1)

    normalized, input_lufs, gain_db = nodes_module.MelBandRoFormerLUFSNormalize().normalize(
        {"waveform": waveform, "sample_rate": 44100},
        target_lufs=-14.0,
    )

    assert normalized["waveform"].shape == waveform.shape
    assert input_lufs == -20.0
    assert gain_db == 6.0


def test_package_metadata_is_buildable_and_complete():
    with (ROOT / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)["project"]

    assert project["license"] == "Apache-2.0"
    assert project["urls"]["Repository"] == "https://github.com/ethanfel/ComfyUI-MelBandRoFormer"
    assert "BS-RoFormer==0.4.1" in project["dependencies"]
    for package in ("pyloudnorm", "soundfile", "matplotlib"):
        assert any(dependency.startswith(package) for dependency in project["dependencies"])
