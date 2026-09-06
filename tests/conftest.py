import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def bs_model_class():
    """Import the real architecture separately from the node API test stubs."""
    package = types.ModuleType("melband_architecture")
    package.__path__ = [str(ROOT / "model")]
    sys.modules[package.__name__] = package
    return importlib.import_module("melband_architecture.bs_roformer").BSRoformer


@pytest.fixture(scope="session")
def nodes_module(tmp_path_factory):
    """Import nodes.py with the small subset of ComfyUI APIs it needs."""
    model_dir = tmp_path_factory.mktemp("models")

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(model_dir)
    folder_paths.get_folder_paths = lambda _name: [str(model_dir)]
    folder_paths.get_filename_list = lambda _name: []
    folder_paths.get_full_path_or_raise = lambda _name, filename: str(model_dir / filename)
    folder_paths.add_model_folder_path = lambda *_args, **_kwargs: None
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    model_management = types.ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: torch.device("cpu")
    model_management.unet_offload_device = lambda: torch.device("cpu")
    comfy_utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, _total):
            pass

        def update(self, _amount):
            pass

    comfy_utils.ProgressBar = ProgressBar
    comfy_utils.load_torch_file = lambda *_args, **_kwargs: {}
    comfy.model_management = model_management
    comfy.utils = comfy_utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy.utils"] = comfy_utils

    torchaudio = types.ModuleType("torchaudio")
    torchaudio_functional = types.ModuleType("torchaudio.functional")
    torchaudio_functional.resample = lambda waveform, **_kwargs: waveform
    torchaudio.functional = torchaudio_functional
    sys.modules["torchaudio"] = torchaudio
    sys.modules["torchaudio.functional"] = torchaudio_functional

    package = types.ModuleType("melband_plugin")
    package.__path__ = [str(ROOT)]
    sys.modules["melband_plugin"] = package

    bundled_model = types.ModuleType("melband_plugin.model.mel_band_roformer")
    bundled_model.MelBandRoformer = type("MelBandRoformer", (), {})
    sys.modules["melband_plugin.model.mel_band_roformer"] = bundled_model

    bs_package = types.ModuleType("melband_plugin.model.bs_roformer")
    bs_package.BSRoformer = type("BSRoformer", (), {})
    sys.modules[bs_package.__name__] = bs_package

    spec = importlib.util.spec_from_file_location("melband_plugin.nodes", ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
