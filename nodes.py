import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio.functional as TAF

import folder_paths

from .model.mel_band_roformer import MelBandRoformer

try:
    from bs_roformer import BSRoformer
    _BS_ROFORMER_AVAILABLE = True
except ImportError:
    _BS_ROFORMER_AVAILABLE = False

script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
from comfy.utils import load_torch_file, ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

# ---------------------------------------------------------------------------
# Base configs — variable fields are overridden by infer_* functions
# ---------------------------------------------------------------------------
_MELBAND_BASE_CONFIG = {
    "stereo": True,
    "num_bands": 60,
    "dim_head": 64,
    "heads": 8,
    "attn_dropout": 0,
    "ff_dropout": 0,
    "flash_attn": True,
    "dim_freqs_in": 1025,
    "sample_rate": 44100,
    "stft_n_fft": 2048,
    "stft_hop_length": 441,
    "stft_win_length": 2048,
    "stft_normalized": False,
    "mask_estimator_depth": 2,
    "multi_stft_resolution_loss_weight": 1.0,
    "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
    "multi_stft_hop_size": 147,
    "multi_stft_normalized": False,
}

_BS_BASE_CONFIG = {
    "stereo": True,
    "dim_head": 64,
    "heads": 8,
    "attn_dropout": 0,
    "ff_dropout": 0,
    "flash_attn": True,
    "dim_freqs_in": 1025,
    "sample_rate": 44100,
    "stft_n_fft": 2048,
    "stft_hop_length": 441,
    "stft_win_length": 2048,
    "stft_normalized": False,
    "mask_estimator_depth": 2,
    "multi_stft_resolution_loss_weight": 1.0,
    "multi_stft_resolutions_window_sizes": (4096, 2048, 1024, 512, 256),
    "multi_stft_hop_size": 147,
    "multi_stft_normalized": False,
}

# ---------------------------------------------------------------------------
# HuggingFace model registry  (display_name -> (repo_id, filename))
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # ── Vocals ──────────────────────────────────────────────────────────────
    "Vocals · Kim fp16 ⭐ [Kijai]":                  ("Kijai/MelBandRoFormer_comfy",                      "MelBandRoformer_fp16.safetensors"),
    "Vocals · Kim fp32 [Kijai]":                     ("Kijai/MelBandRoFormer_comfy",                      "MelBandRoformer_fp32.safetensors"),
    "Vocals · Kim original [KimberleyJSN]":          ("KimberleyJSN/melbandroformer",                     "MelBandRoformer.ckpt"),
    "Vocals · Kim FT v2 ⭐ [pcunwa]":                ("pcunwa/Kim-Mel-Band-Roformer-FT",                  "kimmel_unwa_ft2.ckpt"),
    "Vocals · Kim FT v2 bleedless [pcunwa]":         ("pcunwa/Kim-Mel-Band-Roformer-FT",                  "kimmel_unwa_ft2_bleedless.ckpt"),
    "Vocals · Kim FT v1 [pcunwa]":                   ("pcunwa/Kim-Mel-Band-Roformer-FT",                  "kimmel_unwa_ft.ckpt"),
    "Vocals · becruily":                             ("becruily/mel-band-roformer-vocals",                "mel_band_roformer_vocals_becruily.ckpt"),
    "Vocals · GaboxR67 fv7":                         ("GaboxR67/MelBandRoformers",                        "melbandroformers/vocals/voc_fv7.ckpt"),
    "Vocals · GaboxR67 fv6":                         ("GaboxR67/MelBandRoformers",                        "melbandroformers/vocals/voc_fv6.ckpt"),
    "Vocals small · pcunwa":                         ("pcunwa/Mel-Band-Roformer-small",                   "melband_roformer_small_v1.ckpt"),
    # ── Instrumental ────────────────────────────────────────────────────────
    "Instrumental · becruily":                       ("becruily/mel-band-roformer-instrumental",          "mel_band_roformer_instrumental_becruily.ckpt"),
    "Instrumental v2 (depth-12) · pcunwa ⭐":        ("pcunwa/Mel-Band-Roformer-Inst",                    "melband_roformer_inst_v2.ckpt"),
    "Instrumental v1 · pcunwa":                      ("pcunwa/Mel-Band-Roformer-Inst",                    "melband_roformer_inst_v1.ckpt"),
    "Instrumental · GaboxR67 INSTV6 ⭐":             ("GaboxR67/MelBandRoformers",                        "melbandroformers/instrumental/INSTV6.ckpt"),
    "Instrumental · GaboxR67 Fv9":                   ("GaboxR67/MelBandRoformers",                        "melbandroformers/instrumental/Inst_GaboxFv9.ckpt"),
    "Instrumental · GaboxR67 Fv8":                   ("GaboxR67/MelBandRoformers",                        "melbandroformers/instrumental/Inst_GaboxFv8.ckpt"),
    # ── Big models (dim=512) ─────────────────────────────────────────────────
    "Vocals big beta6 (dim=512) · pcunwa ⭐":        ("pcunwa/Mel-Band-Roformer-big",                     "big_beta6.ckpt"),
    "Vocals big beta6x (dim=512) · pcunwa":          ("pcunwa/Mel-Band-Roformer-big",                     "big_beta6x.ckpt"),
    "Vocals big beta7 · pcunwa":                     ("pcunwa/Mel-Band-Roformer-big",                     "big_beta7.ckpt"),
    # ── Karaoke (lead-vocal removal) ────────────────────────────────────────
    "Karaoke · aufr33/viperx ⭐":                    ("jarredou/aufr33-viperx-karaoke-melroformer-model", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"),
    "Karaoke · becruily (2-stem)":                   ("becruily/mel-band-roformer-karaoke",               "mel_band_roformer_karaoke_becruily.ckpt"),
    "Karaoke · GaboxR67 V1":                         ("GaboxR67/MelBandRoformers",                        "melbandroformers/karaoke/Karaoke_GaboxV1.ckpt"),
    # ── Vocals + Instrumental 2-stem ────────────────────────────────────────
    "Vocals+Instrumental 2-stem · becruily":         ("becruily/mel-band-roformer-deux",                  "becruily_deux.ckpt"),
    # ── 4-stem (vocals, drums, bass, other) ─────────────────────────────────
    "4-stem large · Aname-Tommy [stem_1=vox only]":  ("Aname-Tommy/melbandroformer4stems",                "mel_band_roformer_4stems_large_ver1.ckpt"),
    "4-stem XL · Aname-Tommy [stem_1=vox only]":     ("Aname-Tommy/melbandroformer4stems",                "mel_band_roformer_4stems_xl_ver1.ckpt"),
    # ── Dereverb / Echo removal ─────────────────────────────────────────────
    "Dereverb · anvuew ⭐ (SDR 19.17)":              ("anvuew/dereverb_mel_band_roformer",                "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"),
    "Dereverb less-aggressive · anvuew (SDR 18.80)": ("anvuew/dereverb_mel_band_roformer",                "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt"),
    "Dereverb mono-optimized · anvuew (SDR 20.40)":  ("anvuew/dereverb_mel_band_roformer",                "dereverb_mel_band_roformer_mono_anvuew_sdr_20.4029.ckpt"),
    "Dereverb+Echo v2 · Sucial":                     ("Sucial/Dereverb-Echo_Mel_Band_Roformer",           "dereverb_echo_mbr_v2_sdr_dry_13.4843.ckpt"),
    "Dereverb+Echo fused · Sucial":                  ("Sucial/Dereverb-Echo_Mel_Band_Roformer",           "dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt"),
    "Dereverb big reverb · Sucial":                  ("Sucial/Dereverb-Echo_Mel_Band_Roformer",           "de_big_reverb_mbr_ep_362.ckpt"),
    "Dereverb super-big reverb · Sucial":            ("Sucial/Dereverb-Echo_Mel_Band_Roformer",           "de_super_big_reverb_mbr_ep_346.ckpt"),
    "Dereverb+Echo v1 · Sucial":                     ("Sucial/Dereverb-Echo_Mel_Band_Roformer",           "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt"),
    # ── Denoise ─────────────────────────────────────────────────────────────
    "Denoise · aufr33 ⭐":                           ("poiqazwsx/melband-roformer-denoise",               "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"),
    "Denoise aggressive · aufr33":                   ("poiqazwsx/melband-roformer-denoise",               "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt"),
    # ── Aspiration (breath/mouth sounds) ────────────────────────────────────
    "Aspiration · Sucial ⭐":                        ("Sucial/Aspiration_Mel_Band_Roformer",              "aspiration_mel_band_roformer_sdr_18.9845.ckpt"),
    "Aspiration less-aggressive · Sucial":           ("Sucial/Aspiration_Mel_Band_Roformer",              "aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt"),
    # ── BS-RoFormer: Vocals (better bass, competitive vocals) ────────────────
    "[BS] Vocals revive v3e ⭐ · pcunwa":            ("pcunwa/BS-Roformer-Revive",                        "bs_roformer_revive3e.ckpt"),
    "[BS] Vocals revive v2 · pcunwa":                ("pcunwa/BS-Roformer-Revive",                        "bs_roformer_revive2.ckpt"),
    "[BS] Vocals revive v1 · pcunwa":                ("pcunwa/BS-Roformer-Revive",                        "bs_roformer_revive.ckpt"),
    # ── BS-RoFormer: Dereverb ────────────────────────────────────────────────
    "[BS] Dereverb · anvuew ⭐ (SDR 22.51)":         ("anvuew/deverb_bs_roformer",                        "dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt"),
}

_ACK_FILE = os.path.join(folder_paths.get_folder_paths("MelBandRoFormer")[0], ".ckpt_risk_acknowledged")

# Latest/best model from each series — shown in the curated loader node.
# Older versions that are superseded are excluded (Kim FT v1, GaboxR67 fv6, etc.)
_LATEST_MODEL_NAMES = frozenset({
    # Vocals
    "Vocals · Kim FT v2 ⭐ [pcunwa]",
    "Vocals · Kim FT v2 bleedless [pcunwa]",
    "Vocals · Kim fp16 ⭐ [Kijai]",
    "Vocals · becruily",
    "Vocals · GaboxR67 fv7",
    "Vocals small · pcunwa",
    # Instrumental
    "Instrumental · GaboxR67 INSTV6 ⭐",
    "Instrumental v2 (depth-12) · pcunwa ⭐",
    "Instrumental · becruily",
    # Big (dim=512)
    "Vocals big beta6 (dim=512) · pcunwa ⭐",
    "Vocals big beta7 · pcunwa",
    # Karaoke
    "Karaoke · aufr33/viperx ⭐",
    "Karaoke · becruily (2-stem)",
    "Karaoke · GaboxR67 V1",
    # 2-stem direct
    "Vocals+Instrumental 2-stem · becruily",
    # 4-stem
    "4-stem large · Aname-Tommy [stem_1=vox only]",
    "4-stem XL · Aname-Tommy [stem_1=vox only]",
    # Dereverb
    "[BS] Dereverb · anvuew ⭐ (SDR 22.51)",
    "Dereverb · anvuew ⭐ (SDR 19.17)",
    "Dereverb mono-optimized · anvuew (SDR 20.40)",
    "Dereverb less-aggressive · anvuew (SDR 18.80)",
    "Dereverb+Echo v2 · Sucial",
    "Dereverb big reverb · Sucial",
    "Dereverb super-big reverb · Sucial",
    "Dereverb+Echo fused · Sucial",
    # Denoise
    "Denoise · aufr33 ⭐",
    "Denoise aggressive · aufr33",
    # Aspiration
    "Aspiration · Sucial ⭐",
    "Aspiration less-aggressive · Sucial",
    # BS-RoFormer vocals
    "[BS] Vocals revive v3e ⭐ · pcunwa",
})


def _ckpt_acknowledged():
    return os.path.exists(_ACK_FILE)


def _save_ckpt_ack():
    with open(_ACK_FILE, "w") as f:
        f.write("acknowledged")


# Basenames of every file managed by the registry.
# Used to suppress local duplicates: if a model was downloaded via [HF],
# its bare filename is hidden from the local list so it only appears once.
_REGISTRY_FILENAMES = frozenset(
    os.path.basename(filename) for _, filename in MODEL_REGISTRY.values()
)


_MODEL_EXTENSIONS = {".ckpt", ".safetensors", ".pt", ".pth"}


def _manual_local_choices():
    """Local model files not managed by the registry.

    Filters out:
    - Non-model files (.metadata, .json, .ckpt_risk_acknowledged, etc.)
    - Hidden files / dotfiles
    - Files whose basename matches a registry entry (already shown as [HF])
    """
    result = []
    for f in folder_paths.get_filename_list("MelBandRoFormer"):
        base = os.path.basename(f)
        if base.startswith("."):
            continue
        if os.path.splitext(base)[1].lower() not in _MODEL_EXTENSIONS:
            continue
        if base in _REGISTRY_FILENAMES:
            continue
        result.append(f)
    return result


def _hf_model_choices():
    return list(MODEL_REGISTRY.keys())


def _latest_hf_model_choices():
    return [name for name in MODEL_REGISTRY if name in _LATEST_MODEL_NAMES]


def _all_model_choices():
    return _manual_local_choices() + _hf_model_choices()


def _latest_model_choices():
    return _manual_local_choices() + _latest_hf_model_choices()


def _detect_model_type(sd):
    """Return 'melband' or 'bsroformer' by checking whether band-split is overlapping."""
    band_weights = [v for k, v in sd.items()
                    if k.startswith('band_split.to_features.') and k.endswith('.1.weight')]
    if not band_weights:
        return 'melband'  # safe fallback
    total_input = sum(w.shape[1] for w in band_weights)
    # BSRoformer: non-overlapping bands → total = 2 * audio_channels * dim_freqs_in
    # For stereo+complex: 4 * 1025 = 4100
    # MelBandRoformer: overlapping mel bands → total > 4100 (typically ~6000-8000)
    return 'bsroformer' if total_input <= 4 * 1025 else 'melband'


def _infer_shared_params(sd, config):
    """Fill dim, depth, num_stems, transformer depths, mask_estimator_depth — shared logic."""
    config["dim"] = sd["band_split.to_features.0.1.weight"].shape[0]
    config["depth"] = max(int(k.split('.')[1]) for k in sd if k.startswith('layers.')) + 1
    config["num_stems"] = max(int(k.split('.')[1]) for k in sd if k.startswith('mask_estimators.')) + 1

    time_keys = [k for k in sd if k.startswith('layers.0.0.layers.')]
    config["time_transformer_depth"] = max(int(k.split('.')[4]) for k in time_keys) + 1 if time_keys else 1

    freq_keys = [k for k in sd if k.startswith('layers.0.1.layers.')]
    config["freq_transformer_depth"] = max(int(k.split('.')[4]) for k in freq_keys) + 1 if freq_keys else 1

    mlp_keys = [k for k in sd if k.startswith('mask_estimators.0.to_freqs.0.0.') and k.endswith('.weight')]
    if mlp_keys:
        config["mask_estimator_depth"] = max(int(k.split('.')[5]) for k in mlp_keys) // 2


def infer_melband_config(sd):
    """Auto-detect MelBandRoformer architecture from state dict."""
    config = dict(_MELBAND_BASE_CONFIG)
    _infer_shared_params(sd, config)
    return config


def infer_bs_roformer_config(sd):
    """Auto-detect BSRoformer architecture from state dict."""
    config = dict(_BS_BASE_CONFIG)
    _infer_shared_params(sd, config)

    # Reconstruct freqs_per_bands from the band_split input layer shapes.
    # Each band N: band_split.to_features.N.1.weight shape = [dim, 2 * freqs[N] * channels]
    band_keys = sorted(
        (int(k.split('.')[3]), k)
        for k in sd
        if k.startswith('band_split.to_features.') and k.endswith('.1.weight')
    )
    # Detect stereo: input divisible by 4 (2 complex × 2 ch) vs 2 (2 complex × 1 ch)
    first_input = sd[band_keys[0][1]].shape[1]
    divisor = 4 if first_input % 4 == 0 else 2
    config["stereo"] = (divisor == 4)
    config["freqs_per_bands"] = tuple(sd[k].shape[1] // divisor for _, k in band_keys)

    return config


def infer_config(sd):
    """Detect model type and return (model_type, config)."""
    model_type = _detect_model_type(sd)
    if model_type == 'bsroformer':
        return 'bsroformer', infer_bs_roformer_config(sd)
    return 'melband', infer_melband_config(sd)


def download_hf_model(repo_id, filename):
    """Download a model from HuggingFace into ComfyUI's diffusion_models folder."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for auto-download. "
            "Install with:  pip install huggingface_hub"
        )
    save_dir = folder_paths.get_folder_paths("MelBandRoFormer")[0]
    save_path = os.path.join(save_dir, filename)
    if not os.path.exists(save_path):
        print(f"[MelBandRoFormer] Downloading {filename} from {repo_id} ...")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=save_dir)
        print(f"[MelBandRoFormer] Saved to {save_path}")
    return save_path


def get_windowing_array(window_size, fade_size, device):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

class MelBandRoFormerModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    _all_model_choices(),
                    {
                        "tooltip": (
                            "Local files from ComfyUI/models/MelBandRoFormer/ come first. "
                            "Registry models are downloaded automatically from HuggingFace on first use."
                        ),
                    },
                ),
                "acknowledge_ckpt_risk": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Most models use the .ckpt format, which is based on Python pickle. "
                        "Pickle can execute arbitrary code when loaded — a malicious .ckpt file "
                        "could run anything on your machine. All models in this registry come from "
                        "known, trusted authors on HuggingFace, so the practical risk is low. "
                        "Check this box to confirm you understand and accept this risk. "
                        "Your acknowledgment is saved to disk and won't be asked again."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MELROFORMERMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Mel-Band RoFormer"

    def loadmodel(self, model_name, acknowledge_ckpt_risk=False):
        if model_name in MODEL_REGISTRY:
            repo_id, filename = MODEL_REGISTRY[model_name]
            if filename.endswith(".ckpt") and not _ckpt_acknowledged():
                if not acknowledge_ckpt_risk:
                    raise ValueError(
                        "[MelBandRoFormer] This model uses the .ckpt format, which is based on Python pickle.\n"
                        "Pickle files can execute arbitrary code when loaded — a malicious .ckpt could run "
                        "anything on your machine.\n"
                        "The models in this registry come from known, trusted authors on HuggingFace, "
                        "so the practical risk is low, but you should be aware of it.\n\n"
                        "To proceed: check the 'acknowledge_ckpt_risk' box on the Loader node and run again. "
                        "Your acknowledgment will be saved and you won't be asked again."
                    )
                _save_ckpt_ack()
                print("[MelBandRoFormer] .ckpt risk acknowledged and saved.")
            model_path = download_hf_model(repo_id, filename)
        else:
            model_path = folder_paths.get_full_path_or_raise("MelBandRoFormer", model_name)

        sd = load_torch_file(model_path)
        model_type, config = infer_config(sd)
        print(f"[MelBandRoFormer] Detected {model_type}: dim={config['dim']}, depth={config['depth']}, "
              f"num_stems={config['num_stems']}, time_depth={config['time_transformer_depth']}, "
              f"freq_depth={config['freq_transformer_depth']}")

        if model_type == 'bsroformer':
            if not _BS_ROFORMER_AVAILABLE:
                raise ImportError(
                    "BS-RoFormer model requires the bs_roformer package. "
                    "Install with:  pip install BS-RoFormer"
                )
            model = BSRoformer(**config).eval()
        else:
            model = MelBandRoformer(**config).eval()

        model.load_state_dict(sd, strict=True)
        return (model,)


class MelBandRoFormerSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MELROFORMERMODEL",),
                "audio": ("AUDIO",),
                "chunk_size": ("FLOAT", {
                    "default": 8.0, "min": 1.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Chunk size in seconds. Smaller = less VRAM, larger = better quality on sustained sounds.",
                }),
                "overlap": ("INT", {
                    "default": 2, "min": 2, "max": 8, "step": 1,
                    "tooltip": "Overlap factor. Higher = smoother transitions but slower.",
                }),
                "fade_size": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01,
                    "tooltip": "Crossfade ratio relative to chunk size. Higher = smoother blending.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Number of chunks processed in parallel. Higher = faster but more VRAM. Start at 1 and increase until you hit OOM.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("stem_1", "stem_2")
    FUNCTION = "process"
    CATEGORY = "Mel-Band RoFormer"

    def process(self, model, audio, chunk_size=8.0, overlap=2, fade_size=0.1, batch_size=1):
        audio_input = audio["waveform"]
        sample_rate = audio["sample_rate"]

        B, audio_channels, audio_length = audio_input.shape
        sr = 44100

        if audio_channels == 1:
            audio_input = audio_input.repeat(1, 2, 1)
            audio_channels = 2
            print("[MelBandRoFormer] Converted mono input to stereo.")

        if sample_rate != sr:
            print(f"[MelBandRoFormer] Resampling {sample_rate} → {sr}")
            audio_input = TAF.resample(audio_input, orig_freq=sample_rate, new_freq=sr)

        audio_input = original_audio = audio_input[0]  # [channels, time]
        audio_length = audio_input.shape[1]            # use resampled length for border logic

        C = int(chunk_size * sr)
        step = C // overlap
        fade_samples = max(1, int(C * fade_size))
        border = C - step

        if audio_length > 2 * border and border > 0:
            audio_input = F.pad(audio_input, (border, border), mode='reflect')

        windowing_array = get_windowing_array(C, fade_samples, device)

        audio_input = audio_input.to(device)
        num_stems = len(model.mask_estimators)
        total_length = audio_input.shape[1]
        chunk_starts = list(range(0, total_length, step))
        num_chunks = len(chunk_starts)

        # accumulators: [num_stems, channels, time]
        acc = torch.zeros(num_stems, *audio_input.shape, dtype=torch.float32, device=device)
        cnt = torch.zeros(num_stems, *audio_input.shape, dtype=torch.float32, device=device)

        model.to(device)
        comfy_pbar = ProgressBar(num_chunks)

        with torch.no_grad():
            for b_start in tqdm(range(0, num_chunks, batch_size), desc="Processing chunks"):
                batch_starts = chunk_starts[b_start:b_start + batch_size]

                # build batch
                parts, lengths = [], []
                for i in batch_starts:
                    part = audio_input[:, i:i + C]
                    length = part.shape[-1]
                    lengths.append(length)
                    if length < C:
                        pad_mode = 'reflect' if length > C // 2 + 1 else 'constant'
                        part = F.pad(part, (0, C - length), mode=pad_mode)
                    parts.append(part)

                batch_in = torch.stack(parts, dim=0)          # [B, channels, C]
                batch_out = model(batch_in)                    # [B, channels, C] or [B, stems, channels, C]
                if num_stems == 1:
                    batch_out = batch_out.unsqueeze(1)         # → [B, 1, channels, C]

                # accumulate each chunk in the batch
                for idx, (i, length) in enumerate(zip(batch_starts, lengths)):
                    out = batch_out[idx]                       # [stems, channels, C]
                    window = windowing_array.clone()
                    if i == 0:
                        window[:fade_samples] = 1
                    if i + C >= total_length:
                        window[-fade_samples:] = 1

                    acc[..., i:i + length] += out[..., :length] * window[..., :length]
                    cnt[..., i:i + length] += window[..., :length]

                comfy_pbar.update(len(batch_starts))

        model.to(offload_device)

        estimated = acc / cnt.clamp(min=1e-8)   # [num_stems, channels, time]

        if audio_length > 2 * border and border > 0:
            estimated = estimated[..., border:-border]

        stem1 = estimated[0]
        if num_stems >= 2:
            stem2 = estimated[1]
            if num_stems > 2:
                print(f"[MelBandRoFormer] Model has {num_stems} stems; only stem_1 and stem_2 are output. "
                      f"Stems 3–{num_stems} are discarded.")
        else:
            stem2 = original_audio.to(device) - stem1

        def to_audio(t):
            return {"waveform": t.unsqueeze(0).cpu(), "sample_rate": sr}

        return (to_audio(stem1), to_audio(stem2))


def _audio_to_mono(audio):
    """Return (mono_wav [T], sample_rate) from a ComfyUI AUDIO dict."""
    waveform = audio["waveform"]  # [B, C, T]
    sr = audio["sample_rate"]
    wav = waveform[0].mean(0).cpu().float()  # [T]
    return wav, sr


_DB_FLOOR = -80.0     # dB floor — bins below this relative to peak are black
_LOG_FREQ_BINS = 256  # output rows after log-frequency resampling


def _db_spectrogram(wav, n_fft, hop_length):
    """Compute dB-magnitude spectrogram [freq, time], floored at _DB_FLOOR below peak."""
    import numpy as np
    hop_length = min(hop_length, n_fft)   # torch.stft requires hop_length <= n_fft
    window = torch.hann_window(n_fft)
    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop_length,
                      window=window, return_complex=True)
    mag = stft.abs().numpy()
    db = 20.0 * np.log10(np.maximum(mag, 1e-8))
    db = np.maximum(db, db.max() + _DB_FLOOR)
    return db.astype(np.float32)


def _to_log_freq(spec, n_out=_LOG_FREQ_BINS):
    """Resample linear-frequency spectrogram to log-spaced frequency bins."""
    import numpy as np
    n_freqs, n_t = spec.shape
    # Log-spaced source indices from bin 1 (avoid DC) to n_freqs-1
    src_idx = np.logspace(0, np.log10(max(n_freqs - 1, 2)), n_out)
    lo = np.floor(src_idx).astype(int).clip(0, n_freqs - 2)
    hi = lo + 1
    frac = (src_idx - lo)[:, None]                          # [n_out, 1]
    return ((1 - frac) * spec[lo] + frac * spec[hi]).astype(np.float32)


def _log_freq_yticks(n_fft, n_out=_LOG_FREQ_BINS, sr=44100):
    """Tick positions/labels for a log-freq resampled, vertically-flipped spectrogram."""
    import numpy as np
    n_freqs = n_fft // 2 + 1
    src_idx = np.logspace(0, np.log10(max(n_freqs - 1, 2)), n_out)
    target_hz = [100, 500, 1000, 2000, 4000, 8000, 16000]
    positions, labels = [], []
    for hz in target_hz:
        bin_f = hz * n_fft / sr
        if bin_f < 1 or bin_f >= n_freqs:
            continue
        pos = int(np.searchsorted(src_idx, bin_f))
        pos = min(pos, n_out - 1)
        positions.append(n_out - 1 - pos)   # after [::-1] flip
        labels.append(f"{hz // 1000}k" if hz >= 1000 else str(hz))
    return positions, labels


def _shared_vrange(*specs, percentile_lo=2.0):
    """Compute vmin/vmax across all specs for a shared color scale."""
    import numpy as np
    combined = np.concatenate([s.ravel() for s in specs])
    return float(np.percentile(combined, percentile_lo)), float(np.percentile(combined, 99.5))


def _draw_spec(ax, spec, vmin, vmax, cmap, tpos, tlbl, xlabel=False):
    im = ax.imshow(spec, aspect="auto", cmap=cmap, origin="upper",
                   vmin=vmin, vmax=vmax, interpolation="antialiased")
    ax.set_yticks(tpos)
    ax.set_yticklabels(tlbl, fontsize=8)
    ax.set_ylabel("Hz", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    if xlabel:
        ax.set_xlabel("Time frames", fontsize=9)
    return im


def _render_figure(spec_a, spec_b, label_a, label_b, mode, n_fft, sr=44100):
    """Render comparison figure → numpy [H, W, 3] uint8."""
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    min_t = min(spec_a.shape[1], spec_b.shape[1])

    # Resample to log-frequency and flip so low Hz is at bottom
    sa = _to_log_freq(spec_a[:, :min_t])[::-1]
    sb = _to_log_freq(spec_b[:, :min_t])[::-1]

    tpos, tlbl = _log_freq_yticks(n_fft, sr=sr)
    CMAP = "inferno"
    DPI = 150

    if mode == "stacked":
        vmin, vmax = _shared_vrange(sa, sb)
        fig = Figure(figsize=(14, 6), dpi=DPI, tight_layout=True)
        ax_a = fig.add_subplot(2, 1, 1)
        ax_b = fig.add_subplot(2, 1, 2)
        im_a = _draw_spec(ax_a, sa, vmin, vmax, CMAP, tpos, tlbl)
        ax_a.set_title(label_a, fontsize=10)
        im = _draw_spec(ax_b, sb, vmin, vmax, CMAP, tpos, tlbl, xlabel=True)
        ax_b.set_title(label_b, fontsize=10)
        fig.colorbar(im_a, ax=ax_a, label="dB", fraction=0.02, pad=0.01)
        fig.colorbar(im, ax=ax_b, label="dB", fraction=0.02, pad=0.01)

    elif mode == "difference":
        diff = sa - sb
        abs_max = float(np.percentile(np.abs(diff), 99.5)) + 1e-8
        fig = Figure(figsize=(14, 4), dpi=DPI, tight_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", origin="upper",
                       vmin=-abs_max, vmax=abs_max, interpolation="antialiased")
        ax.set_yticks(tpos)
        ax.set_yticklabels(tlbl, fontsize=8)
        ax.set_ylabel("Hz", fontsize=9)
        ax.set_xlabel("Time frames", fontsize=9)
        ax.set_title(f"[{label_a}] − [{label_b}]  ·  red = A louder  ·  blue = B louder", fontsize=10)
        fig.colorbar(im, ax=ax, label="ΔdB", fraction=0.02, pad=0.01)

    else:  # stacked + difference
        vmin, vmax = _shared_vrange(sa, sb)
        diff = sa - sb
        abs_max = float(np.percentile(np.abs(diff), 99.5)) + 1e-8
        fig = Figure(figsize=(14, 9), dpi=DPI, tight_layout=True)
        ax_a = fig.add_subplot(3, 1, 1)
        ax_b = fig.add_subplot(3, 1, 2)
        ax_d = fig.add_subplot(3, 1, 3)
        im_a = _draw_spec(ax_a, sa, vmin, vmax, CMAP, tpos, tlbl)
        ax_a.set_title(label_a, fontsize=10)
        im_b = _draw_spec(ax_b, sb, vmin, vmax, CMAP, tpos, tlbl)
        ax_b.set_title(label_b, fontsize=10)
        fig.colorbar(im_a, ax=ax_a, label="dB", fraction=0.015, pad=0.01)
        fig.colorbar(im_b, ax=ax_b, label="dB", fraction=0.015, pad=0.01)
        im_d = ax_d.imshow(diff, aspect="auto", cmap="RdBu_r", origin="upper",
                           vmin=-abs_max, vmax=abs_max, interpolation="antialiased")
        ax_d.set_yticks(tpos)
        ax_d.set_yticklabels(tlbl, fontsize=8)
        ax_d.set_ylabel("Hz", fontsize=9)
        ax_d.set_xlabel("Time frames", fontsize=9)
        ax_d.set_title(f"[{label_a}] − [{label_b}]", fontsize=10)
        fig.colorbar(im_d, ax=ax_d, label="ΔdB", fraction=0.015, pad=0.01)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)[..., :3].copy()
    return img


class MelBandRoFormerSpectrogram:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_a": ("AUDIO", {}),
                "audio_b": ("AUDIO", {}),
                "label_a": ("STRING", {"default": "A"}),
                "label_b": ("STRING", {"default": "B"}),
                "mode": (["stacked", "difference", "stacked + difference"],),
                "n_fft": ("INT", {
                    "default": 2048, "min": 512, "max": 8192, "step": 512,
                    "tooltip": "FFT size. Larger = better frequency resolution, lower time resolution.",
                }),
                "hop_length": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 64,
                    "tooltip": "Hop size in samples. Smaller = better time resolution.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("spectrogram",)
    FUNCTION = "compare"
    CATEGORY = "Mel-Band RoFormer"

    def compare(self, audio_a, audio_b, label_a, label_b, mode, n_fft, hop_length):
        sr_target = 44100

        wav_a, sr_a = _audio_to_mono(audio_a)
        wav_b, sr_b = _audio_to_mono(audio_b)

        if sr_a != sr_target:
            wav_a = TAF.resample(wav_a.unsqueeze(0), sr_a, sr_target).squeeze(0)
        if sr_b != sr_target:
            wav_b = TAF.resample(wav_b.unsqueeze(0), sr_b, sr_target).squeeze(0)

        spec_a = _db_spectrogram(wav_a, n_fft, hop_length)
        spec_b = _db_spectrogram(wav_b, n_fft, hop_length)

        img = _render_figure(spec_a, spec_b, label_a, label_b, mode, n_fft, sr=sr_target)

        img_tensor = torch.from_numpy(img).float() / 255.0  # [H, W, 3]
        img_tensor = img_tensor.unsqueeze(0)                 # [1, H, W, 3]
        return (img_tensor,)


class MelBandRoFormerModelLoaderLatest(MelBandRoFormerModelLoader):
    """Curated loader — shows only the latest/best model from each series."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    _latest_model_choices(),
                    {
                        "tooltip": (
                            "Curated list showing only the latest or best model from each series. "
                            "Older superseded versions are hidden. "
                            "Use the full Model Loader node to access every available model. "
                            "Local files appear first; registry models auto-download on first use."
                        ),
                    },
                ),
                "acknowledge_ckpt_risk": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Most models use the .ckpt format, which is based on Python pickle. "
                        "Pickle can execute arbitrary code when loaded — a malicious .ckpt file "
                        "could run anything on your machine. All models in this registry come from "
                        "known, trusted authors on HuggingFace, so the practical risk is low. "
                        "Check this box to confirm you understand and accept this risk. "
                        "Your acknowledgment is saved to disk and won't be asked again."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MELROFORMERMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Mel-Band RoFormer"


NODE_CLASS_MAPPINGS = {
    "MelBandRoFormerModelLoader": MelBandRoFormerModelLoader,
    "MelBandRoFormerModelLoaderLatest": MelBandRoFormerModelLoaderLatest,
    "MelBandRoFormerSampler": MelBandRoFormerSampler,
    "MelBandRoFormerSpectrogram": MelBandRoFormerSpectrogram,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MelBandRoFormerModelLoader": "Mel-Band RoFormer Model Loader",
    "MelBandRoFormerModelLoaderLatest": "Mel-Band RoFormer Model Loader (Latest)",
    "MelBandRoFormerSampler": "Mel-Band RoFormer Sampler",
    "MelBandRoFormerSpectrogram": "Mel-Band RoFormer Spectrogram",
}
