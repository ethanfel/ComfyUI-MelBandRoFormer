# ComfyUI-MelBandRoFormer

ComfyUI nodes for **Mel-Band RoFormer** and **BS-RoFormer** — state-of-the-art audio source separation models. Split audio into vocals and instruments, remove reverb, denoise recordings, isolate breath sounds, and more.

Both architectures are supported from a single loader node — the checkpoint is automatically detected.

Based on the papers:
- [Mel-Band RoFormer for Music Source Separation](https://arxiv.org/abs/2310.01809) (Lu et al., 2023)
- [Music Source Separation with Band-Split RoFormer](https://arxiv.org/abs/2309.02612) (Yu et al., 2023)

![Workflow example](https://github.com/user-attachments/assets/05468504-b1b8-41da-8453-ae61e2c56a53)

---

## Installation

### Via ComfyUI Manager (recommended)
Search for **ComfyUI-MelBandRoFormer** in the Manager and install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ethanfel/ComfyUI-MelBandRoFormer
pip install huggingface_hub
```

### Models
Models are stored in `ComfyUI/models/MelBandRoFormer/` (created automatically on first run).

**You don't need to download anything manually.** In the loader node, any entry starting with `[HF]` downloads automatically from HuggingFace the first time you run it, then loads from disk on subsequent runs.

To use a model you downloaded yourself, drop the `.ckpt` or `.safetensors` file into `ComfyUI/models/MelBandRoFormer/` and it will appear at the top of the dropdown.

---

## Which model should I pick?

| I want to… | Recommended model |
|---|---|
| Extract vocals from a song | **Vocals · Kim FT v2 ⭐** |
| Extract vocals with minimal instrument bleed | **Vocals · Kim FT v2 bleedless** |
| Best vocal quality, BS-RoFormer architecture | **[BS] Vocals revive v3e · pcunwa** |
| Extract the instrumental / backing track | **Instrumental · GaboxR67 INSTV6 ⭐** |
| Make a karaoke track (remove lead vocal) | **Karaoke · aufr33/viperx ⭐** |
| Remove room reverb from a recording | **[BS] Dereverb · anvuew (SDR 22.51)** |
| Denoise a recording | **Denoise · aufr33 ⭐** |
| Isolate breath / mouth sounds | **Aspiration · Sucial ⭐** |
| Separate vocals + drums + bass + other | **4-stem large · Aname-Tommy** |
| Low VRAM / fast preview | **Vocals · Kim fp16** |
| Highest possible quality, have lots of VRAM | **Vocals big beta6 (dim=512) · pcunwa** |

---

## Nodes

### Mel-Band RoFormer Model Loader

Loads a model from your local folder or downloads it automatically from HuggingFace.

| Input | Description |
|---|---|
| `model_name` | Local filenames appear first. `[HF] …` entries auto-download on first use. |

**Output:** a `MELROFORMERMODEL` — connect it to the Sampler node.

The loader automatically detects whether the checkpoint is a **Mel-Band RoFormer** or **BS-RoFormer** model and instantiates the correct architecture. No configuration needed.

---

### Mel-Band RoFormer Sampler

Runs the separation and returns two audio streams.

| Input | Default | Description |
|---|---|---|
| `model` | — | Connect from the Loader node. |
| `audio` | — | Any ComfyUI `AUDIO` input. |
| `chunk_size` | 8.0 s | Audio is processed in overlapping chunks. Larger = better quality on long sustained sounds, but more VRAM. Reduce if you get out-of-memory errors. |
| `overlap` | 2 | How many chunks overlap. Higher = smoother crossfades between chunks, but slower. 2 is fine for most use cases; try 4 for very clean results. |
| `fade_size` | 0.1 | Fraction of each chunk used for crossfading (0.01–0.5). Higher = smoother blending at chunk boundaries. |
| `batch_size` | 1 | Process N chunks simultaneously. Higher = faster, but uses more VRAM. Start at 1 and increase until you hit an out-of-memory error. |

| Output | Description |
|---|---|
| `stem_1` | The primary separated audio (e.g. vocals for a vocal model, dry signal for a dereverb model). |
| `stem_2` | The residual (original minus stem_1) for single-stem models, or the second stem for two-stem models (karaoke, aspiration). |

> **4-stem models:** Only `stem_1` (vocals) and `stem_2` (drums) are output. Stems 3 and 4 (bass, other) are discarded with a console warning.

---

## Model Catalog

### Vocals

Extract the vocal track from a song.

| Model | SDR | Notes | Quality |
|---|---|---|---|
| **Vocals · Kim FT v2** | — | Finetuned version of the Kim model, best general-purpose vocal separator | ⭐⭐⭐⭐⭐ Best overall |
| **Vocals · Kim FT v2 bleedless** | — | Same as FT v2 but tuned to reduce instrument bleed into the vocal output | ⭐⭐⭐⭐⭐ Best for clean isolation |
| **Vocals · Kim FT v1** | — | Earlier finetune, slightly less refined than v2 | ⭐⭐⭐⭐ |
| **Vocals · Kim fp16** | 10.98 | Original Kim model, fp16 precision — fastest, lowest VRAM | ⭐⭐⭐⭐ Best for speed/VRAM |
| **Vocals · Kim fp32** | 10.98 | Same, full precision — marginally higher quality at double the VRAM | ⭐⭐⭐⭐ |
| **Vocals · Kim original** | 10.98 | Original checkpoint from KimberleyJSN | ⭐⭐⭐⭐ |
| **Vocals · becruily** | — | Community-trained alternative | ⭐⭐⭐ |
| **Vocals · GaboxR67 fv7** | — | Community model, dim=256/depth=12 — deep but narrow | ⭐⭐⭐ |
| **Vocals · GaboxR67 fv6** | — | Previous iteration of the fv series | ⭐⭐⭐ |
| **Vocals small · pcunwa** | — | Smaller/faster model, good for previewing | ⭐⭐ Fast |
| **Vocals big beta6 (dim=512)** | — | Wider architecture than standard (dim=512 vs 384) — potentially higher quality at the cost of more VRAM | ⭐⭐⭐⭐⭐ Best quality, high VRAM |
| **Vocals big beta6x** | — | Experimental variant of beta6 | ⭐⭐⭐⭐ |
| **Vocals big beta7** | — | Latest big series, depth=8 | ⭐⭐⭐⭐ |

---

### BS-RoFormer Vocals

BS-RoFormer uses non-overlapping linear frequency bands instead of overlapping mel bands. It is slightly weaker than the best MelBandRoFormer models on vocals alone, but handles bass frequencies more precisely.

| Model | Notes | Quality |
|---|---|---|
| **[BS] Vocals revive v3e · pcunwa** | Latest and best of the Revive series | ⭐⭐⭐⭐⭐ Best BS-RoFormer vocal |
| **[BS] Vocals revive v2 · pcunwa** | Second iteration | ⭐⭐⭐⭐ |
| **[BS] Vocals revive v1 · pcunwa** | Original Revive checkpoint | ⭐⭐⭐ |

`stem_1` = vocals. `stem_2` = residual (instrumental).

---

### Instrumental

Extract the backing track / remove vocals.

| Model | SDR | Notes | Quality |
|---|---|---|---|
| **Instrumental · GaboxR67 INSTV6** | — | Latest and most refined instrumental model from GaboxR67 | ⭐⭐⭐⭐⭐ Best overall |
| **Instrumental · GaboxR67 Fv9** | — | F-series, one step below INSTV6 | ⭐⭐⭐⭐ |
| **Instrumental · GaboxR67 Fv8** | — | F-series v8 | ⭐⭐⭐⭐ |
| **Instrumental v2 (depth-12) · pcunwa** | — | Deeper model (depth=12), potentially captures more complex patterns | ⭐⭐⭐⭐ Best depth |
| **Instrumental v1 · pcunwa** | — | Standard depth baseline | ⭐⭐⭐ |
| **Instrumental · becruily** | — | Community alternative | ⭐⭐⭐ |

---

### Karaoke

Remove the lead vocal while keeping backing vocals and instruments.

| Model | SDR | Notes | Quality |
|---|---|---|---|
| **Karaoke · aufr33/viperx** | 10.20 | The standard karaoke model, widely used and validated | ⭐⭐⭐⭐⭐ Best overall |
| **Karaoke · becruily (2-stem)** | — | 2-stem model: stem_1 = vocals, stem_2 = karaoke directly | ⭐⭐⭐⭐ |
| **Karaoke · GaboxR67 V1** | — | Community alternative | ⭐⭐⭐ |

---

### Vocals + Instrumental (2-stem direct)

Models that output both vocals and instrumental simultaneously rather than computing one as the residual of the other.

| Model | Vocals SDR | Instrumental SDR | Quality |
|---|---|---|---|
| **Vocals+Instrumental 2-stem · becruily** | 11.37 | 17.55 | ⭐⭐⭐⭐⭐ Best dual-output |

`stem_1` = vocals, `stem_2` = instrumental.

---

### 4-Stem Separation

Separates audio into vocals, drums, bass, and other simultaneously.

> **Note:** The current sampler node only outputs `stem_1` (vocals) and `stem_2` (drums). Bass and other are discarded. Full 4-stem support is planned.

| Model | Size | Notes | Quality |
|---|---|---|---|
| **4-stem large · Aname-Tommy** | 3.76 GB | Good balance of quality and VRAM usage | ⭐⭐⭐⭐ Recommended |
| **4-stem XL · Aname-Tommy** | 6.41 GB | Higher capacity — requires significant VRAM | ⭐⭐⭐⭐⭐ Best quality |

MelBandRoFormer's mel-band scheme has limited low-frequency resolution, so bass separation is less precise than for vocals or drums. Consider [BS-RoFormer](https://github.com/ZFTurbo/Music-Source-Separation-Training) for bass-critical work.

---

### Dereverb / Echo Removal

Remove room reverb and echo from recordings. Useful for cleaning up vocal takes, speech recordings, and field recordings.

| Model | SDR | Notes | Quality |
|---|---|---|---|
| **[BS] Dereverb · anvuew** | 22.51 | BS-RoFormer model — highest SDR of any dereverb model here | ⭐⭐⭐⭐⭐ Best overall |
| **Dereverb mono-optimized · anvuew** | 20.40 | MelBand — optimized for mono or centered sources | ⭐⭐⭐⭐⭐ Best MelBand SDR |
| **Dereverb · anvuew** | 19.17 | MelBand — general-purpose, best for stereo material | ⭐⭐⭐⭐⭐ |
| **Dereverb less-aggressive · anvuew** | 18.80 | Same model, less processing — good when you want to preserve some room feel | ⭐⭐⭐⭐ |
| **Dereverb+Echo v2 · Sucial** | 13.48 | Trained on singing voice data (opencpop/GTSinger) | ⭐⭐⭐ Good for singing |
| **Dereverb+Echo fused · Sucial** | — | Blend of v2, big, and super-big Sucial models | ⭐⭐⭐ |
| **Dereverb big reverb · Sucial** | — | Tuned for large reverberant spaces | ⭐⭐⭐ Large rooms |
| **Dereverb super-big reverb · Sucial** | — | Tuned for very large reverberant spaces (halls, cathedrals) | ⭐⭐⭐ Halls |
| **Dereverb+Echo v1 · Sucial** | 10.02 | Original Sucial model | ⭐⭐ |

`stem_1` = dry (cleaned) signal. `stem_2` = extracted reverb/echo tail.

---

### Denoise

Remove background noise from recordings.

| Model | SDR | Notes | Quality |
|---|---|---|---|
| **Denoise · aufr33** | 27.99 | Best general-purpose denoiser — strong SDR | ⭐⭐⭐⭐⭐ Best overall |
| **Denoise aggressive · aufr33** | 27.97 | More aggressive noise removal — may remove some wanted signal | ⭐⭐⭐⭐ For heavy noise |

`stem_1` = clean signal. `stem_2` = extracted noise.

---

### Aspiration / Breath Sounds

Isolate or remove breath sounds, aspiration noise, and mouth sounds from vocal recordings.

| Model | SDR | Notes | Quality |
|---|---|---|---|
| **Aspiration · Sucial** | 18.98 | Strongest aspiration removal | ⭐⭐⭐⭐⭐ Best overall |
| **Aspiration less-aggressive · Sucial** | 18.12 | Preserves more of the vocal character | ⭐⭐⭐⭐ More natural |

`stem_1` = dry vocal (aspiration removed). `stem_2` = isolated aspiration/breath signal.

> **Tip for isolating specific mouth sounds (e.g. lip smacks, lollipop sounds):** These sounds overlap spectrally with breath and aspiration. Run the Aspiration model first. The resulting `stem_2` will contain the isolated breath-and-mouth-noise component, which you can then process further with EQ or gating.

---

## Tips

### VRAM & Speed

| Setting | Effect |
|---|---|
| ↓ `chunk_size` | Less VRAM, slightly lower quality on long sustained notes |
| ↑ `batch_size` | Faster processing, more VRAM. Increase until you hit OOM, then step back by 1. |
| ↑ `overlap` | Smoother results at chunk boundaries, slower. 2 is fine; 4 for critical work. |

**Starting points by VRAM:**
- 4 GB → `chunk_size=4`, `batch_size=1`
- 8 GB → `chunk_size=8`, `batch_size=2`
- 16 GB → `chunk_size=8`, `batch_size=4`
- 24 GB+ → `chunk_size=12`, `batch_size=8`

### Chaining Models

You can connect the output of one Sampler into a second Sampler for two-stage processing:

- **Vocal cleanup:** Vocal model → Dereverb model on `stem_1`
- **Clean instrument isolation:** Instrumental model → Denoise model on `stem_1`
- **Mouth sound isolation:** Vocal model → Aspiration model on `stem_1` → `stem_2` contains mouth sounds

### Audio Format

- Mono inputs are automatically duplicated to stereo before processing.
- Audio at sample rates other than 44100 Hz is automatically resampled.
- Output is always at 44100 Hz.

---

## Credits & Attribution

### Papers
- **Mel-Band RoFormer** — Lu et al. (2023): [arxiv.org/abs/2310.01809](https://arxiv.org/abs/2310.01809)
- **BS-RoFormer** — Yu et al. (2023): [arxiv.org/abs/2309.02612](https://arxiv.org/abs/2309.02612)

### Original Implementation & Models
- **[KimberleyJensen](https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model)** — original vocal model implementation and checkpoint
- **[Kijai](https://huggingface.co/Kijai/MelBandRoFormer_comfy)** — ComfyUI-optimized safetensors (fp16/fp32)
- **[ZFTurbo](https://github.com/ZFTurbo/Music-Source-Separation-Training)** — training framework used by most models in this registry
- **[lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)** — BS-RoFormer PyPI package used for BS-RoFormer inference

### Model Authors
| Author | Models |
|---|---|
| **[pcunwa](https://huggingface.co/pcunwa)** | Kim FT finetuned variants, Inst v1/v2, big beta series, BS-RoFormer Revive series |
| **[aufr33](https://huggingface.co/poiqazwsx/melband-roformer-denoise) & [viperx](https://huggingface.co/jarredou/aufr33-viperx-karaoke-melroformer-model)** | Karaoke model, denoise models |
| **[Sucial](https://huggingface.co/Sucial)** | Dereverb+Echo models, Aspiration models |
| **[becruily](https://huggingface.co/becruily)** | Vocals, instrumental, karaoke, deux (2-stem) |
| **[anvuew](https://huggingface.co/anvuew)** | High-SDR MelBand dereverb models + BS-RoFormer dereverb (SDR 22.51) |
| **[GaboxR67](https://huggingface.co/GaboxR67/MelBandRoformers)** | Extensive community vocal, instrumental, karaoke variants |
| **[Aname-Tommy](https://huggingface.co/Aname-Tommy/melbandroformer4stems)** | 4-stem large and XL models |

### This Node
Originally forked from [kijai/ComfyUI-MelRoFormer](https://github.com/kijai/ComfyUI-MelRoFormer). Extended with multi-model support, HuggingFace auto-download, architecture auto-detection, batched inference, and multi-stem output by [ethanfel](https://github.com/ethanfel).
