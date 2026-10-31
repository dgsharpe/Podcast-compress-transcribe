# Neural codecs for general audio and music: state of the art as of late 2026

Scope: full-band (44.1/48 kHz) general-audio and music codecs and autoencoders. Speech-only low-bitrate codecs, production standards, and benchmark/open-weight surveys are covered by other researchers. Each result is dated. Unless stated otherwise, quality numbers are the authors' own claims from their own papers. Most papers use different test sets, so numbers from different papers are not directly comparable.

## 1. Leading general-audio/music codecs: bitrates and reported quality

### Takeaway
The open, widely used baselines are still DAC (2023, 44.1 kHz, up to 8 kbps mono) and EnCodec (2022, 48 kHz stereo music model, 3–24 kbps). Newer models report better results mainly at very low rates (2–16 kbps), where the codec is used as a tokenizer: Google DeepMind's SpectroStream (Aug 2025), Meta's FlowDec (ICLR 2025), CoDiCodec (ISMIR 2025), Adobe's GenAE (Feb 2026), and the codec in Alibaba's Qwen-Audio-3.0 (Jul 2026). The only 2026 paper I found that explicitly targets streaming-grade bitrates (32–128 kbps) is Tencent Music's TQCodec. It reports LSD/SNR plus one informal preference test, not MUSHRA. The open neural codecs from NVIDIA I found are trained on speech only.

### Cited Findings
**Classic baselines (pre-2025)**
- EnCodec (Meta, Oct 2022), 24 kHz mono, streamable MUSHRA. Music Set-1 / Set-2 scores:
  - EnCodec 3 kbps: 89.6 / 87.8. EnCodec 6 kbps: 92.9 / 91.3. EnCodec 12 kbps: 91.8 / 92.9.
  - Opus 6 kbps: 20.6 / 17.9. Opus 12 kbps: 77.8 / 65.4. EVS 9.6 kbps: 89.9 / 87.7. Lyra-v2 6 kbps: 75.7 / 48.6.
  - Reference: 93.2 / 97.1.
  - Entropy coding with a small LM cuts the actual rate, e.g. 6 kbps becomes 4.1 kbps. — [Défossez et al., High Fidelity Neural Audio Compression, arXiv 2210.13438](https://arxiv.org/pdf/2210.13438)
- EnCodec, 48 kHz stereo music MUSHRA (Table 4, 2022): Reference 95.1, MP3 64 kbps 82.7, Opus 6 kbps 17.7, Opus 24 kbps 82.9, EnCodec 6 kbps 82.9, EnCodec 12 kbps 88.0, EnCodec 24 kbps 87.5. The authors state that "EnCodec at 6 kbps … is comparable to MP3 at 64 kbps" and that 12 kbps ≈ 24 kbps. — [arXiv 2210.13438](https://arxiv.org/pdf/2210.13438)
- DAC / Improved RVQGAN (Descript, Jun 2023), 44.1 kHz, 86 Hz frames, up to 8 kbps.
  - Reports much higher MUSHRA than EnCodec at all bitrates. A secondary summary says DAC at 8 kbps beats EnCodec at 12 kbps.
  - The authors concede that even at 8 kbps DAC falls short of the hidden reference. — [arXiv 2306.06546](https://arxiv.org/pdf/2306.06546) (exact MUSHRA values only in figures; not retrieved)
- Gull (2024), subband gain-shape neural codec. MUSHRA with 30 listeners, 15 items from an internal lossless set, 44.1 kHz music:
  - Reference 92.8, MP3 24 kbps 71.0, Opus 24 kbps 92.3, EVS 24.4 kbps 91.8, EnCodec 24 kbps 83.7.
  - Gull-L at 16.8 / 21.6 / 24 kbps: 91.7 / 90.7 / 92.5. Gull-S at the same rates: 85.8 / 89.7 / 90.2.
  - On ViSQOL for music, Opus beats Gull at ≥20 kbps. — [Gull, arXiv 2404.04947v2](https://arxiv.org/html/2404.04947v2)
- APCodec (Feb 2024), 48 kHz, 6 kbps: ViSQOL 4.07. — [arXiv 2402.10533](https://arxiv.org/pdf/2402.10533)
- MDCTCodec (Nov 2024), 48 kHz: ViSQOL 4.18 at 6 kbps and 4.32 at higher rates (6/9/12 kbps tested). Both were measured on VCTK, which is speech, so this is not music evidence. At higher rates DAC and HiFi-Codec beat it on some metrics. — [arXiv 2411.00464](https://arxiv.org/pdf/2411.00464), summarised via search
  - Another paper (STFTCodec) lists APCodec's ViSQOL as 4.224, which conflicts with the original 4.07. The setups differ. — [STFTCodec arXiv 2503.16989](https://arxiv.org/pdf/2503.16989)
- MuCodec (Tencent, Sep 2024), music, 0.35 and 1.33/1.35 kbps (the two sources disagree on the second figure). Uses a Conformer feature extractor, RVQ and flow-matching reconstruction. Claims to beat DAC and SemantiCodec at both rates. I did not retrieve its ViSQOL numbers. — [arXiv 2409.13216](https://www.arxiv.org/pdf/2409.13216)

**2025 models**
- FlowDec (Meta and University of Hamburg, ICLR 2025), 48 kHz general audio.
  - Non-adversarial codec plus a conditional flow-matching postfilter. Down to about 4 kbps; 75 Hz feature rate.
  - FAD better than DAC. MUSHRA "on par with DAC … no significant differences … at any given bitrate."
  - Per category it is better than DAC on speech, slightly worse on sound, and on par on music. Licence is CC-BY-NC. — [FlowDec ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3d937d5e40a883d5da9fd4aeeeb372b8-Abstract-Conference.html), [arXiv 2503.01485](https://arxiv.org/pdf/2503.01485)
- SpectroStream (Google DeepMind, Aug 2025), 48 kHz stereo music.
  - 2D-conv STFT codec, 25 Hz embeddings, 64 RVQ levels of 1024 entries, 4–16 kbps, 61M parameters.
  - Causal, with 80 ms architectural latency; runs in real time on one desktop CPU.
  - ViSQOL vs DAC on 49 MUSDB tracks (bitrates are per channel):

    | Per-channel bitrate | SpectroStream | DAC |
    |---|---|---|
    | 2.7 kbps | 3.21 | 1.47 |
    | 5.3 kbps | 3.83 | 2.41 |
    | 8 kbps | 4.00 | 3.33 |

  - A/B test with 60 raters on 20 MUSDB clips; SpectroStream's win rate vs DAC was 76.3% at 2.7 kbps, 55.0% at 5.3 kbps and 50.8% at 8 kbps.
  - Only DAC is compared; there is no Opus or AAC anchor. — [arXiv 2508.05207](https://arxiv.org/html/2508.05207v1)
  - It is the tokenizer for Magenta RT. Magenta RT runs it at 16 kbps, or 4 kbps by generating only the first 16 RVQ levels. — [Live Music Models, arXiv 2508.04651](https://arxiv.org/pdf/2508.04651)
- CoDiCodec (Sony CSL, ISMIR 2025; successor to Music2Latent / Music2Latent2).
  - One consistency-trained model gives continuous latents at about 11 Hz and FSQ discrete tokens, e.g. 2.38 kbps.
  - Claims better reconstruction than continuous and discrete autoencoders at similar bitrates. — [ISMIR 2025 poster](https://ismir2025program.ismir.net/poster_141.html), [arXiv 2509.09836](https://arxiv.org/abs/2509.09836v1)
  - Third-party measurement shows high mel distance: 0.93 for CoDiCodec vs 0.51 for DAC. The GenAE authors note that "CoDiCodec sounds better than its metrics suggest." — [GenAE arXiv 2602.15749](https://arxiv.org/html/2602.15749v1)
- UniCodec (ACL 2025): single-codebook universal codec at 24 kHz and 75 tokens/s. Details in section 4. — [arXiv 2502.20067](https://arxiv.org/html/2502.20067v1)

**2026 models**
- GenAE (Adobe Research, Feb 2026), 44.1 kHz.
  - Supports mono, stereo and mid/side through a format token. Latents at 13.125 Hz (3360× downsampling, 64-d) or 36.75 Hz.
  - KL (continuous) and RVQ (16×1024) variants share one backbone.
  - Stereo L/R mel distance (lower is better):

    | Model | Latent rate | L/R mel |
    |---|---|---|
    | GenAE-KL | 36.75 Hz | 0.40 |
    | GenAE-KL | 13.125 Hz | 0.54 |
    | GenAE-VQ | 36.75 Hz | 0.47 |
    | GenAE-VQ | 13.125 Hz | 0.60 |
    | DAC | 86 Hz | 0.51 |
    | EnCodec-24 | 75 Hz | 0.61 |
    | EnCodec-48 | 150 Hz | 0.55 |
    | Stable Audio Open | 21.5 Hz | 0.69 |
    | CoDiCodec | 11 Hz | 0.93 |

  - PESQ-WB (speech quality, measured on music content): GenAE-KL 36.75 Hz 4.04, DAC 3.49, SAO 2.76.
  - Encodes about 10× faster than baselines. No MUSHRA, FAD or ViSQOL. — [arXiv 2602.15749](https://arxiv.org/html/2602.15749v1)
- TQCodec (Tencent Music, Mar 2026), 44.1 kHz, 32–128 kbps.
  - SEANet-based with SimVQ and perceptual band-wise bit allocation. Stereo is coded as two separate mono channels; the authors found a joint two-channel network "performed poorly."
  - LSD results:

    | System | LSD |
    |---|---|
    | DAC 16 kbps | 0.927 |
    | TQCodec 32 kbps | 0.822 |
    | TQCodec 64 kbps | 0.772 |
    | TQCodec 128 kbps | 0.672 |

  - Against Ogg Vorbis 48 kbps, TQCodec at 43 kbps has much better high-band LSD (0.954 vs 2.847) but slightly lower SNR (16.07 vs 17.00 dB).
  - Subjective test: the codec (called "AICodec") at 64 kbps vs Ogg at 96 kbps. MOS 4.18; preference 39.1% codec vs 39.4% Ogg, 21.5% no difference.
  - No MUSHRA, and no AAC, Opus or MP3 baseline. — [arXiv 2603.01592](https://arxiv.org/html/2603.01592)
- Qwen-Audio-3.0-Gen-Preview VAE (Alibaba, Jul 2026): continuous VAE, 48 kHz stereo, 25 Hz, 128-d, with SAO-style convolutions. Reconstruction on the Song Describer Dataset (music):

  | Model | Latent rate | ViSQOL | Mel | SI-SDR (dB) | ICPC (stereo phase coherence) |
  |---|---|---|---|---|---|
  | Qwen VAE (acoustic) | 25 Hz | 4.30 | 0.453 | 11.4 | 0.692 |
  | DAC | 86 Hz | 4.20 | 0.872 | 10.9 | 0.726 |
  | EAR-VAE | 43 Hz | 4.06 | 0.459 | 11.9 | – |
  | SAME-L | 43 Hz | 4.01 | 0.504 | 11.8 | – |
  | Stable Audio Open | 21.5 Hz | 4.07 | 0.601 | 6.06 | 0.571 |

  — [arXiv 2607.27011](https://arxiv.org/pdf/2607.27011)
- SwitchCodec (Jan 2026), residual-expert sparse quantization. Per a search snippet, it claims "near-transparent" quality at 2.67 and 5.33 kbps. The comparison set does not include AAC. I could not retrieve the full text, so this is unverified. — [arXiv 2601.20362](https://arxiv.org/pdf/2601.20362)
- Pupu-Codec (Dec 2025; TASLP, v3 Jun 2026): adds differentiable anti-aliasing to activations and upsampling. Claims to beat existing systems on singing voice, music and audio, and to be comparable on speech. Numbers not retrieved. — [arXiv 2512.20211](https://arxiv.org/abs/2512.20211)
- NVIDIA: the HF cards for `nvidia/audio-codec-44khz` (FSQ, 8×1000) and `nvidia/mel-codec-44khz` (86.1 fps, about 6.9 kbps) say they are trained on speech only (14.2k h). The mel codec "has not been trained or evaluated on non-speech audio." — [audio-codec-44khz](https://huggingface.co/nvidia/audio-codec-44khz), [mel-codec-44khz](https://huggingface.co/nvidia/mel-codec-44khz)
- HoliTok (SJTU, Jun 2026), 48 kHz, 25 Hz, 128-d continuous latents. The model card example is mono and stereo is unconfirmed. In the Qwen report's Song Describer table (shown for speech sets), HoliTok was strong on speech: Seed-TTS EN ViSQOL 4.39, PESQ 3.92. — [HF card](https://huggingface.co/bovod-sjtu/HoliTok), [Qwen report](https://arxiv.org/pdf/2607.27011)

### Inferences
- The research frontier has split in two:
  - (a) Low-rate tokenizers and latents for generation, at 10–40 Hz and roughly 1–16 kbps. Nearly all 2025–26 activity is here.
  - (b) A small effort at streaming bitrates (TQCodec). It shows that neural codecs at 32–64 kbps per stereo pair are at least competitive with Vorbis on objective spectral metrics. It does not show this rigorously against AAC or Opus.
- DAC remains the reference baseline in almost every 2025–26 paper, so DAC-relative claims are the most common currency. On music metrics, the low-rate state of the art is roughly SpectroStream (stereo, 4–16 kbps) and Qwen/GenAE-style continuous VAEs at 25–37 Hz.

### Gaps
- I did not retrieve the exact DAC MUSHRA values (in figures only).
- I did not retrieve numbers for SNAC (multi-scale, 2024), Spectral Codecs (NVIDIA 2024), SemantiCodec (2024), WavTokenizer music, Music2Latent2, MuCodec, DisCoder or Pupu-Codec.
- I found no 2025–26 general-audio codec from Apple, and no ByteDance release beyond those named elsewhere. No dedicated new Stability AI codec appeared in my searches; Stability's speech-oriented TAAE appears as a baseline in UniCodec.

## 2. Stereo, high sample rate, and music failure modes

### Takeaway
Stereo is still mostly handled naively, by coding each channel independently (DAC, TQCodec) or with a stereo-trained discriminator (EnCodec). Exceptions are SpectroStream's delayed channel fusion, GenAE's L/R and M/S format token, and the stereo VAEs used for generation, which are now evaluated with phase-coherence metrics. Documented music failure modes are aliasing from nonlinearities and upsampling, loss of cross-channel phase consistency, tonal and high-band smearing, and degradation on vocals and polyphony. These are mostly described qualitatively.

### Cited Findings
- SpectroStream reports that fusing channels too early hurts acoustic quality. Fusing too late "loses phase consistency," yielding "unfocused, directionless" sound. It gives no numbers. — [arXiv 2508.05207](https://arxiv.org/html/2508.05207v1)
- TQCodec found a two-channel network "performed poorly." It codes stereo as dual mono at 32 + 32 kbps for the 64 kbps setting. — [arXiv 2603.01592](https://arxiv.org/html/2603.01592)
- EnCodec moved to stereo "by only modifying our discriminator setup." It is non-causal and music-only at 48 kHz. — [arXiv 2210.13438](https://arxiv.org/pdf/2210.13438)
- Stereo phase coherence (ICPC) on Song Describer: Stable Audio Open's 21.5 Hz VAE scores 0.571, against 0.69–0.73 for DAC, EAR-VAE, SAME-L and the Qwen VAE. Very low-rate continuous latents can therefore degrade stereo image as well as SI-SDR. SAO's SI-SDR is 6.1 dB vs about 11 dB for the others. — [Qwen-Audio-3.0 report, arXiv 2607.27011](https://arxiv.org/pdf/2607.27011)
- The Fraunhofer/ODAQ MUSHRA study (Sep 2025, revised Jan 2026) used 15 stereo excerpts at 48 kHz from ODAQ.
  - DAC was run as dual mono at 16 and 32 kb/s, with resampling 48→44.1→48 kHz. DAC at 16 kb/s beat EnCodec at 24 kb/s.
  - An enhanced DAC (DACe, 48 kHz, about 30 kb/s per channel) beat DAC at 32 kb/s. The biggest gains were on tonal items and a panned speech item, which implies DAC struggles with tonal and panned content. — [arXiv 2509.18823](https://arxiv.org/html/2509.18823)
- Aliasing: "severe aliasing artifacts are introduced by non-linear activation functions and upsampling layers in existing architectures," hurting singing and music. — [Pupu-Codec, arXiv 2512.20211](https://arxiv.org/abs/2512.20211)
- Music is harder than speech for DAC-style vocoders and codecs because of overlapping instruments, complex vocals and wider bandwidth. DisCoder (ICASSP 2025, 44.1 kHz) builds on DAC to reduce these artifacts. — [arXiv 2502.12759](https://arxiv.org/abs/2502.12759)
- TQCodec's design reflects the authors' view of where neural codecs fail at high rates. It adds SimVQ for mid-frequency detail and a phase-aware waveform loss, and allocates bits toward low frequencies. With subbands, its high-band LSD stays near 0.95 at every rate from 32 to 128 kbps while low-band LSD keeps improving, so the high band saturates. — [arXiv 2603.01592](https://arxiv.org/html/2603.01592)
- Sample-rate handling:
  - FlowDec chose 48 kHz so frame rates and bitrates are whole numbers (75 Hz vs DAC's 86.13 Hz at 44.1 kHz). — [FlowDec](https://arxiv.org/pdf/2503.01485)
  - Many "universal" tokenizers (UniCodec, WavTokenizer) run at only 24 kHz. — [UniCodec](https://arxiv.org/html/2502.20067v1)

### Inferences
- A neural codec's stereo image depends on its architecture, and neural codecs have no equivalent of the mature M/S and intensity-stereo tools in AAC and Opus. This is plausibly one reason high-rate neural codecs still fall back to dual mono, which wastes bits.
- The saturating high-band error in TQCodec suggests that GAN-style decoders "synthesise" high bands plausibly rather than reproduce them. That is fine for MOS but a barrier to transparency.

### Gaps
- I found no quantitative, per-genre or per-artifact study (transients, castanets/glockenspiel-style killer items, reverb tails, dense polyphony) comparing neural and classical codecs.
- I found no published multichannel or immersive results (5.1 or Ambisonics) for any neural general-audio codec.

## 3. Continuous VAE latents for diffusion vs discrete tokens for LMs

### Takeaway
Generative music systems in 2025–26 mostly use continuous VAE latents at about 20–40 Hz for diffusion/DiT, or low-rate discrete RVQ/FSQ tokens for LMs. These systems are optimised for latent rate and how easily the latents can be generated, not for bitrate. Reconstruction from SAO-class VAEs is clearly below DAC on signal metrics. Newer 25–37 Hz VAEs (GenAE-KL, Qwen VAE) now match or beat 86 Hz DAC on mel distance and ViSQOL, at a 3–4× lower frame rate. Models that unify the two (CoDiCodec, GenAE) give continuous and discrete representations from one backbone.

### Cited Findings
- Stable Audio 2.0 autoencoder (2024): 44.1 kHz, 21.5 Hz, 64-ch continuous latent. On Song Describer it scored STFT 1.19, MEL 0.71, SI-SDR 7.14, vs DAC at 86 Hz with 0.96 / 0.52 / 10.83. The authors frame the autoencoder as the quality ceiling of the generator. Numbers are as reported in a search summary of the paper. — [arXiv 2404.10301](https://arxiv.org/pdf/2404.10301)
- Stable Audio Open's VAE performs "nearly similar" to Stable Audio 2.0's despite training only on CC data. — [arXiv 2407.14358](https://arxiv.org/pdf/2407.14358)
- STAR-VAE (Jun 2026) beats SAO at the same 21.5 Hz on all its metrics (e.g. 1.17 / 0.75 vs 1.25 / 0.86). A 43 Hz EAR-VAE scores better on signal metrics because it compresses less. Figures are from a search snippet and I did not verify the metric labels. — [arXiv 2606.23064](https://arxiv.org/pdf/2606.23064)
- On general audio, SAO 1.0 scored ViSQOL 4.04 / LSD 0.936 vs DAC 4.02 / 0.959 (search snippet). — [SwanTale/SwanVAE, arXiv 2608.02023](https://arxiv.org/pdf/2608.02023)
- Qwen-Audio-3.0's 25 Hz, 128-d stereo VAE beats DAC (86 Hz) on Song Describer: ViSQOL 4.30 vs 4.20, mel 0.45 vs 0.87. It also beats SAO on every listed metric (table in section 1). Adding semantic supervision changes music ViSQOL only slightly (4.29) and degrades spectral and stereo metrics a little. — [arXiv 2607.27011](https://arxiv.org/pdf/2607.27011)
- GenAE argues that "reconstruction-first" autoencoders have high token rates and slow encoding, and need separate models per latent type or channel format.
  - At 13.125 Hz, GenAE beats SAO on all metrics at 60% of SAO's rate and matches DAC at about 15% of DAC's rate.
  - Context fits 832 s at 13 Hz vs 127 s for DAC.
  - A post-training RVQ stage turns the continuous model into a token model. — [arXiv 2602.15749](https://arxiv.org/html/2602.15749v1)
- CoDiCodec gives about 11 Hz continuous latents and FSQ tokens (e.g. 2.38 kbps) from one consistency-trained model. FSQ-dropout lets one model serve both uses. — [arXiv 2509.09836](https://arxiv.org/abs/2509.09836v1)
- SpectroStream is the discrete tokenizer for Magenta RT, a live music LM (16 kbps; 4 kbps truncated). — [arXiv 2508.04651](https://arxiv.org/pdf/2508.04651)

### Inferences
- A 25 Hz × 128-d float latent is not a compressed bitstream. Unquantised it is about 80 kfloat/s, so these models' "compression" lives in frame-rate and dimension ratios, not kbps. To compare them with codecs, the latents must be quantised.
- Even so, a 25 Hz VAE that matches DAC on ViSQOL suggests these latents would be very efficient bit-for-bit if entropy-coded. I found no paper that does this rigorously for music.
- Generative-model latent design has largely decoupled from compression state of the art. The goals are generation ease, long context, encode speed and semantics, so low-rate reconstruction metrics are a weak proxy for codec quality.

### Gaps
- I found no entropy-coded bitrate for any continuous music VAE, and no listening test of VAE reconstructions against Opus or AAC.

## 4. Universal codecs vs domain-specific: the speech penalty

### Takeaway
At equal capacity and token rate, naive unification costs speech quality. WavTokenizer-unified loses about 0.5 PESQ, 0.44 UTMOS and 5 MUSHRA points on speech vs its speech-only sibling, and its music MUSHRA collapses from 75 to 56. Domain-aware designs largely close the gap. UniCodec's partitioned codebook plus domain MoE beats the domain-specific WavTokenizers in every domain. High-rate universal codecs (DAC at 86 Hz, FlowDec, 2026 universal VAEs) remain competitive on speech.

### Cited Findings
- UniCodec (ACL 2025), all at 24 kHz and 75 tokens/s, MUSHRA:

  | Model | Speech | Music | Sound |
  |---|---|---|---|
  | Ground truth | 93.5 | 96.2 | 95.3 |
  | WavTokenizer speech-only | 85.4 | – | – |
  | WavTokenizer music/audio-only | – | 75.2 | 80.2 |
  | WavTokenizer unified | 80.4 | 56.1 | 62.2 |
  | UniCodec (unified) | 90.7 | 77.8 | 82.4 |

  — [arXiv 2502.20067](https://arxiv.org/html/2502.20067v1)
- Same paper, LibriTTS speech, PESQ / UTMOS:

  | Model | PESQ | UTMOS |
  |---|---|---|
  | WavTokenizer speech | 2.37 | 4.05 |
  | WavTokenizer unified | 1.84 | 3.61 |
  | UniCodec | 3.03 | 3.99 |
  | DAC (600 TPS) | 3.52 | 3.69 |

  Speech-only codecs collapse on music: BigCodec has music mel distance 1.38 vs 0.36 for DAC at 600 TPS, and TAAE 1.41. — [arXiv 2502.20067](https://arxiv.org/html/2502.20067v1)
- FlowDec, a general 48 kHz codec, is rated better than DAC on speech, on par on music and slightly worse on sound. — [FlowDec](https://arxiv.org/pdf/2503.01485)
- Universal 2026 VAEs on speech (Seed-TTS EN), PESQ / ViSQOL:

  | Model | PESQ | ViSQOL |
  |---|---|---|
  | Qwen VAE | 3.65 | 4.39 |
  | DAC | 3.77 | 4.39 |
  | HoliTok | 3.92 | 4.39 |
  | Stable Audio Open (music-oriented) | 2.54 | 4.26 |

  — [arXiv 2607.27011](https://arxiv.org/pdf/2607.27011)
- In Gull's tests, Gull-L at 4.8 kbps reached speech MUSHRA 84.1 at 16 kHz (EVS 7.2 kbps: 85.7) and near-reference music at 24 kbps. One subband architecture covered both. — [arXiv 2404.04947](https://arxiv.org/html/2404.04947v2)

### Inferences
- The speech penalty mostly appears at low token rates and small codebook budgets, where capacity is scarce. At DAC-like rates (≥6–8 kbps) or with domain-partitioned codebooks, it is small or nil on objective metrics.
- Music suffers more than speech from naive unification (−19 MUSHRA vs −5 in WavTokenizer).

### Gaps
- I found no rigorous MUSHRA of a universal full-band codec against a state-of-the-art speech codec at matched low bitrate. That belongs to the speech researcher's scope.

## 5. Has any neural codec beaten AAC/Opus for transparent music at 32–64 kbps? The transparency frontier

### Takeaway
No. As of Oct 2026 I found no rigorous, peer-reviewed listening test (MUSHRA or ABX with hidden reference) showing a neural codec beating AAC-LC/HE-AAC or Opus at 32–64 kbps stereo music, or reaching transparency there. The strongest positive results are at much lower rates, against weak anchors:
- EnCodec at 6 kbps ≈ MP3 at 64 kbps (2022).
- Gull at 24 kbps ≈ Opus at 24 kbps at near-reference MUSHRA (2024).
- TQCodec at 64 kbps preferred no more often than Vorbis at 96 kbps (2026).

Neural codecs dominate below about 12–16 kbps. Opus/AAC remain unbeaten, and much cheaper, at the bitrates where music becomes transparent.

### Cited Findings
- At 24 kbps on 44.1 kHz music, Opus already scored 92.3 MUSHRA vs reference 92.8. Gull-L at 24 kbps scored 92.5 and EnCodec 83.7. Neural parity with Opus was reached only near the ceiling, where the test can no longer discriminate.
  - 30 listeners; listeners excluded if they rated the hidden reference below 90 on more than 15% of trials.
  - ViSQOL favoured Opus at ≥20 kbps. — [Gull, arXiv 2404.04947v2](https://arxiv.org/html/2404.04947v2)
- 2022, 48 kHz stereo music: EnCodec at 24 kbps scored 87.5 and at 12 kbps 88.0, vs reference 95.1. Opus 24 kbps scored 82.9 and MP3 64 kbps 82.7. EnCodec plateaued about 7 points below the reference even at 24 kbps, so it was not transparent. — [arXiv 2210.13438](https://arxiv.org/pdf/2210.13438)
- DAC's authors state that even at 8 kbps DAC falls short of the reference's MUSHRA. — [arXiv 2306.06546](https://arxiv.org/pdf/2306.06546)
- TQCodec (2026) is the only paper I found targeting 32–128 kbps.
  - Its subjective test was 64 kbps vs Ogg Vorbis 96 kbps: 39.1% vs 39.4% preference, 21.5% no difference, MOS 4.18.
  - It has no AAC/Opus anchor, no MUSHRA, and no listener count. — [arXiv 2603.01592](https://arxiv.org/html/2603.01592)
- The 2025 Fraunhofer/ODAQ MUSHRA tested DAC and EnCodec at 16–32 kb/s stereo but included no classical anchors. — [arXiv 2509.18823](https://arxiv.org/html/2509.18823)
- An informal Hydrogenaudio personal blind test (June 2025, 15 samples) compared xHE-AAC (exhale), HE-AACv2, Opus and Fabrice Bellard's neural TSAC at low bitrates. xHE-AAC and Opus "were the best in fidelity" (MOS > 3.0). This is not a rigorous study. — [Hydrogenaudio thread](https://hydrogenaudio.org/index.php/topic,127980.0.html) (page returned 403; snippet only)
- Classical reference points:
  - Opus beat HE-AAC and Vorbis at 64 kb/s and LC-AAC at 96 kb/s in Xiph/HA public tests (2011–2014 era). — [opus-codec.org comparison](https://opus-codec.org/comparison/)
  - A 2025 HA thread notes there have been no new Opus vs FDK-AAC listening tests recently. — [Hydrogenaudio 2025](https://hydrogenaudio.org/index.php/topic,127595.0.html)
- Industry overviews (late 2025–2026) say neural codecs reach Opus-level quality at about 3–4× fewer bits but cost much more compute, and that Opus remains the default for live use. This is a secondary, non-rigorous source. — [Fora Soft 2026](https://www.forasoft.com/learn/audio-for-video/articles-audio/neural-audio-codecs-lyra-encodec-soundstream)
- Metric caution: on MUSHRA data, FAD over codec embeddings correlates only moderately with listener scores (Pearson 0.38 for EnCodec embeddings, 0.67–0.70 for DAC/DACe). CLAP-M and OpenL3 embeddings do better at 0.84–0.85. Objective "near-transparent" claims based on FAD, ViSQOL or mel therefore need listening-test backing. — [arXiv 2509.18823](https://arxiv.org/html/2509.18823)
- PEAQ-based evaluations of classical codecs exist (Nov 2025, revised Mar 2026) but include no neural codecs. — [arXiv 2511.11527](https://arxiv.org/abs/2511.11527)

### Inferences
- Where neural codecs win:
  - Below about 12–16 kbps stereo, they clearly beat Opus and MP3. Opus at 6 kbps scores about 18–21 MUSHRA vs 83–93 for EnCodec and DAC-class models.
  - At about 24 kbps, the best neural codecs and Opus both approach the MUSHRA ceiling on typical material.
- Where the transparency frontier sits:
  - For classical codecs, it is traditionally about 96–128 kbps stereo for AAC and Opus on critical items.
  - For neural codecs it is undemonstrated. GAN and flow decoders show residual gaps (EnCodec about 7 points below reference at 24 kbps; DAC below reference at 8 kbps) and saturating high-band error (TQCodec).
  - Plausibly, perceptual quality near transparency is limited by generative "plausible but not faithful" synthesis rather than by bitrate.
- Practical status:
  - Classical codecs still win on compute, latency, standardisation and robustness at 32–64 kbps.
  - Neural codecs win on extreme compression and as generative tokenizers.
  - A rigorous MUSHRA of a 2025–26 neural codec vs Opus 1.5 and FDK/exhale AAC at 32/48/64 kbps stereo, with killer items, has not been published as far as I could find. This is the key open gap.

### Gaps
- I found no MUSHRA, BS.1116 or ABX study of any neural codec vs AAC or Opus at 32–64 kbps.
- I could not access TSAC's quantitative results (HA page 403).
- I could not confirm whether any 2026 MPEG or AES work has tested neural codecs against USAC at these rates. My searches of AES 2025 material found none.
