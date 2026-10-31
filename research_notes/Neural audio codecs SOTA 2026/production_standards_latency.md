# Neural / ML-enhanced audio codecs in production and standardization (state as of 2026-10-30), with latency, streaming and compute constraints

Research method note: 15 web searches and fetches on 2026-10-30. Primary sources were fetched where possible: the Opus 1.6 release page, the IETF DRED draft -07, and the Samsung Research ULBC blog. Some items come from search-result snippets only and are labeled that way. Items marked "[training knowledge, not re-verified 2026-10-30]" are well-known facts I could not re-fetch within the tool budget. The report writer should treat them as lower confidence.

## 1. Opus ML features (1.5 / 1.6) and IETF mlcodec work

### Takeaway
Opus is the most widely deployed production codec with neural components. Its ML features are all decoder-side or redundancy add-ons that keep RFC 6716 compatibility. Opus 1.5 (2024) added Deep PLC, LACE/NoLACE and experimental DRED. Opus 1.6 (released 2025-12-15) added a neural wideband-to-fullband BWE, a smaller and incompatible DRED model, and Opus HD (QEXT). The IETF mlcodec WG is standardizing DRED, speech-coding enhancement and the quality extension as drafts. None is an RFC as of Oct 2026.

### Cited Findings
- Opus 1.6 was released 2025-12-15 and stays fully compatible with RFC 6716. Major changes: (a) an experimental ML wideband-to-fullband BWE that generates 8–20 kHz from 0–8 kHz SILK speech; (b) Opus HD, 96 kHz support at up to 2 Mb/s with up to 20 bits per coefficient as a scalable extension layer (`--enable-qext`); (c) a much improved DRED; (d) a 24-bit API. — [Opus 1.6 release page](https://opus-codec.org/demo/opus-1.6/)
- BWE requirements: `--enable-osce` at build time, explicit run-time enable, and decoder complexity ≥4. It applies only to SILK wideband speech decoded at 48 kHz and is off by default. — [Opus 1.6 release page](https://opus-codec.org/demo/opus-1.6/)
- Listening claims (Xiph's own tests): on coded speech, NoLACE plus BWE at 9 kb/s matches Opus 1.4 fullband at 18 kb/s. The page says it "closes the gap with EVS" and beats EnCodec at ≥9 kb/s. On uncoded speech, BWE closes about half the WB→FB quality gap. — [Opus 1.6 release page](https://opus-codec.org/demo/opus-1.6/)
- DRED in 1.6:
  - Training uses a 4th-power error term, which improves intelligibility but not perceived quality.
  - The training mix includes noisy/reverberant speech.
  - The encoder and decoder models total about 600 kB, down from about 1800 kB in 1.5.
  - DRED is signaled in the bitstream, so the 1.6 model cannot be used by 1.5 decoders. Mixing versions causes no artifacts; the incompatible decoder just ignores the DRED data.
  - Xiph "hopes" the model won't change in the final standard.
  - Source: [Opus 1.6 release page](https://opus-codec.org/demo/opus-1.6/)
- IETF drafts linked from the 1.6 release: draft-ietf-mlcodec-opus-speech-coding-enhancement, draft-ietf-mlcodec-opus-dred, draft-ietf-mlcodec-opus-scalable-quality-extension, and draft-lechler-mlcodec-test-battery-02 (intelligibility test data). — [Opus 1.6 release page](https://opus-codec.org/demo/opus-1.6/)
- draft-ietf-mlcodec-opus-dred-07 is dated 2026-08-04 (authors Valin and Büthe; Standards Track; updates RFC 6716):
  - Redundancy costs "about 1/50 of the regular Opus bitrate", compared with about 2/3 for classic LBRR FEC.
  - The encoder runs every 20 ms on pairs of 20-dim feature vectors. Each latent covers 40 ms, and only every other latent is sent.
  - One packet can carry "up to one second or more" of redundancy. The offset field can signal about 20.5 s, but the draft says there "appears to be little use" beyond a few seconds.
  - Quantization: 16 quantizer levels (4-bit Q0 plus a 3-bit slope), entropy-coded with Laplace.
  - The encoder is an RDO-VAE and is non-normative. The decoder is normative: DenseNet-like, with 5 GRU+conv stages and 64-unit GRUs, outputting 4×20 features per latent.
  - The vocoder is non-normative. It SHOULD NOT use features centered beyond t+5 ms, which bounds the extra look-ahead.
  - DRED is carried in Opus padding extension ID 32 (ID 126 for experimental use until IANA assigns one).
  - Complexity is "independent of the signal characteristics". No MFLOPS figure is given.
  - Source: [draft-ietf-mlcodec-opus-dred-07](https://www.ietf.org/ietf-ftp/internet-drafts/draft-ietf-mlcodec-opus-dred-07.html)
- Draft revision 06 is dated June 2026 and 07 August 2026. I found no WGLC, IESG or RFC status. — [datatracker DRED](https://datatracker.ietf.org/doc/draft-ietf-mlcodec-opus-dred/); [IETF 123 mlcodec minutes, July 2025](https://datatracker.ietf.org/meeting/123/materials/minutes-123-mlcodec-202507241500-00) (chairs Greg Maxwell and Mo Zanaty; notes that the DRED document is non-normative on behavior)
- A May 2026 bloggeek.me post advised tracking DRED until it "lands in libwebrtc and the spec freezes". This implies DRED was not in upstream libwebrtc's default build at that time (secondary source). — [bloggeek.me DRED glossary](https://bloggeek.me/webrtcglossary/dred/)
- Downstream WebRTC M156 builds bundling libopus 1.6 enable Deep PLC and OSCE (LACE/NoLACE) but leave DRED off. This is a third-party build, not upstream Google. — [drag0n-flyer WebRTC M156 + Opus 1.6 release](https://github.com/drag0n-flyer/myApp/releases/tag/webrtc-m156.1)
- Decoder complexity mapping in libopus: 5 = Deep PLC, 6 = LACE, 7 = NoLACE, ≥4 plus a flag = BWE (as described by a pure-Rust port of libopus 1.6.1). — [RealEffect/opus1](https://github.com/RealEffect/opus1)
- Complexity figures:
  - Original LACE: 300 K parameters, about 100 MFLOPS, causal.
  - NoLACE paper: 620 MFLOPS with 1.8 M parameters (a larger LACE variant there is 280 MFLOPS / 900 K).
  - The Opus build configuration describes NoLACE as "~400 MFLOPS".
  - FARGAN vocoder (used for DRED/PLC synthesis): about 600 MFLOPS.
  - Sources: [NoLACE arXiv 2309.14521](https://arxiv.org/pdf/2309.14521); [ptxdist opus.in](https://git.pengutronix.de/cgit/ptxdist/tree/rules/opus.in); [LACE WASPAA slides](https://jmvalin.ca/slides/LACE_waspaa.pdf); [FARGAN arXiv 2307.06610](https://arxiv.org/pdf/2307.06610)

### Inferences
- Xiph's design pattern for deployable neural coding is: keep the classic bitstream, put ML only in the decoder (enhancement, PLC, BWE), and put neural redundancy in ignorable extensions. This avoids a flag day and adds no algorithmic delay beyond Opus's own (≈26.5 ms default for 20 ms frames in CELP/hybrid mode [training knowledge, not re-verified 2026-10-30]).
- The compute budget Xiph targets is roughly 100–600 MFLOPS per decoder feature. That is one or two orders of magnitude below what generative research codecs and LLM-style token codecs need.
- The 1.5→1.6 DRED bitstream break shows that neural payloads freeze slowly. Standardizing the normative decoder weights is the real bottleneck.

### Gaps
- The exact DRED bitrate in kb/s for typical settings and the measured CPU % on phones were not in the sources fetched.
- I could not find a current mlcodec WG milestone date (WGLC/IESG) for DRED or the enhancement draft.
- I could not confirm whether Google upstream libwebrtc / Chrome enables LACE/NoLACE/DRED by default as of Oct 2026.
- I found no Opus 1.7 or "Opus next-gen" announcement in this session.

## 2. Company deployments (Google, Microsoft, Meta, Cisco, Zoom, Apple, WhatsApp, Discord)

### Takeaway
Shipping proprietary neural or ML-hybrid speech codecs are concentrated in low-bandwidth and high-loss fallback modes:
- Meta MLow: 6 kb/s-class, all Meta apps.
- Cisco Webex AI Codec: about 6 kb/s with 1 kb/s redundant copies, triggered at >30% loss.
- Microsoft Satin: decoder-side DNN bandwidth extension, 6 kb/s and up.
- Google Lyra in Meet/Duo: low-bandwidth mode, 3 kb/s and up.

I found no evidence of Zoom, Apple or Discord shipping a neural speech codec. Notably, MLow is not a neural codec: it is a classic CELP design tuned for low compute.

### Cited Findings
- **Meta MLow**:
  - Announced 2024-06-13 and designed for the regime where bandwidth estimation pushes Opus down to 6 kb/s.
  - Meta reports POLQA MOS of about 3.9 for MLow vs 1.89 for Opus at 6 kb/s wideband ("two-times-better"). The two are equal at about 25 kb/s.
  - Built to help low-end devices while keeping E2EE.
  - Fully launched on Instagram and Messenger calls; rolling out on WhatsApp as of June 2024.
  - Source: [Engineering at Meta, MLow](https://engineering.fb.com/2024/06/13/web/mlow-metas-low-bitrate-audio-codec/); [InfoQ, Aug 2024](https://www.infoq.com/news/2024/08/meta-mlow-audio-codec)
- A third-party blog from about September 2026 says MLow now carries WhatsApp, Messenger and Instagram calls, and describes a clean-room pure-Go reimplementation. This is not an official Meta source. — [WhatsMCP blog](https://whatsmcp.com/blog/mlow-codec-pure-go); see also [wacore crate audio.rs](https://docs.rs/crate/wacore/latest/source/src/voip/audio.rs)
- MLow's architecture is CELP-based (classic signal processing, not a neural network), with lower compute than Opus. This is the substance of Meta's post, but the details were not re-fetched [training knowledge, not re-verified 2026-10-30]. — [Engineering at Meta, MLow](https://engineering.fb.com/2024/06/13/web/mlow-metas-low-bitrate-audio-codec/)
- **Cisco Webex AI Codec**:
  - Generally available in the Webex App (desktop and mobile) for Calling and Meetings.
  - Turns on automatically when packet loss exceeds 30%.
  - Webex claims intelligible audio at about 80–90% packet loss, compared against Opus at about 80% loss. These are vendor claims.
  - A secondary source reports speech coded at about 6 kb/s plus redundant copies of previous frames at about 1 kb/s.
  - Trained with injected noise, reverb, bandwidth limits and packet loss.
  - Source: [Webex blog, availability](https://blog.webex.com/collaboration/announcing-the-availability-of-industry-first-webex-ai-codec-transforming-your-audio-experiences/); [heise](https://heise.de/-9848886); [TechCrunch Oct 2023](https://techcrunch.com/2023/10/25/cisco-announces-several-new-ai-tools-to-enhance-webex-experience); [Webex AI Codec whitepaper](https://www.webex.com/content/dam/www/us/en/whitepaper/ai-codec-whitepaper-cm-6165.pdf)
- **Microsoft Satin**:
  - Uses DNNs at the decoder to estimate high-band parameters from low-band data, a neural BWE approach.
  - In crowdsourced tests at 6 kb/s, Satin scored MOS +1.7 over SILK.
  - Designed so a lost packet does not degrade the packets that follow.
  - Microsoft (2021) said it was used on all Skype and Teams two-party calls, with Teams Meetings to follow. Teams "music mode" depends on Satin.
  - Source: [Microsoft Teams blog, Satin](https://techcommunity.microsoft.com/blog/microsoftteamsblog/satin-microsoft%E2%80%99s-latest-ai-powered-audio-codec-for-real-time-communications/2141382); [UC Today](https://www.uctoday.com/collaboration/microsoft-extends-ai-powered-satin-codec-rollout/); [Teams music mode discussion](https://techcommunity.microsoft.com/discussions/microsoftteams/how-the-satin-codec-makes-teams-music-mode-work/2686195)
- **Google Lyra**:
  - Meet uses Lyra for audio when bandwidth is limited.
  - The minimum bitrate is about 3 kb/s.
  - A 2025-era industry assessment calls Lyra a niche tool, not a near-term Opus replacement for interactive streaming, with compute and device support limiting adoption.
  - Source: [Wikipedia, Lyra](https://en.wikipedia.org/wiki/Lyra_(codec)); [Red5 blog](https://www.red5.net/blog/what-does-the-new-google-lyra-audio-codec-mean-for-real-time-video-streaming/); [Techzine](https://www.techzine.eu/news/devops/58020/googles-lyra-delivers-high-quality-calls-with-low-bandwidth/)
- Lyra v2 (Sept 2022) is SoundStream-based, with 3.2, 6 and 9.2 kb/s modes, 20 ms frames and about 20 ms algorithmic latency, and runs much faster than real time on a Pixel 6 [training knowledge, not re-verified 2026-10-30]. The SoundStream paper is [arXiv 2107.03312](https://arxiv.org/pdf/2107.03312).
- **Zoom**: documentation lists Opus and SILK audio plus proprietary adaptive logic that cuts video first to keep audio intelligible. It mentions no neural codec. — [Zoom library: codecs and bandwidth](https://library.zoom.com/admin-corner/network-management/quality-of-service-and-network-best-practices-explainer/how-zoom-audio-and-video-codecs-affect-bandwidth-usage)
- **Apple**: I found no primary source on a neural audio codec in FaceTime. Only a weak third-party design article turned up. — [techinterview.org](https://www.techinterview.org/post/3233475043/design-mobile-video-conferencing-zoom-facetime/)

### Inferences
- In production, "neural codec" mostly means one of four things:
  - a neural decoder-side enhancer or BWE (Satin, LACE/NoLACE, Opus BWE);
  - neural PLC or redundancy (Opus Deep PLC/DRED, the Webex redundancy scheme);
  - a SoundStream-style end-to-end codec used only as a low-bandwidth fallback (Lyra);
  - an efficient classic codec marketed alongside AI (MLow).
- All of them sit at 20 ms-class frames and bitrates of 1–9 kb/s+.
- In every case, deployed systems keep Opus or another classic codec as the main path and use neural modes for degraded networks. No major VoIP platform has publicly replaced its main wideband codec with an end-to-end neural codec.

### Gaps
- I found no current (2025–2026) official confirmation of Satin's scope in Teams Meetings, Lyra's current status in Meet, or MLow's rollout percentage on WhatsApp.
- I found no information on Discord, Signal, Telegram or FaceTime neural codec use.
- I found no vendor-published phone CPU figures for MLow, Satin or the Webex AI Codec.

## 3. Standards bodies: 3GPP (IVAS, ULBC, 6G), ITU-T, MPEG, AOMedia, ETSI

### Takeaway
3GPP SA4 is running the first 3GPP voice codec effort expected to be AI-based: the Rel-20 Ultra-Low Bitrate speech Codec (ULBC), targeting about 1.0 and 1.4 kb/s for voice over GEO satellite via NB-IoT NTN. First standards are targeted for early 2027. IVAS (Rel-18, classic, not neural) is "ready for deployment" with demos at MWC 2026, but I could confirm no operator or handset launch. A Rel-20 6G media study (FS_6G_MED) exists. MPEG has no neural speech/audio coding call; its nearest item is Audio Coding for Machines.

### Cited Findings
- The 3GPP SA4 Rel-20 work-item list includes:
  - FS_ULBC, "Study on Ultra Low Bitrate Speech Codec".
  - FS_6G_MED, "Study on Media Aspects for 6G System" (early 6G study).
  - AIML_IMS-MED, media aspects for AI/ML in IMS services.
  - Source: [3GPP SA4 work items](https://www.3gpp.org/dynareport?code=TSG-WG--S4--wis.htm); [3GPP Release 20](https://www.3gpp.org/specifications-technologies/releases/release-20)
- The FS_ULBC study objective is to recommend possible normative work on an ultra-low bitrate codec. The SID's note keeps scope from being limited to GEO satellite (SA4 audio ad-hoc contribution, May 2025). A follow-on item, ULBC-MED, covers IMS voice over NB-IoT NTN via GEO and is linked to TS 26.114. The study TR is reported as TR 26.940 "Study on Ultra Low Bit Rate Speech Codecs" (from search snippets; TR not opened). — [S4aV250028 (May 2025)](https://www.3gpp.org/ftp/TSG_SA/WG4_CODEC/3GPP_SA4_AHOC_MTGs/SA4_Audio/Inbox/Drafts/S4aA250028r01.doc); [TS 26.114 summary](https://whatthespec.net/3gpp/spec/26.114)
- Samsung Research blog (undated; adjacent stories dated Aug–Oct 2026):
  - 3GPP has begun standardizing ULBC at two target bitrates, 1.0 kb/s and 1.4 kb/s. One is quality-preferred; the other is more robust with extra link margin.
  - Candidates are expected to be AI-based, "a first for 3GPP voice codec standardization".
  - First standards are targeted for early 2027.
  - Algorithmic delay, audio bandwidth, training data, and neural-model complexity and memory limits are still under discussion, with no numbers published.
  - Low complexity matters for wearables.
  - It cites wideband neural codecs reaching AMR-WB-comparable clean-speech quality at under a tenth of the bitrate.
  - Source: [Samsung Research, Voice over NB-IoT NTN in Rel-20](https://research.samsung.com/blog/Voice-Over-NB-IoT-NTN-In-3GPP-Release-20)
- The June 2026 3GPP Highlights newsletter ties the 6G media-related work to TR 22.847 (6G_MED). — [3GPP Highlights Issue 12, June 2026](https://www.3gpp.org/ftp/Information/Highlights/3GPP_Highlights_Issue_12_WEB.pdf)
- IVAS:
  - A 3GPP article (about Aug 2026) says SA4 companies jointly demonstrated immersive live calls over IMS on commercially available devices at MWC Barcelona 2026, and calls the codec "ready for deployment now".
  - Nokia's FAQ says widespread commercial deployment across chipsets, devices and networks is "still on the horizon".
  - Nokia and Fraunhofer IIS made the first interoperable standardized IVAS call over 5G IMS.
  - Source: [3GPP IVAS article](https://www.3gpp.org/technologies/ivas-06-2026); [Nokia IVAS FAQ](https://www.nokia.com/multimedia/audio/immersive-voice/faq/); [Fraunhofer IIS blog](https://www.audioblog.iis.fraunhofer.com/ivas-call); [HEAD acoustics IVAS testing](https://audioxpress.com/news/head-acoustics-standard-compliant-3gpp-ivas-testing)
- MPEG:
  - The Audio Coding for Machines (ACoM) call for technologies was lossless-focused. Registration was due 2025-11-01, bitstreams 2025-12-01 and documentation 2026-01-09, with evaluation at MPEG 153 (Jan 2026).
  - MPEG-AI strategy (MPEG 150) builds on Neural Network Coding (ISO/IEC 15938-17), which compresses NNs, not audio.
  - The MPEG 155 report (July 2026) mentions no neural audio call.
  - Source: [MPEG 150 column (SIGMM)](https://records.sigmm.org/2025/05/07/mpeg-column-150th-mpeg-meeting-virtual-online/); [MPEG calls page](https://www.mpeg.org/emerging-technologies/); [Multimedia Communication, July 2026](https://multimediacommunication.blogspot.com/2026/07/?m=0)

### Inferences
- The ULBC timeline (Rel-20 study in 2025, normative work with first specs targeted early 2027) makes it the first likely standardized end-to-end neural speech codec in cellular. Its operating point (about 1–1.4 kb/s, GEO satellite) is exactly where classic CELP fails and the decoder compute budget is small (wearables/NTN modems).
- Delay budgets for ULBC are still open. GEO round-trip (~500+ ms) makes tens of ms of extra codec delay relatively cheap, so ULBC may tolerate longer frames (e.g. 40–80 ms) than terrestrial VoLTE/VoNR (20 ms EVS frames, ~32 ms algorithmic delay [training knowledge, not re-verified 2026-10-30]). This is inference, not confirmed.
- IVAS is the current non-neural 3GPP flagship (an EVS extension for immersive audio). Neural work in 3GPP is entering through ULBC and AI/ML-media studies, not through an EVS replacement.

### Gaps
- I found no AOMedia audio codec activity, ITU-T SG12/SG16 neural codec work, or ETSI neural codec work in this session. I did not confirm whether any exists.
- I could not access the TR 26.940 contents (candidate list, design constraints, test results) or SA4 meeting reports.
- I found no definite 6G voice-codec study item beyond the general FS_6G_MED.
- I could not confirm any IVAS commercial network launch date.

## 4. Satellite / very-low-bandwidth / emergency / military speech; codec2 → LPCNet → FARGAN lineage

### Takeaway
The main standardized very-low-bitrate neural effort is 3GPP ULBC for GEO NB-IoT NTN (about 1.0 and 1.4 kb/s). The Xiph/Valin lineage (LPCNet → FARGAN → DRED's RDO-VAE latents) delivers about 1 kb/s-scale neural speech today, but only as a redundancy layer inside Opus. Webex similarly uses about 1 kb/s redundant frames.

### Cited Findings
- 3GPP ULBC targets 1.0 and 1.4 kb/s for voice over GEO satellite via NB-IoT NTN, with AI-based candidates expected and first standards targeted for early 2027. — [Samsung Research](https://research.samsung.com/blog/Voice-Over-NB-IoT-NTN-In-3GPP-Release-20); [3GPP SA4 work items](https://www.3gpp.org/dynareport?code=TSG-WG--S4--wis.htm)
- FARGAN is a roughly 600 MFLOPS vocoder that beats earlier low-complexity vocoders in both quality and complexity. It succeeds LPCNet in Valin's pipeline. — [FARGAN arXiv 2307.06610](https://arxiv.org/pdf/2307.06610)
- DRED encodes speech as RDO-VAE latents, one per 40 ms, at about 1/50 of the Opus bitrate. Up to about 1 s or more of history fits in each packet. — [draft-ietf-mlcodec-opus-dred-07](https://www.ietf.org/ietf-ftp/internet-drafts/draft-ietf-mlcodec-opus-dred-07.html)
- The Webex AI Codec reportedly sends about 1 kb/s redundant copies of previous frames on top of about 6 kb/s primary speech. — [heise](https://heise.de/-9848886)
- 2026 academic papers target ultra-low-bitrate neural speech, which shows activity but not deployment. — [arXiv 2606.11631 benchmarking neural speech compression](https://arxiv.org/pdf/2606.11631); [arXiv 2606.05876](https://arxiv.org/pdf/2606.05876); [arXiv 2605.25669](https://arxiv.org/pdf/2605.25669)

### Inferences
- The DRED latent stream at 1/50 of Opus bitrate (≈0.5–1 kb/s for typical 24–48 kb/s Opus) is effectively a deployed sub-kb/s neural speech codec running in a redundancy role. It is the closest production analogue to ULBC's operating point.

### Gaps
- I found nothing in this session on Iridium, Inmarsat, Starlink direct-to-cell, emergency (FirstNet/TETRA) or military (MELPe/NATO STANAG successors) adoption of neural codecs.
- I found no Codec 2 / FreeDV neural mode status (e.g. FreeDV RADE) in this session.

## 5. Real-time constraints: latency, causality, compute, packet loss; research codecs vs deployment

### Takeaway
Deployed real-time speech codecs share four constraints:
- frames of about 20 ms, with total algorithmic delay in the 20–40 ms class;
- strictly causal processing with at most about 5 ms of look-ahead in the neural parts (DRED's vocoder rule);
- decoder add-ons in the 100–600 MFLOPS range on phones;
- loss handling via neural PLC or long redundancy.

Research codecs built for LLM tokenization use far longer frames (Mimi: 80 ms) and far more compute. They suit generation and asynchronous use better than duplex telephony.

### Cited Findings
- DRED: 20 ms encoder cadence, 40 ms per latent, vocoder look-ahead capped at about t+5 ms, complexity independent of signal, and redundancy of 1 s or more per packet to ride out burst loss. — [draft-ietf-mlcodec-opus-dred-07](https://www.ietf.org/ietf-ftp/internet-drafts/draft-ietf-mlcodec-opus-dred-07.html)
- Compute budgets: LACE about 100 MFLOPS (causal, 300 K parameters), NoLACE about 400–620 MFLOPS, FARGAN about 600 MFLOPS. — [LACE slides](https://jmvalin.ca/slides/LACE_waspaa.pdf); [NoLACE arXiv](https://arxiv.org/pdf/2309.14521); [ptxdist opus.in](https://git.pengutronix.de/cgit/ptxdist/tree/rules/opus.in); [FARGAN arXiv](https://arxiv.org/pdf/2307.06610)
- Lyra's compute demand and device support are cited as limits on its use for video calls at scale. — [Red5 blog](https://www.red5.net/blog/what-does-the-new-google-lyra-audio-codec-mean-for-real-time-video-streaming/)
- Meta built MLow to run on low-end devices. — [Engineering at Meta](https://engineering.fb.com/2024/06/13/web/mlow-metas-low-bitrate-audio-codec/)
- Webex AI Codec targets more than 30% packet loss, with claimed tolerance of 80–90%. — [Webex blog](https://blog.webex.com/collaboration/announcing-the-availability-of-industry-first-webex-ai-codec-transforming-your-audio-experiences/); [TechCrunch](https://techcrunch.com/2023/10/25/cisco-announces-several-new-ai-tools-to-enhance-webex-experience)
- Mimi (Kyutai, Sept 2024) runs at a 12.5 Hz frame rate (80 ms frames) at about 1.1 kb/s and is streaming/causal, with about 80 ms codec latency, built as Moshi's tokenizer [training knowledge, not re-verified 2026-10-30]. — [Moshi paper arXiv 2410.00037](https://arxiv.org/abs/2410.00037)
- SoundStream (2021) is the basis of Lyra v2. — [arXiv 2107.03312](https://arxiv.org/pdf/2107.03312)

### Inferences
- Budget comparison: a telephony mouth-to-ear budget of about 150 ms (ITU-T G.114 guidance [training knowledge, not re-verified 2026-10-30]) normally reserves about 20–40 ms for the codec. Mimi's 80 ms frame alone, before look-ahead, jitter buffer and network, uses 2–4× that allocation. Its 1.1 kb/s operating point is in the same band as 3GPP ULBC, where satellite RTT dominates anyway. That makes satellite or asynchronous messaging a more natural fit for 80 ms-frame tokenizers than terrestrial duplex calls.
- Compute: production decoder add-ons are capped at about 1 GFLOPS (and LACE at 100 MFLOPS) so they run on low-end Android phones. Transformer-based codecs (Mimi has transformer bottleneck layers) and diffusion/flow decoders typically exceed that by an order of magnitude or more. This is inference; the other researchers' figures should be checked.
- Packet loss: deployed systems treat loss robustness as a first-class design axis (DRED's 1 s redundancy, the Webex 30% trigger, Satin's no-error-propagation design). Research codecs with RVQ and long-context transformers seldom report bursty-loss behavior.

### Gaps
- I did not obtain measured phone CPU % or real-time-factor figures for any shipped neural codec in this session.
- I did not verify Mimi's or other research codecs' exact latency and FLOPs (other researchers cover these).

## 6. Neural vs classic codecs at matched bitrate (listening tests)

### Takeaway
Published head-to-heads are almost all vendor or proponent tests, and they show large neural/hybrid wins only at low bitrates (≤9 kb/s). The gap closes by about 18–25 kb/s. I found no independent, standards-grade test pitting neural codecs against EVS, xHE-AAC or LC3 in this session.

### Cited Findings
- MLow vs Opus at 6 kb/s WB: POLQA about 3.9 vs 1.89. The two are equal at about 25 kb/s (Meta's own data). — [Engineering at Meta](https://engineering.fb.com/2024/06/13/web/mlow-metas-low-bitrate-audio-codec/)
- Satin vs SILK at 6 kb/s: MOS +1.7 in crowdsourced tests (Microsoft's own data). — [Microsoft Teams blog](https://techcommunity.microsoft.com/blog/microsoftteamsblog/satin-microsoft%E2%80%99s-latest-ai-powered-audio-codec-for-real-time-communications/2141382)
- Opus 1.6 NoLACE plus BWE at 9 kb/s matches Opus 1.4 FB at 18 kb/s. Xiph says it "closes the gap with EVS" and beats EnCodec at ≥9 kb/s (Xiph's own tests). — [Opus 1.6 release page](https://opus-codec.org/demo/opus-1.6/)
- Wideband neural codecs reportedly reach AMR-WB-comparable clean-speech quality at under one-tenth of the bitrate. Samsung cites this as motivation for ULBC. — [Samsung Research](https://research.samsung.com/blog/Voice-Over-NB-IoT-NTN-In-3GPP-Release-20)
- Webex AI Codec vs Opus at about 80% loss is a demo only, with no MOS published in the sources found. — [Webex blog](https://blog.webex.com/collaboration/announcing-the-availability-of-industry-first-webex-ai-codec-transforming-your-audio-experiences/)
- A 2025 arXiv study evaluates audio compression codecs (classic and neural); not fetched. — [arXiv 2511.11527](https://arxiv.org/pdf/2511.11527)

### Inferences
- The consistent picture: at 6 kb/s, neural or highly tuned codecs beat Opus/SILK by about 1.5–2 MOS points. At about 9 kb/s, enhanced Opus gets near EVS. Above about 20–25 kb/s, classic codecs are at parity. That explains why neural modes ship as low-bandwidth fallbacks.

### Gaps
- I found no independent (3GPP/ITU P.800-style) listening test comparing neural codecs with EVS, IVAS, xHE-AAC or LC3 at matched bitrate in this session.
- I found no published MUSHRA results for DRED-recovered speech under burst loss compared with classic FEC.
