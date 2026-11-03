"""Ad / self-promotion detection for mimi-hub transcripts.

An LLM reads the episode's cues (index, time, speaker, text) and returns cue ranges that are ads or the show's
self-promotion; those become time ranges a player can skip. The LLM is either an API (OpenRouter: Claude Haiku 5.5,
~$0.004 per hour of audio, needing only OPENROUTER_API_KEY, so a hub anywhere can run it) or a subscription-logged-in
agent CLI on the server (claude / codex / agy) in print mode, from an empty temp dir with tools off, costing nothing
per episode.

    MIMI_HUB_SEGMENTER = "<backend>:<model>[+<flag>...][,<backend>:<model>...]"   tried in order until one works;
        default openrouter:anthropic/claude-haiku-5.5,claude:claude-opus-5-5,codex:gpt-5.6-terra+high (an API
        backend whose key isn't set is skipped, so without OPENROUTER_API_KEY it's Opus through the CLI), or "off"
        codex flags: minimal / low / medium / high / xhigh (reasoning effort; without one, ~/.codex/config.toml's),
        fast / priority / flex / default (service tier); e.g. codex:gpt-6.1-sol+low+fast
        claude flags: low / medium / high / xhigh / max (--effort), fast (fast mode); e.g. claude:claude-opus-5-5+fast
        openrouter:<model id>[+<effort>] (OpenRouter's API, key in OPENROUTER_API_KEY; effort low .. xhigh as
            its reasoning.effort); e.g. openrouter:anthropic/claude-haiku-5.5, openrouter:inception/mercury-3
        inception:<model id> (Inception Labs' API, key in INCEPTION_API_KEY); e.g. inception:mercury-3

Against judged ground truths (2026-11-02; scripts/ads_truth.py): 13 episodes the v2 prompt was written from (50
breaks) and 11 fresh shows (48), with the v2 prompt: Haiku 5.5 through OpenRouter all 98, nothing extra, edges a
little looser (skipped-seconds IoU 0.94-0.95), ~9 s an episode, $0.097 for the 24 h of audio (high effort: no better,
~20% more); Opus 5.5 97, IoU 0.99, ~5 s; gpt-5.6-terra at high effort 98, ~8 s; Sol at high effort, the previous default, 96 with 3 extra.
Several transcripts per request (--many) found 1-2 fewer of the short self-promos. scripts/ads_bench.py compares.

CLI: python -m hub.segments <episode.vtt> [backend:model]   (prints the segments JSON)
     python -m hub.segments --many 4 <episode.vtt>...      (a backlog: 4 transcripts per request, detect_many;
                                                           each <episode>.segments.json written beside it)
"""

import json
import os
import pwd
import re
import shutil
import subprocess
import sys
import tempfile
import time

SEGMENTER = os.environ.get("MIMI_HUB_SEGMENTER",
                           "openrouter:anthropic/claude-haiku-5.5,claude:claude-opus-5-5,codex:gpt-5.6-terra+high")
CHUNK_CHARS = 90_000  # agy takes the prompt as one argv string (Linux caps those at 128 KiB)
OVERLAP_CUES = 40
TIMEOUT_S = 600
GAP_CUES, GAP_S = 2, 10.0  # ads this close together are one break (the gap is "Uh-huh." / a jingle)
MIN_S = 6.0  # shorter plugs mid-conversation aren't worth a jarring skip

PROMPT = """You are marking the parts of a podcast episode that a listener would want to skip.
Below is its automatic transcript, one cue per line: [cue index] mm:ss speaker: text.
Speaker labels come from automatic diarization and can be wrong; ASR errors are common.

Mark every segment of these kinds:
- "ad": advertising. Inserted or pre-recorded ads, host-read sponsor messages ("this episode is brought to you
  by...", promo codes, "go to example.com/podcast"), and trailers/promos for other podcasts or shows that play in
  an ad break.
- "promo": the show's own self-promotion: pitches for its subscription, membership, Patreon, merch, newsletter,
  live shows, tours or events, "rate and review us", the hosts' books or other projects.

Do not mark the actual content, even where it discusses products or companies; news rundowns; the show intro;
short listener-feedback invitations ("email us at..."); or closing credits. A run of back-to-back ads and promos
in one break is a single "ad" segment. Segments start at the first cue of the ad or promo and end at its last
cue; don't include the hosts' lead-in ("we'll be right back") or return ("welcome back").

Reply with only JSON, no prose or code fences:
{"segments": [{"first": <cue index>, "last": <cue index>, "kind": "ad" | "promo", "label": "<sponsors or what is promoted>"}]}
or {"segments": []} if there are none.

Transcript:
"""

# v2: the same task, with the patterns the models missed most spelled out (scripts/ads_truth.py: sponsored inserts and
# teasers inside a break, post-roll ads, whole breaks as one segment) and what not to mark listed more fully.
# MIMI_HUB_SEGMENTS_PROMPT (default v2; v1 the original).
RULES_V2 = """You are marking the parts of a podcast episode that a listener would want to skip: ad breaks and the
show's self-promotion. The transcript is automatic, one cue per line: [cue index] mm:ss speaker: text. Speaker labels
come from automatic diarization and can be wrong; ASR errors, misspelled brand names especially, are common.

Mark every segment of these kinds:
- "ad": advertising. Pre-recorded or inserted ads, host-read sponsor messages ("this episode is brought to you
  by...", promo codes, "go to example.com/podcast"), trailers and promos for other podcasts or shows, and sponsored
  content placed in an ad break even when it sounds like content: a short news bulletin or feature "brought to you
  by" a sponsor, or a teaser for a sponsored segment or series ("...find out more later in the podcast"). Ads often
  also play after the closing credits; mark those too.
- "promo": the show's own self-promotion: pitches for its subscription, membership or ad-free tier, Patreon, merch,
  newsletter, Discord or community, live shows, tours or events, its video version, "rate and review us", the hosts'
  books or other projects.

An ad break is usually several ads back to back, often in other voices, between parts of the show. Mark a whole
break as one "ad" segment, from the first cue of its first ad or teaser to the last cue of its last ad, including
any sponsored bulletin or teaser at its start or in its middle. A run of the show's promos is one "promo" segment;
when ads follow it directly, one "ad" segment covering both.

Do not mark: the content itself, even where it discusses products, companies or the hosts' other shows; news
rundowns that are part of the show; the show's intro or theme; the hosts' lead-in ("we'll be right back") or return
("welcome back"); a network's audio logo ("... Audio Studios", "... Podcast Network"); short listener-feedback
invitations ("email us at...", "call our hotline"); closing credits; a call to action that is part of the
conversation (a guest on how to apply, where to find their work).
"""

REPLY_ONE = """Reply with only JSON, no prose or code fences:
{"segments": [{"first": <cue index>, "last": <cue index>, "kind": "ad" | "promo", "label": "<sponsors or what is promoted>"}]}
or {"segments": []} if there are none.

Transcript:
"""

REPLY_MANY = """Below are {n} episodes' transcripts, each starting with a line "### Episode <number>"; cue indices
restart at 0 in each. Mark each episode on its own.
Reply with only JSON, no prose or code fences:
{{"episodes": [{{"episode": <number>, "segments": [{{"first": <cue index>, "last": <cue index>, "kind": "ad" | "promo", "label": "<sponsors or what is promoted>"}}]}}, ...]}}
listing every episode, with "segments": [] for one with none.

Transcripts:
"""

PROMPTS = {"v1": PROMPT, "v2": RULES_V2 + "\n" + REPLY_ONE}
PROMPT_VERSION = os.environ.get("MIMI_HUB_SEGMENTS_PROMPT", "v2")

_CUE = re.compile(
    r"(\d+):(\d\d):(\d\d\.\d+) --> (\d+):(\d\d):(\d\d\.\d+)[^\n]*\n(?:<v ([^>]+)>)?([^\n]*)"
)


def read_cues(vtt_text):
    """[(start_s, end_s, speaker or None, text)] from a mimi-hub WebVTT."""
    cues = []
    for m in _CUE.finditer(vtt_text):
        a = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
        b = int(m[4]) * 3600 + int(m[5]) * 60 + float(m[6])
        text = m[8].replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        cues.append((a, b, m[7], text))
    return cues


def _line(i, cue):
    t = int(cue[0])
    spk = (cue[2] or "?").replace("Speaker ", "S")
    return f"[{i}] {t // 60:02d}:{t % 60:02d} {spk}: {cue[3]}"


def _windows(lines):
    """Index ranges of at most CHUNK_CHARS of transcript, overlapping by OVERLAP_CUES cues."""
    out, start = [], 0
    while start < len(lines):
        end, size = start, 0
        while end < len(lines) and (end == start or size + len(lines[end]) < CHUNK_CHARS):
            size += len(lines[end]) + 1
            end += 1
        out.append((start, end))
        if end >= len(lines):
            break
        start = max(start + 1, end - OVERLAP_CUES)
    return out


# the hub's systemd unit has a minimal PATH; the agent CLIs live in the user profile
_PATH = os.pathsep.join([os.environ.get("PATH", ""), os.path.expanduser("~/.local/bin"),
                         f"/etc/profiles/per-user/{pwd.getpwuid(os.getuid()).pw_name}/bin", "/run/current-system/sw/bin"])


EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")
TIERS = ("fast", "priority", "flex", "default")


def _run(backend, model, prompt):
    model, *flags = model.split("+")
    env = dict(os.environ, PATH=_PATH)
    exe = shutil.which(backend, path=_PATH) or backend
    with tempfile.TemporaryDirectory(prefix="mimi-seg-") as tmp:
        if backend == "claude":
            cmd = [exe, "-p", "--model", model, "--tools", "", "--no-session-persistence",
                   "--setting-sources", "", "--strict-mcp-config", "--output-format", "text"]
            for f in flags:
                if f == "fast":
                    cmd += ["--settings", '{"fastMode": true}']
                elif f in EFFORTS and f != "minimal":
                    cmd += ["--effort", f]
                else:
                    raise ValueError(f"unknown claude flag {f!r}")
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=tmp, env=env, timeout=TIMEOUT_S)
            out = r.stdout
        elif backend == "codex":
            last = os.path.join(tmp, "last.txt")
            cmd = [exe, "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-m", model, "-o", last]
            for f in flags:
                if f in EFFORTS:
                    cmd += ["-c", f"model_reasoning_effort={f}"]
                elif f in TIERS:
                    cmd += ["-c", f"service_tier={f}"]
                else:
                    raise ValueError(f"unknown codex flag {f!r}")
            cmd.append("-")
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=tmp, env=env, timeout=TIMEOUT_S)
            out = open(last).read() if os.path.exists(last) else ""
        elif backend in _OPENAI_COMPAT:
            return _chat(backend, model, flags, prompt)
        elif backend == "agy":
            cmd = [exe, "--model", model, "--output-format", "text", "-p", prompt]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp, env=env, timeout=TIMEOUT_S)
            out = r.stdout
        else:
            raise ValueError(f"unknown segmenter backend {backend!r}")
    if r.returncode != 0:
        raise RuntimeError(f"{backend} exited {r.returncode}: {(r.stderr or r.stdout)[-400:]}")
    return out


# OpenAI-style chat APIs: backend -> (endpoint, environment variable with the key)
_OPENAI_COMPAT = {"openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
                  "inception": ("https://api.inceptionlabs.ai/v1/chat/completions", "INCEPTION_API_KEY")}


def _chat(backend, model, flags, prompt):
    import urllib.request
    # uv's Python doesn't know NixOS's CA bundle (as hub/mimi_hub.py; this may run on its own)
    if not os.environ.get("SSL_CERT_FILE") and os.path.exists("/etc/ssl/certs/ca-certificates.crt"):
        os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"
    url, var = _OPENAI_COMPAT[backend]
    key = os.environ.get(var)
    if not key:
        raise RuntimeError(f"{var} not set")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if not model.startswith("anthropic/"):  # Claude 5.5 models reject non-default sampling
        body["temperature"] = 0
    for f in flags:
        if f not in EFFORTS or backend != "openrouter":
            raise ValueError(f"unknown {backend} flag {f!r}")
        body["reasoning"] = {"effort": f}
    body = json.dumps(body)
    req = urllib.request.Request(url, data=body.encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    import urllib.error
    for attempt in range(6):  # 429: the provider's rate limit (OpenRouter's keys share a pool); back off
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                out = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise RuntimeError(f"{backend} HTTP {e.code}: {e.read().decode()[:300]}") from None
            time.sleep(2 * 2 ** attempt)
    if "error" in out:
        raise RuntimeError(f"{backend}: {out['error']}")
    return out["choices"][0]["message"]["content"] or ""


def _parse(text, lo, hi):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {text[:200]!r}")
    found = []
    for s in json.loads(m[0]).get("segments", []):
        first, last = int(s["first"]), int(s["last"])
        kind = s.get("kind") if s.get("kind") in ("ad", "promo") else "ad"
        first, last = max(first, lo), min(last, hi - 1)
        if first <= last:
            found.append({"first": first, "last": last, "kind": kind, "label": str(s.get("label", ""))[:200]})
    return found


def _merge(segs, cues):
    """Union of overlapping or nearly touching cue ranges (chunk overlaps also report a break twice)."""
    out = []
    for s in sorted(segs, key=lambda s: (s["first"], s["last"])):
        if out and (s["first"] <= out[-1]["last"] + 1 or (
                s["first"] <= out[-1]["last"] + 1 + GAP_CUES
                and cues[s["first"]][0] - cues[out[-1]["last"]][1] <= GAP_S)):
            prev = out[-1]
            if s["last"] > prev["last"]:
                prev["last"] = s["last"]
            if s["kind"] == "ad":
                prev["kind"] = "ad"
            if s["label"] and s["label"] not in prev["label"]:
                prev["label"] = f"{prev['label']}; {s['label']}" if prev["label"] else s["label"]
        else:
            out.append(dict(s))
    return out


def _choices(segmenter):
    """The [segmenter] list's entries, without API backends whose key isn't set (unless that leaves none)."""
    every = [c.strip() for c in segmenter.split(",") if c.strip()]
    keyed = [c for c in every if c.partition(":")[0] not in _OPENAI_COMPAT
             or os.environ.get(_OPENAI_COMPAT[c.partition(":")[0]][1])]
    return keyed or every


def _ask(segmenter, lines, log, prompt=None):
    """Cue ranges from the first backend in the comma-separated [segmenter] list that answers."""
    errors = []
    head = PROMPTS[prompt or PROMPT_VERSION]
    for choice in _choices(segmenter):
        backend, _, model = choice.partition(":")
        try:
            found = []
            for lo, hi in _windows(lines):
                found += _parse(_run(backend, model, head + "\n".join(lines[lo:hi]) + "\n"), lo, hi)
            return choice, found
        except Exception as e:  # expired login, timeout, unparseable reply: try the next one
            log(f"segments: {choice} failed: {e}")
            errors.append(f"{choice}: {e}")
    raise RuntimeError("; ".join(errors) or "no segmenter configured")


def _finish(found, cues):
    segs = []
    for s in _merge(found, cues):
        start = cues[s["first"]][0]
        # resume where the next cue starts (skips the music bed / silence after the break too)
        end = cues[s["last"] + 1][0] if s["last"] + 1 < len(cues) else cues[s["last"]][1]
        if end - start >= MIN_S:
            segs.append({"start": round(start, 2), "end": round(end, 2), **s})
    return segs


def detect(cues, segmenter=SEGMENTER, log=print, prompt=None):
    """Skip ranges for [cues]: {"model", "segments": [{"start", "end", "kind", "label", "first", "last"}]}."""
    lines = [_line(i, c) for i, c in enumerate(cues)]
    t0 = time.time()
    used, found = _ask(segmenter, lines, log, prompt)
    segs = _finish(found, cues)
    log(f"segments: {len(segs)} via {used} in {time.time() - t0:.0f} s")
    return {"model": used, "segments": segs}


def detect_many(episodes, segmenter=SEGMENTER, log=print):
    """detect() for several episodes' cues in one request per backend tried (the v2 rules; whole transcripts, no
    windows: for backends with long contexts): a list of results in the same order."""
    blocks = [f"### Episode {k + 1}\n" + "\n".join(_line(i, c) for i, c in enumerate(cues))
              for k, cues in enumerate(episodes)]
    prompt = RULES_V2 + "\n" + REPLY_MANY.format(n=len(episodes)) + "\n\n".join(blocks) + "\n"
    errors = []
    for choice in _choices(segmenter):
        backend, _, model = choice.partition(":")
        try:
            text = _run(backend, model, prompt)
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                raise ValueError(f"no JSON in reply: {text[:200]!r}")
            by_ep = {int(e["episode"]): e.get("segments", []) for e in json.loads(m[0]).get("episodes", [])}
            out = []
            for k, cues in enumerate(episodes):
                found = _parse(json.dumps({"segments": by_ep.get(k + 1, [])}), 0, len(cues))
                out.append({"model": choice, "segments": _finish(found, cues)})
            return out
        except Exception as e:  # noqa: BLE001
            log(f"segments: {choice} failed: {e}")
            errors.append(f"{choice}: {e}")
    raise RuntimeError("; ".join(errors) or "no segmenter configured")


def _many(n, paths):
    import pathlib
    for k in range(0, len(paths), n):
        group = [pathlib.Path(p) for p in paths[k:k + n]]
        t0 = time.time()
        res = detect_many([read_cues(p.read_text()) for p in group], log=lambda m: print(m, file=sys.stderr))
        for p, r in zip(group, res, strict=True):
            out = p.with_name(p.name.split(".")[0] + ".segments.json")
            out.write_text(json.dumps(r, indent=1))
            print(f"{p.name}: {len(r['segments'])} segments via {r['model']} -> {out.name}", file=sys.stderr)
        print(f"{len(group)} episodes in {time.time() - t0:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    if sys.argv[1] == "--many":
        _many(int(sys.argv[2]), sys.argv[3:])
        sys.exit()
    cues = read_cues(open(sys.argv[1]).read())
    res = detect(cues, *(sys.argv[2:3] or [SEGMENTER]), log=lambda m: print(m, file=sys.stderr))
    for s in res["segments"]:
        print(f"{s['kind']:5s} {int(s['start'] // 60):02d}:{s['start'] % 60:04.1f}-"
              f"{int(s['end'] // 60):02d}:{s['end'] % 60:04.1f}  cues {s['first']}-{s['last']}  {s['label']}",
              file=sys.stderr)
    print(json.dumps(res, indent=1))
