"""Ad / self-promotion detection for mimi-hub transcripts.

An LLM reads the episode's cues (index, time, speaker, text) and returns cue ranges that are ads or the show's
self-promotion; those become time ranges a player can skip. The LLM runs through a subscription-logged-in agent
CLI on the server (claude / codex / agy) in print mode, from an empty temp dir with tools off, so it costs nothing
per episode and needs no API key in the hub.

    MIMI_HUB_SEGMENTER = "<backend>:<model>[,<backend>:<model>...]"   tried in order until one works;
        default codex:gpt-6.1-sol,claude:claude-haiku-5-5 (also agy:gemini-4-argon-high), or "off"

On six episodes (2026-10-31) Sol and Haiku found the same ad breaks, matching hand labels; Sol ~6-15 s,
Haiku ~2-17 s, Argon ~220 s per episode.

CLI: python -m hub.segments <episode.vtt> [backend:model]   (prints the segments JSON)
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

SEGMENTER = os.environ.get("MIMI_HUB_SEGMENTER", "codex:gpt-6.1-sol,claude:claude-haiku-5-5")
SUMMARIZER = os.environ.get("MIMI_HUB_SUMMARIZER", "off")
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


def _run(backend, model, prompt):
    env = dict(os.environ, PATH=_PATH)
    exe = shutil.which(backend, path=_PATH) or backend
    with tempfile.TemporaryDirectory(prefix="mimi-seg-") as tmp:
        if backend == "claude":
            cmd = [exe, "-p", "--model", model, "--tools", "", "--no-session-persistence",
                   "--setting-sources", "", "--strict-mcp-config", "--output-format", "text"]
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=tmp, env=env, timeout=TIMEOUT_S)
            out = r.stdout
        elif backend == "codex":
            last = os.path.join(tmp, "last.txt")
            cmd = [exe, "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only",
                   "-m", model, "-o", last, "-"]
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=tmp, env=env, timeout=TIMEOUT_S)
            out = open(last).read() if os.path.exists(last) else ""
        elif backend == "agy":
            cmd = [exe, "--model", model, "--output-format", "text", "-p", prompt]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp, env=env, timeout=TIMEOUT_S)
            out = r.stdout
        else:
            raise ValueError(f"unknown segmenter backend {backend!r}")
    if r.returncode != 0:
        raise RuntimeError(f"{backend} exited {r.returncode}: {(r.stderr or r.stdout)[-400:]}")
    return out


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


def _ask(segmenter, lines, log):
    """Cue ranges from the first backend in the comma-separated [segmenter] list that answers."""
    errors = []
    for choice in [c.strip() for c in segmenter.split(",") if c.strip()]:
        backend, _, model = choice.partition(":")
        try:
            found = []
            for lo, hi in _windows(lines):
                found += _parse(_run(backend, model, PROMPT + "\n".join(lines[lo:hi]) + "\n"), lo, hi)
            return choice, found
        except Exception as e:  # expired login, timeout, unparseable reply: try the next one
            log(f"segments: {choice} failed: {e}")
            errors.append(f"{choice}: {e}")
    raise RuntimeError("; ".join(errors) or "no segmenter configured")


def detect(cues, segmenter=SEGMENTER, log=print):
    """Skip ranges for [cues]: {"model", "segments": [{"start", "end", "kind", "label", "first", "last"}]}."""
    lines = [_line(i, c) for i, c in enumerate(cues)]
    t0 = time.time()
    used, found = _ask(segmenter, lines, log)
    segs = []
    for s in _merge(found, cues):
        start = cues[s["first"]][0]
        # resume where the next cue starts (skips the music bed / silence after the break too)
        end = cues[s["last"] + 1][0] if s["last"] + 1 < len(cues) else cues[s["last"]][1]
        if end - start >= MIN_S:
            segs.append({"start": round(start, 2), "end": round(end, 2), **s})
    log(f"segments: {len(segs)} via {used} in {time.time() - t0:.0f} s")
    return {"model": used, "segments": segs}


SUMMARY_PROMPT = """You are creating a comprehensive, detailed executive summary and highlights breakdown of a podcast episode based on its transcript.
All advertisements and promotional segments have already been removed.

Create a detailed summary containing enough depth to cover all major discussions, arguments, and takeaways:
1. Overview: 2-3 sentences synthesizing the core premise and big-picture theme.
2. Topics & Key Discussion Highlights: Break down each major segment/topic in order. For each topic, provide detailed bullet points explaining the ideas, arguments, stories, or technical details discussed by the speakers, including approximate timestamps [mm:ss] from the transcript cues where relevant.
3. Notable Takeaways: Key conclusions, memorable insights, recommendations, or lessons.

Format constraints:
- Do not include any conversational filler, introductory remarks, or preambles (no "Sure, here is...").
- Begin your response directly with <summary> and end with </summary>.
- Inside <summary>, format the summary using clean HTML elements: <h4>, <p>, <ul>, <li>, <strong>, <em>.
- Do not use markdown code fences (no ```html).

Clean Transcript:
"""


def _extract_summary(text):
    """Extract clean HTML summary from model output, stripping any conversational preamble or fences."""
    text = text.strip()
    m = re.search(r"<summary[^>]*>(.*?)</summary>", text, re.DOTALL | re.IGNORECASE)
    if m:
        body = m.group(1).strip()
    else:
        m = re.search(r"<summary[^>]*>(.*)", text, re.DOTALL | re.IGNORECASE)
        body = m.group(1).strip() if m else text

    # Strip code fences if present
    body = re.sub(r"^```[a-zA-Z]*\n?", "", body)
    body = re.sub(r"\n?```$", "", body).strip()

    # Trim conversational preamble before the first HTML tag if model omitted tags
    first_tag = re.search(r"<(h[1-6]|p|div|ul|ol|strong|em)[^>]*>", body, re.IGNORECASE)
    if first_tag and 0 < first_tag.start() < 250 and not body.startswith("<"):
        body = body[first_tag.start():].strip()

    return body


def summarize(cues, segments=None, summarizer=SUMMARIZER, log=print):
    """Generate a detailed HTML summary from [cues], excluding cues marked in [segments]."""
    if summarizer.strip().lower() == "auto":
        summarizer = SEGMENTER
    if summarizer.strip().lower() in ("", "off", "0", "false"):
        raise RuntimeError("summarizer disabled")

    skip_cues = set()
    if segments:
        for s in segments:
            first = int(s.get("first", 0))
            last = int(s.get("last", 0))
            for idx in range(first, last + 1):
                skip_cues.add(idx)

    clean_lines = [_line(i, c) for i, c in enumerate(cues) if i not in skip_cues]
    if not clean_lines:
        clean_lines = [_line(i, c) for i, c in enumerate(cues)]

    t0 = time.time()
    errors = []
    for choice in [c.strip() for c in summarizer.split(",") if c.strip()]:
        backend, _, model = choice.partition(":")
        try:
            log(f"summary: generating via {choice} with {len(clean_lines)} clean cues...")
            text = "\n".join(clean_lines)
            prompt = SUMMARY_PROMPT + "\n" + text[:CHUNK_CHARS]
            out = _run(backend, model, prompt)
            out = _extract_summary(out)
            log(f"summary: generated in {time.time() - t0:.1f} s via {choice}")
            return {"model": choice, "summary": out}
        except Exception as e:
            log(f"summary: {choice} failed: {e}")
            errors.append(f"{choice}: {e}")
    raise RuntimeError("; ".join(errors) or "no summarizer configured")


if __name__ == "__main__":
    cues = read_cues(open(sys.argv[1]).read())
    res = detect(cues, *(sys.argv[2:3] or [SEGMENTER]), log=lambda m: print(m, file=sys.stderr))
    for s in res["segments"]:
        print(f"{s['kind']:5s} {int(s['start'] // 60):02d}:{s['start'] % 60:04.1f}-"
              f"{int(s['end'] // 60):02d}:{s['end'] % 60:04.1f}  cues {s['first']}-{s['last']}  {s['label']}",
              file=sys.stderr)
    print(json.dumps(res, indent=1))
