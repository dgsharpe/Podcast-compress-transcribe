"""An H100 on Modal kept up for fast debugging turns (a Modal Sandbox, the benchmark's image and caches):

    python scripts/modal_box.py start [--gpu H100] [--hours 2] [--idle-min 15] [--vm]
    python scripts/modal_box.py push                      hub/ and scripts/ *.py to the box (after local edits)
    python scripts/modal_box.py run [--timeout 600] CMD   a shell command in the repo dir on the box, output streamed
    python scripts/modal_box.py stop

Guards against a forgotten GPU: the box ends after --hours whatever happens, after --idle-min with no command
running, every run has its own timeout, and stop terminates it. The box's id is kept in ~/.cache/mimi-hub-box.id.
"""
import argparse
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
ID_FILE = Path.home() / ".cache" / "mimi-hub-box.id"


def box():
    if not ID_FILE.exists():
        raise SystemExit("no box: python scripts/modal_box.py start")
    return modal.Sandbox.from_id(ID_FILE.read_text().strip())


def run(sb, cmd, timeout):
    import modal_hub_bench as b
    p = sb.exec("bash", "-c", f"cd {b.NA} && ( {cmd} ) 2>&1", timeout=timeout)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "push", "run", "stop", "status"])
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    ap.add_argument("--gpu", default="H100")
    ap.add_argument("--hours", type=float, default=2)
    ap.add_argument("--idle-min", type=float, default=15)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--vm", action="store_true", help="Modal's VM runtime instead of gVisor")
    a = ap.parse_args()
    import modal_hub_bench as b

    if a.action == "start":
        if ID_FILE.exists():
            try:
                if box().poll() is None:
                    raise SystemExit(f"a box is already running ({ID_FILE.read_text().strip()}); stop it first")
            except modal.exception.NotFoundError:
                pass
        app = modal.App.lookup("mimi-hub-box", create_if_missing=True)
        with modal.enable_output():
            sb = modal.Sandbox.create(
                "sleep", "infinity", app=app, image=b.image, gpu=a.gpu, cpu=32, memory=128 * 1024,
                timeout=int(a.hours * 3600), idle_timeout=int(a.idle_min * 60), workdir=b.NA,
                volumes={"/cache": b.CACHE}, runtime="vm" if a.vm else None,
                env={"TORCHINDUCTOR_CACHE_DIR": "/cache/inductor", "MIMI_HUB_AOTI": "/cache/aoti"})
        ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        ID_FILE.write_text(sb.object_id)
        print(f"box {sb.object_id}: {a.gpu}, ends in {a.hours} h or after {a.idle_min} min idle")
        # the converted Parakeet the hub loads (as the benchmark function does first)
        run(sb, "python -c \"from hub import parakeet_native as p; d='/models/hub/nemo-parakeet-tdt-0.6b-v2-hf'; "
                "p.convert('nemo-parakeet-tdt-0.6b-v2', d); p.convert_decoder('nemo-parakeet-tdt-0.6b-v2', d)\" "
                "&& nvidia-smi --query-gpu=name,memory.used --format=csv,noheader", 900)
    elif a.action == "push":
        import io
        import tarfile
        buf, n = io.BytesIO(), 0
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for d in ("hub", "scripts"):
                for f in sorted((ROOT / d).glob("*.py")):
                    tar.add(f, arcname=f"{d}/{f.name}")
                    n += 1
        p = box().exec("tar", "-xzf", "-", "-C", b.NA, timeout=120)
        p.stdin.write(buf.getvalue())
        p.stdin.write_eof()
        p.stdin.drain()
        rc = p.wait()
        print(f"pushed {n} files" if rc == 0 else f"push failed ({rc}): {p.stderr.read()[:300]}")
    elif a.action == "run":
        cmd = " ".join(a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd)
        sys.exit(run(box(), cmd, a.timeout))
    elif a.action == "status":
        rc = box().poll()
        print("running" if rc is None else f"ended ({rc})")
    elif a.action == "stop":
        box().terminate()
        ID_FILE.unlink()
        print("terminated")


if __name__ == "__main__":
    main()
