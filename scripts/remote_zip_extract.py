"""Extract selected members of a remote zip via HTTP range requests (no full download).

usage: python -I scripts/remote_zip_extract.py URL OUTDIR SUFFIX [SUFFIX ...]
Extracts every member whose path ends with one of the SUFFIXes, flattened into OUTDIR.
With no SUFFIX matches it lists members containing 'lib/' and exits.
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0"}


class HttpRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        r = urllib.request.urlopen(urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"}))
        self.url = r.geturl()  # follow the redirect once
        self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req).read()
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def main(url, out, suffixes):
    f = io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20)
    z = zipfile.ZipFile(f)
    names = z.namelist()
    hits = [n for n in names if any(n.endswith(s) for s in suffixes)]
    if not hits:
        for n in names:
            if "/lib/" in n and n.endswith(".so"):
                print(n)
        return
    Path(out).mkdir(parents=True, exist_ok=True)
    for n in hits:
        dest = Path(out) / Path(n).name
        dest.write_bytes(z.read(n))
        print(f"{n} -> {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
