"""Minimal PDF text extractor for the problem statement (no third-party deps)."""
import re
import zlib
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "hackathon/Problem Statement.pdf"
data = open(path, "rb").read()

chunks = []
for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
    s = m.group(1)
    try:
        d = zlib.decompress(s)
    except Exception:
        d = s
    chunks.append(d)
blob = b"\n".join(chunks)

# Collect text-show strings from Tj / TJ operators
pieces = []
for m in re.finditer(rb"\((?:[^()\\]|\\.)*\)", blob):
    pieces.append(m.group(0)[1:-1])
text = b" ".join(pieces)
text = text.replace(b"\\(", b"(").replace(b"\\)", b")")
out = text.decode("latin-1", "ignore")
print(out[:30000])
