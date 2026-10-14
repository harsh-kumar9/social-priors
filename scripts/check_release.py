"""Check the Git publication boundary, Markdown links and file hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "provenance/release-files.json"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="Update checksums after reviewed edits")
    a = ap.parse_args()
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    files = [ROOT / n for n in names if n and n != MANIFEST]
    if not files:
        raise SystemExit("Stage the explicit public snapshot before running this check")
    errors = []
    patterns = [re.compile(x) for x in (
        r"gh[pousr]_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{30,}",
        r"hf_[A-Za-z0-9]{25,}", r"sk-(?:proj-|or-v1-)?[A-Za-z0-9_-]{32,}",
        r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")]
    for p in files:
        rel = p.relative_to(ROOT)
        if not p.is_file() or p.is_symlink():
            errors.append(f"Missing/nonregular file: {rel}")
            continue
        if p.suffix in (".safetensors", ".bin", ".pt", ".pth", ".pkl") or (
            p.name.startswith(".env") and p.name != ".env.example"):
            errors.append(f"Disallowed public file: {rel}")
        if rel.parts[0] in ("generated", "generated_io", "training", "training_io", "outputs", "reproduced", "third_party"):
            errors.append(f"Runtime artifact in release: {rel}")
        if p.stat().st_size > 50 * 1024 * 1024:
            errors.append(f"Large file: {rel}")
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in patterns):
            errors.append(f"Potential credential (inspect locally; value not printed): {rel}")
    docs = [p for p in files if p.suffix == ".md" and p.is_file()]
    for p in docs:
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", p.read_text()):
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            local = unquote(url.path)
            if not (p.parent / local).exists():
                errors.append(f"Broken documentation link: {p.relative_to(ROOT)} -> {target}")
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(files) if p.is_file()}
    if errors:
        raise SystemExit("\n".join(errors))
    if a.write:
        (ROOT / MANIFEST).write_text(json.dumps({"sha256": hashes}, indent=2) + "\n")
    else:
        recorded = json.loads((ROOT / MANIFEST).read_text())["sha256"]
        if hashes != recorded:
            diff = sorted(k for k in hashes.keys() | recorded.keys() if hashes.get(k) != recorded.get(k))
            raise SystemExit("Release checksum mismatch: " + ", ".join(diff))
    print(f"PASS: {len(files)} public files, links in {len(docs)} Markdown files, and SHA-256 manifest")

if __name__ == "__main__":
    main()
