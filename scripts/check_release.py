"""Check the Git publication boundary, curated documentation links and file hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

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
    docs = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *(ROOT / "docs").glob("*.md")]
    for p in docs:
        if p.name.startswith("historical-"):
            continue
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", p.read_text()):
            if re.match(r"\w+://", target) or target.startswith("#"):
                continue
            local = target.split("#")[0]
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
    print(f"PASS: {len(files)} public files, curated documentation links, and SHA-256 manifest")

if __name__ == "__main__":
    main()
