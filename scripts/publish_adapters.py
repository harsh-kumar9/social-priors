"""Validate and optionally publish six raw seed-0 adapters. Dry-run by default."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("P", "S", "N", "A", "MIX", "REPLAY")
FILES = ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
         "tokenizer_config.json", "chat_template.jinja")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect_run(source, arm):
    run = source / f"sdf-{arm}-32b-seed0"
    adapter = run / "adapter"
    manifest = json.loads((run / "train_manifest.json").read_text())
    config = json.loads((adapter / "adapter_config.json").read_text())
    if manifest["arm"] != arm or manifest["seed"] != 20260808:
        raise ValueError(f"Wrong condition/seed: {run}")
    if manifest["model_id"] != "Qwen/Qwen3-32B" or config["base_model_name_or_path"] != "Qwen/Qwen3-32B":
        raise ValueError(f"Wrong base model: {run}")
    if config["r"] != 32 or config["lora_alpha"] != 64 or config["peft_type"] != "LORA":
        raise ValueError(f"Wrong adapter recipe: {run}")
    paths = {name: adapter / name for name in FILES}
    paths["train_manifest.json"] = run / "train_manifest.json"
    paths["LICENSE"] = ROOT / "artifacts/ADAPTER-LICENSE"
    for p in paths.values():
        if not p.is_file() or p.is_symlink():
            raise ValueError(f"Missing file or symlink: {p}")
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("adapter-publication.json"))
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", a.namespace):
        ap.error("Invalid Hugging Face namespace")
    # Validate all six before any network mutation.
    sources = {arm: inspect_run(a.source_root.resolve(), arm) for arm in ARMS}
    for arm, files in sources.items():
        print(arm, sum(p.stat().st_size for p in files.values()), "bytes; validated", flush=True)
    if not a.publish:
        print("Dry run: no files uploaded. Namespace:", a.namespace)
        return
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    api = HfApi()
    identity = api.whoami()
    allowed = {identity["name"], *(o["name"] for o in identity.get("orgs", []))}
    if a.namespace not in allowed:
        raise SystemExit("Namespace is not the logged-in account or one of its organizations")
    published = []
    with tempfile.TemporaryDirectory(prefix="social-priors-cards-") as tmp:
        for arm, files in sources.items():
            repo_id = f"{a.namespace}/social-priors-qwen3-32b-{arm.lower()}-seed0"
            card = Path(tmp) / f"{arm}.md"
            card.write_text((ROOT / f"artifacts/model-cards/{arm}.md").read_text().replace("HF_NAMESPACE", a.namespace))
            files = {**files, "README.md": card}
            hashes = {name: digest(path) for name, path in files.items()}
            release = {"release": "social-priors-v0.1.0", "arm": arm, "seed": 20260808,
                       "base_model": "Qwen/Qwen3-32B", "historical_base_revision": None,
                       "sha256": hashes}
            # Existing identical releases are resumable; different contents are never overwritten.
            exists = api.repo_exists(repo_id, repo_type="model")
            if exists and any(n != ".gitattributes" for n in api.list_repo_files(repo_id)):
                old = json.loads(Path(hf_hub_download(repo_id, "release-manifest.json", token=api.token)).read_text())
                if old != release:
                    raise SystemExit(f"Refusing to overwrite different content: {repo_id}")
                info = api.model_info(repo_id, files_metadata=True)
                print("Already uploaded; verifying", repo_id, flush=True)
            else:
                if not exists:
                    api.create_repo(repo_id, repo_type="model", private=True)
                rp = Path(tmp) / f"{arm}.json"
                rp.write_text(json.dumps(release, indent=2) + "\n")
                operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(p)) for name, p in files.items()]
                operations.append(CommitOperationAdd(path_in_repo="release-manifest.json", path_or_fileobj=str(rp)))
                print("Uploading", repo_id, flush=True)
                api.create_commit(repo_id, operations=operations, commit_message="Release raw seed-0 Social Priors adapter")
                info = api.model_info(repo_id, files_metadata=True)
            weights = next(f for f in info.siblings if f.rfilename == "adapter_model.safetensors")
            if weights.lfs is None or weights.lfs.sha256 != hashes["adapter_model.safetensors"]:
                raise SystemExit(f"Uploaded weight checksum mismatch: {repo_id}")
            api.update_repo_settings(repo_id, private=False)
            published.append({"arm": arm, "repo_id": repo_id, "url": f"https://huggingface.co/{repo_id}",
                              "revision": info.sha, "weight_sha256": hashes["adapter_model.safetensors"]})
            a.output.write_text(json.dumps(published, indent=2) + "\n")
            print("Published", repo_id, info.sha, flush=True)

if __name__ == "__main__":
    main()
