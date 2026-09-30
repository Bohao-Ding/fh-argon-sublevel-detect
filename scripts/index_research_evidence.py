"""Index the additive frozen evidence without rewriting historical receipts."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "source_data_package/research_evidence"


def main():
    sources = json.loads((EVIDENCE / "SOURCE_MAP.json").read_text(encoding="utf-8"))
    artifacts = sorted(p for p in EVIDENCE.rglob("*") if p.is_file() and p.name not in {"FILE_INDEX.csv", "SHA256SUMS.txt"})
    with (EVIDENCE / "FILE_INDEX.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256", "original_workspace_path"])
        for path in artifacts:
            relative = path.relative_to(EVIDENCE).as_posix()
            origin = sources.get(path.relative_to(ROOT).as_posix(), "integration documentation")
            writer.writerow([relative, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest(), origin])
    artifacts.append(EVIDENCE / "FILE_INDEX.csv")
    (EVIDENCE / "SHA256SUMS.txt").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(EVIDENCE).as_posix()}\n"
        for p in sorted(artifacts)), encoding="utf-8")
    print(f"Indexed {len(artifacts)} published files.")


if __name__ == "__main__":
    main()
