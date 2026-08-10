"""Validate package counts, privacy policy, evidence and report integrity."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".bib", ".bbl", ".csv", ".json", ".md", ".py", ".sh", ".tex",
    ".toml", ".txt", ".yaml", ".yml",
}
IGNORED_DIRS = {".git", ".venv", ".ruff_cache", "__pycache__"}

SOURCE_BRAND = re.compile("k" + r"[-_ ]?" + "dense", re.IGNORECASE)
PRIVATE_SESSION = re.compile(r"/app/sandbox|session_\d{8}_\d{6}_[0-9a-f]+", re.IGNORECASE)
PERSONAL_EMAIL = re.compile("methylation5mc2026" + r"\s*@\s*" + "gmail\\.com", re.IGNORECASE)
PERSONAL_NAME = re.compile("Cheng" + r"\s+" + "bai", re.IGNORECASE)
PLACEHOLDER = re.compile("待" + "填写" + r"|\b(?:TODO|TBD|FIXME)\b|2025_" + "xx")
DATED_NAME = re.compile("rev" + "08" + "11" + "|_" + "08" + "11", re.IGNORECASE)
EXACT_TIME = re.compile(r"20\d{2}-\d{2}-\d{2}(?:T| )\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?")
PRIVATE_PATH = re.compile(r"(?:[A-Za-z]:[\\/]+Users[\\/]+|/home/|/Users/)", re.IGNORECASE)


def release_files() -> list[Path]:
    return [
        path for path in ROOT.rglob("*")
        if path.is_file() and not any(part in IGNORED_DIRS for part in path.parts)
    ]


def text_files() -> list[Path]:
    return [
        path for path in release_files()
        if path.suffix.lower() in TEXT_SUFFIXES
    ]


def assert_counts() -> None:
    py_files = list((ROOT / "workflow").glob("*.py"))
    source_py = [path for path in py_files if path.name != "repo_paths.py"]
    assert len(source_py) == 73, f"expected 73 source Python files, got {len(source_py)}"
    assert len(list((ROOT / "workflow").glob("*.sh"))) == 3
    assert len(list((ROOT / "evidence").glob("*"))) == 71
    assert len(list((ROOT / "figures").glob("*.png"))) == 17


def assert_text_privacy() -> None:
    problems: list[str] = []
    for path in text_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel == "tools/validate_release.py":
            continue
        text = path.read_text(encoding="utf-8-sig")
        checks = [SOURCE_BRAND, PRIVATE_SESSION, PERSONAL_EMAIL, PERSONAL_NAME,
                  PLACEHOLDER, DATED_NAME]
        if rel == "LICENSE":
            checks = checks[:4] + checks[5:]
        for pattern in checks:
            if pattern.search(text) or pattern.search(rel):
                problems.append(f"{rel}: {pattern.pattern}")
        if rel.startswith("evidence/") and EXACT_TIME.search(text):
            problems.append(f"{rel}: exact internal timestamp")
        if PRIVATE_PATH.search(text):
            problems.append(f"{rel}: private absolute path")
    assert not problems, "privacy findings:\n" + "\n".join(problems)


def assert_evidence() -> None:
    for path in (ROOT / "evidence").glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    for path in (ROOT / "evidence").glob("*.csv"):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            list(csv.reader(stream))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_evidence_index() -> None:
    evidence = ROOT / "evidence"
    index_path = evidence / "manifest.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = {item["path"]: item for item in payload["artifacts"]}
    expected = {
        path.relative_to(ROOT).as_posix(): path
        for path in evidence.iterdir()
        if path.is_file() and path != index_path
    }
    assert payload["artifact_count"] == len(expected)
    assert set(indexed) == set(expected), "public evidence index is stale"
    for rel, path in expected.items():
        item = indexed[rel]
        assert item["size_bytes"] == path.stat().st_size, f"size mismatch: {rel}"
        assert item["sha256"] == file_sha256(path), f"hash mismatch: {rel}"


def assert_images() -> None:
    for path in (ROOT / "figures").glob("*.png"):
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            assert not image.info, f"PNG metadata remains in {path.name}: {image.info}"


def assert_report() -> None:
    path = ROOT / "docs" / "report" / "GOAI_virtual_cell_preliminary_report.pdf"
    report = PdfReader(path)
    assert len(report.pages) == 36
    assert not {key: value for key, value in (report.metadata or {}).items() if value}
    page_text = [(page.extract_text() or "").strip() for page in report.pages]
    assert all(page_text), "report contains a page with no extractable text"
    all_text = "\n".join(page_text)
    forbidden = ["团队" + "介绍", "成员" + "背景", "团队" + "分工",
                 "团队" + "成果", "待" + "填写"]
    assert not [term for term in forbidden if term in all_text]


def assert_documentation_integrity() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = []
    for target in re.findall(r"\]\(([^)]+)\)", readme):
        if target.startswith(("http://", "https://", "#")):
            continue
        clean = target.split("#", 1)[0]
        if clean and not (ROOT / clean).exists():
            missing.append(clean)
    assert not missing, f"README links are broken: {missing}"

    report_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "docs" / "report" / "source").glob("*.tex")
    )
    stale = ["results_摘要", "README_代码说明", "progress.md", "03_训练与推理源代码"]
    assert not [term for term in stale if term in report_source], "stale release paths remain"

    source_py = [
        path for path in (ROOT / "workflow").glob("*.py")
        if path.name != "repo_paths.py"
    ]
    line_count = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in source_py)
    displayed = f"{line_count:,}".replace(",", r"\,")
    assert displayed in report_source, (
        f"report code line count is stale: expected LaTeX token {displayed}"
    )


def assert_no_unpublished_payloads() -> None:
    forbidden_suffixes = {".docx", ".zip", ".pt", ".pth", ".ckpt", ".parquet"}
    files = release_files()
    bad = [path.relative_to(ROOT) for path in files
           if path.suffix.lower() in forbidden_suffixes]
    assert not bad, f"unpublished payloads found: {bad}"
    oversized = [path.relative_to(ROOT) for path in files
                 if path.stat().st_size > 25 * 1024 * 1024]
    assert not oversized, f"files exceed 25 MiB: {oversized}"


def main() -> None:
    assert_counts()
    assert_text_privacy()
    assert_evidence()
    assert_evidence_index()
    assert_images()
    assert_report()
    assert_documentation_integrity()
    assert_no_unpublished_payloads()
    print("release validation passed")


if __name__ == "__main__":
    main()
