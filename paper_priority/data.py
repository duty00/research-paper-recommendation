"""Read originals without changing them; repair RTF and validate identifiers."""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from striprtf.striprtf import rtf_to_text

FIELDS = ["id", "title", "venue", "year", "authors", "doi"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_text(value: str) -> str:
    value = html.unescape(html.unescape(value or ""))
    return " ".join(unicodedata.normalize("NFKC", value).split())


def read_rows(path: Path) -> tuple[list[dict], bool]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    is_rtf = text.lstrip().startswith("{\\rtf")
    if is_rtf:
        # A proper RTF parser preserves escaped braces, Unicode and line breaks.
        text = rtf_to_text(text)
    reader = csv.DictReader(io.StringIO(text.strip(), newline=""), strict=True)
    if not reader.fieldnames:
        raise ValueError(f"Empty CSV: {path}")
    reader.fieldnames = [field.strip() for field in reader.fieldnames]
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError(f"Duplicate column names: {path}")
    result = []
    for line, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV row {line}: {path}")
        result.append({key.strip(): value.strip() for key, value in row.items()})
    return result, is_rtf


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def normalized_title(row: dict) -> str:
    return re.sub(r"[^\w]+", " ", row["title"].casefold()).strip()


def doi_key(row: dict) -> str:
    value = row["doi"].casefold().strip().rstrip("/.")
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def paper_groups(rows: list[dict]) -> list[str]:
    """Connected components of repeated normalized titles or DOI/URL identifiers."""
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        for kind, key in (("title", normalized_title(row)), ("doi", doi_key(row))):
            if not key:
                continue
            token = (kind, key)
            if token in seen:
                parent[find(index)] = find(seen[token])
            else:
                seen[token] = index
    return [str(find(i)) for i in range(len(rows))]


@dataclass
class Dataset:
    train: list[dict]
    test: list[dict]
    sample: list[dict]
    paths: dict[str, Path]
    audit: dict


def find_data_dir(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"Data directory does not exist: {path}")
        return path
    for path in (Path("assignment-data-mining-261"), Path("../assignment-data-mining-261"),
                 Path("data/raw"), Path(".")):
        if (path / "train.csv").exists():
            return path.resolve()
    raise ValueError("Cannot find data. Pass --data-dir with train.csv, Test.csv and Submission.csv.")


def load_dataset(directory: Path) -> Dataset:
    names = {p.name.casefold(): p for p in directory.iterdir() if p.is_file()}
    aliases = {"train": ["train.csv"], "test": ["test.csv"],
               "sample": ["submission.csv", "sample_submission.csv"]}
    paths, loaded, repairs = {}, {}, {}
    for kind, choices in aliases.items():
        path = next((names[name] for name in choices if name in names), None)
        if path is None:
            raise ValueError(f"Missing {kind} file in {directory}; expected {choices}")
        paths[kind] = path
        loaded[kind], repairs[kind] = read_rows(path)
        if not loaded[kind]:
            raise ValueError(f"No records in {path}")
    for kind in ("train", "test"):
        rows = loaded[kind]
        required = set(FIELDS + (["Label"] if kind == "train" else []))
        if not required.issubset(rows[0]):
            raise ValueError(f"Missing columns in {paths[kind]}: {required - set(rows[0])}")
        if kind == "test" and "Label" in rows[0]:
            raise ValueError("Test labels must not be used.")
        for row in rows:
            for field in FIELDS:
                row[field] = clean_text(row[field])
            if not row["id"] or not row["title"]:
                raise ValueError(f"Missing id/title in {kind}")
            if row["year"]:
                year = float(row["year"])
                if not year.is_integer() or not 1800 <= year <= 2100:
                    raise ValueError(f"Invalid year for id={row['id']}")
                row["year"] = str(int(year))
            if kind == "train":
                label = float(row["Label"])
                if label not in (1, 2, 3, 4, 5):
                    raise ValueError(f"Invalid Label for id={row['id']}")
                row["Label"] = int(label)
    if set(loaded["sample"][0]) != {"id", "priority"}:
        raise ValueError("Submission template must have exactly id,priority columns.")
    ids = {}
    for kind, rows in loaded.items():
        ids[kind] = [row["id"].strip() for row in rows]
        if len(set(ids[kind])) != len(ids[kind]):
            raise ValueError(f"Duplicate IDs in {kind}")
    if set(ids["train"]) & set(ids["test"]):
        raise ValueError("Train and test IDs overlap.")
    if set(ids["test"]) != set(ids["sample"]):
        raise ValueError("Test IDs do not match submission template IDs.")
    if len(loaded["train"]) != 2494 or len(loaded["test"]) != 596:
        raise ValueError("Expected competition files with 2494 train and 596 test rows.")

    train, test = loaded["train"], loaded["test"]
    duplicates = defaultdict(list)
    for row in train:
        duplicates[normalized_title(row)].append(row)
    conflicts = [{"ids": [r["id"] for r in members], "labels": [r["Label"] for r in members]}
                 for members in duplicates.values() if len({r["Label"] for r in members}) > 1]
    audit = {
        "source": "https://www.kaggle.com/competitions/assignment-data-mining-261/data",
        "originals_unchanged": True,
        "files": {kind: {"name": path.name, "sha256": sha256(path), "rtf_converted": repairs[kind]}
                  for kind, path in paths.items()},
        "rows": {kind: len(rows) for kind, rows in loaded.items()},
        "labels": dict(sorted(Counter(r["Label"] for r in train).items())),
        "missing": {kind: {field: sum(not r[field] for r in rows) for field in FIELDS}
                    for kind, rows in (("train", train), ("test", test))},
        "venues": {kind: dict(Counter(r["venue"] for r in rows))
                   for kind, rows in (("train", train), ("test", test))},
        "duplicate_title_extra_rows": len(train) - len(duplicates),
        "duplicate_groups": len(train) - len(set(paper_groups(train))),
        "conflicting_title_labels": conflicts,
        "shared_train_test_titles": len({normalized_title(r) for r in train} & {normalized_title(r) for r in test}),
        "year_range": {kind: [min(int(r["year"]) for r in rows if r["year"]),
                               max(int(r["year"]) for r in rows if r["year"])]
                       for kind, rows in (("train", train), ("test", test))},
    }
    return Dataset(train, test, loaded["sample"], paths, audit)
