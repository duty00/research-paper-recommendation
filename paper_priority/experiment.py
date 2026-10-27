"""Development CV selection, untouched holdout evaluation, then full-data refit."""
from __future__ import annotations

import importlib.metadata
import platform
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from .data import FIELDS, Dataset, paper_groups, sha256, write_csv, write_json
from .metric import make_priorities, score_priorities
from .models import BASE_MODELS, CANDIDATES, combine, fit_model, predict_gain, rows_to_x

FRACTIONS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
PACKAGES = ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "striprtf")


def summarize(metrics: list[dict]) -> dict:
    return {field: float(np.mean([m[field] for m in metrics]))
            for field in ("score", "relevance", "coverage")} | {
        "fold_score_std": float(np.std([m["score"] for m in metrics], ddof=1))}


def evaluate_gain(ids, y, gain, fraction):
    return score_priorities(ids, y, make_priorities(ids, gain, fraction))


def assert_disjoint_groups(groups, fit_indices, validation_indices):
    if set(groups[fit_indices]) & set(groups[validation_indices]):
        raise AssertionError("A duplicate group crossed the train/validation boundary.")


def venue_baseline(train_rows, train_y, validation_rows):
    gains = (np.asarray(train_y) - 1) / 4
    prior = float(np.mean(gains))
    counts, totals = Counter(), Counter()
    for row, gain in zip(train_rows, gains):
        counts[row["venue"]] += 1
        totals[row["venue"]] += float(gain)
    return np.asarray([(totals[r["venue"]] + 10 * prior) / (counts[r["venue"]] + 10)
                       for r in validation_rows])


def save_prepared(data: Dataset, out: Path):
    write_json(out / "reports/data_audit.json", data.audit)
    write_csv(out / "prepared/train.csv", data.train, FIELDS + ["Label"])
    write_csv(out / "prepared/test.csv", data.test, FIELDS)


def validate_submission(path: Path, expected_ids: list[str]) -> dict:
    import csv
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "priority"]:
            raise ValueError("Submission columns must be exactly id,priority, in that order.")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError("Malformed submission rows.")
    ids = [r["id"] for r in rows]
    priorities = np.asarray([float(r["priority"]) for r in rows])
    if len(ids) != len(expected_ids) or len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError("Submission IDs are missing, duplicated or unexpected.")
    if not np.all(np.isfinite(priorities)) or np.any(priorities < 0):
        raise ValueError("Submission priorities must be finite and nonnegative.")
    return {"rows": len(ids), "recommended": int(np.sum(priorities > 0)),
            "sha256": sha256(path)}


def write_submission(path, sample, test_ids, gain, fraction):
    priorities = make_priorities(test_ids, gain, fraction)
    by_id = dict(zip(test_ids, priorities))
    write_csv(path, [{"id": r["id"], "priority": format(by_id[r["id"]], ".12g")}
                     for r in sample], ["id", "priority"])
    return validate_submission(path, list(test_ids))


def run_experiment(data: Dataset, out: Path, seed: int = 261, folds: int = 5):
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output folder is not empty. Choose another --out: {out}")
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    np.random.seed(seed)
    save_prepared(data, out)
    rows = data.train
    x, xt = rows_to_x(rows), rows_to_x(data.test)
    y = np.asarray([row["Label"] for row in rows], dtype=int)
    ids = np.asarray([row["id"] for row in rows], dtype=str)
    groups = np.asarray(paper_groups(rows))
    test_ids = np.asarray([row["id"] for row in data.test], dtype=str)
    target_venues = sorted({row["venue"] for row in data.test})
    # Only the set of visible test venue names informs evaluation scope.
    # Test text, labels and full-test vocabularies never enter model fitting.
    target_mask = np.asarray([row["venue"] in target_venues for row in rows])
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dev, holdout = next(outer.split(x, y, groups))
    assert_disjoint_groups(groups, dev, holdout)
    inner = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + 1)
    cv_splits = list(inner.split(x[dev], y[dev], groups[dev]))
    fold_ids = np.full(len(dev), -1, dtype=int)
    oof = {name: np.zeros(len(dev)) for name in BASE_MODELS}
    baseline_oof = np.zeros(len(dev))
    equal_oof = np.zeros(len(dev))
    print(f"Development: {len(dev)}; untouched holdout: {len(holdout)}", flush=True)
    print(f"Primary evaluation venues: {', '.join(target_venues)}", flush=True)
    warnings.filterwarnings("error", category=ConvergenceWarning)
    with threadpool_limits(limits=1):
        for fold, (fit_local, val_local) in enumerate(cv_splits):
            fit_idx, val_idx = dev[fit_local], dev[val_local]
            assert_disjoint_groups(groups, fit_idx, val_idx)
            fold_ids[val_local] = fold
            fit_rows, val_rows = [rows[i] for i in fit_idx], [rows[i] for i in val_idx]
            baseline_oof[val_local] = venue_baseline(fit_rows, y[fit_idx], val_rows)
            equal_oof[val_local] = float(np.mean((y[fit_idx] - 1) / 4))
            for name in BASE_MODELS:
                print(f"CV fold {fold + 1}/{folds}: {name}", flush=True)
                model = fit_model(name, x[fit_idx], y[fit_idx], seed)
                oof[name][val_local] = predict_gain(model, x[val_idx])
        if np.any(fold_ids < 0):
            raise AssertionError("Some development rows have no out-of-fold predictions.")

        cv_results, best_by_candidate = [], []
        dev_target = target_mask[dev]
        for name, spec in CANDIDATES.items():
            gain = combine(oof, spec["weights"])
            results = []
            for fraction in FRACTIONS:
                fold_metrics = []
                for fold in range(folds):
                    mask = (fold_ids == fold) & dev_target
                    if not np.any(mask):
                        raise ValueError("A validation fold contains no target-venue papers.")
                    fold_metrics.append(evaluate_gain(ids[dev][mask], y[dev][mask], gain[mask], fraction))
                record = {"candidate": name, "family": spec["family"], "fraction": fraction,
                          **summarize(fold_metrics)}
                results.append(record)
            cv_results.extend(results)
            best_by_candidate.append(max(results, key=lambda r: (r["score"], r["fraction"])))
        ranking = sorted(best_by_candidate, key=lambda r: (-r["score"], r["candidate"]))
        selected = [ranking[0]]
        # Second candidate is from a different modeling family for a useful comparison.
        selected.append(next(r for r in ranking[1:] if r["family"] != selected[0]["family"]))
        print("Selected on development CV only:", flush=True)
        for record in selected:
            print(f"  {record['candidate']}, fraction={record['fraction']:.2f}, CV={record['score']:.6f}", flush=True)

        def baseline_cv(gain):
            return summarize([evaluate_gain(ids[dev][m], y[dev][m], gain[m], 1.0)
                              for fold in range(folds)
                              for m in [(fold_ids == fold) & dev_target]])

        baselines = {"equal_priority_all": {"development_cv": baseline_cv(equal_oof)},
                     "smoothed_venue_all": {"development_cv": baseline_cv(baseline_oof)}}
        needed = sorted({base for record in selected for base in CANDIDATES[record["candidate"]]["weights"]})
        hold_predictions = {}
        for name in needed:
            print(f"Fit development / evaluate holdout: {name}", flush=True)
            model = fit_model(name, x[dev], y[dev], seed)
            hold_predictions[name] = predict_gain(model, x[holdout])
        hold_mask = target_mask[holdout]
        hold_venue = venue_baseline([rows[i] for i in dev], y[dev], [rows[i] for i in holdout])
        for name, gains in (("equal_priority_all", np.full(len(holdout), np.mean((y[dev] - 1) / 4))),
                            ("smoothed_venue_all", hold_venue)):
            baselines[name]["holdout_target_venues"] = evaluate_gain(
                ids[holdout][hold_mask], y[holdout][hold_mask], gains[hold_mask], 1.0)
            baselines[name]["holdout_all_venues"] = evaluate_gain(ids[holdout], y[holdout], gains, 1.0)
        for record in selected:
            gain = combine(hold_predictions, CANDIDATES[record["candidate"]]["weights"])
            record["holdout_target_venues"] = evaluate_gain(
                ids[holdout][hold_mask], y[holdout][hold_mask], gain[hold_mask], record["fraction"])
            record["holdout_all_venues"] = evaluate_gain(ids[holdout], y[holdout], gain, record["fraction"])
        # Selection and fractions are now frozen; holdout does not change either.
        full_models, test_predictions = {}, {}
        for name in needed:
            print(f"Refit all training data: {name}", flush=True)
            full_models[name] = fit_model(name, x, y, seed)
            test_predictions[name] = predict_gain(full_models[name], xt)
        for number, record in enumerate(selected, start=1):
            name = record["candidate"]
            weights = CANDIDATES[name]["weights"]
            bundle = {"format_version": 1, "candidate": name, "fraction": record["fraction"],
                      "seed": seed, "weights": weights,
                      "models": {base: full_models[base] for base in weights},
                      "raw_test_sha256": data.audit["files"]["test"]["sha256"],
                      "test_ids": list(test_ids), "train_sha256": data.audit["files"]["train"]["sha256"],
                      "versions": {package: importlib.metadata.version(package) for package in PACKAGES}}
            bundle_path = out / f"models/candidate_{number}.joblib"
            bundle_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, bundle_path, compress=3)
            submission_path = out / f"submissions/candidate_{number}.csv"
            gain = combine(test_predictions, weights)
            record["submission_check"] = write_submission(
                submission_path, data.sample, test_ids, gain, record["fraction"])
            record["model_file"] = str(bundle_path.relative_to(out))
            record["submission_file"] = str(submission_path.relative_to(out))
            write_csv(out / f"predictions/candidate_{number}_test.csv",
                      [{"id": row_id, "predicted_gain": float(value)} for row_id, value in zip(test_ids, gain)],
                      ["id", "predicted_gain"])
        for record in selected:
            gain = combine(hold_predictions, CANDIDATES[record["candidate"]]["weights"])
            write_csv(out / f"predictions/holdout_{record['candidate']}.csv",
                      [{"id": ids[index], "Label": int(y[index]), "in_target_venues": bool(target_mask[index]),
                        "predicted_gain": float(value), "group": groups[index]}
                       for index, value in zip(holdout, gain)],
                      ["id", "Label", "in_target_venues", "predicted_gain", "group"])

    write_csv(out / "reports/cv_results.csv", cv_results,
              ["candidate", "family", "fraction", "score", "relevance", "coverage", "fold_score_std"])
    write_csv(out / "reports/development_oof.csv",
              [{"id": ids[index], "Label": int(y[index]), "fold": int(fold_ids[local]),
                "group": groups[index], "in_target_venues": bool(target_mask[index]),
                **{name: float(values[local]) for name, values in oof.items()}}
               for local, index in enumerate(dev)],
              ["id", "Label", "fold", "group", "in_target_venues", *BASE_MODELS])
    write_csv(out / "reports/splits.csv",
              [{"id": ids[index], "partition": "development", "cv_fold": int(fold_ids[local]),
                "group": groups[index]} for local, index in enumerate(dev)] +
              [{"id": ids[index], "partition": "holdout", "cv_fold": "", "group": groups[index]}
               for index in holdout], ["id", "partition", "cv_fold", "group"])
    results = {
        "selection_rule": "Best development-CV score; second best from a different family. Holdout never selects candidates.",
        "metric": "Detailed Evaluation page: harmonic mean of relevance and worthwhile-paper recall; hidden code unverified.",
        "validation": {"seed": seed, "cv_folds": folds, "outer_folds": 5,
                       "holdout_fold": 0, "development_rows": len(dev), "holdout_rows": len(holdout),
                       "primary_venues": target_venues, "holdout_primary_rows": int(np.sum(hold_mask)),
                       "target_venue_selection_uses_visible_test_metadata": True,
                       "holdout_venues": dict(Counter(rows[i]["venue"] for i in holdout)),
                       "development_primary_rows": int(np.sum(dev_target)),
                       "candidate_parameters": BASE_MODELS, "candidate_definitions": CANDIDATES,
                       "selection_fractions": FRACTIONS},
        "baselines": baselines, "candidate_ranking": ranking, "selected_candidates": selected,
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "limitations": [
            "Competition abstract conflicts with detailed Evaluation page; no hidden metric code was verified.",
            "CV scores tune models/fractions and are development estimates; use untouched holdout as a separate check.",
            "A single holdout and CV-fold standard deviation do not establish population confidence intervals.",
            "Venue distribution differs between train and test. Primary validation excludes train-only venues, not all distribution shift.",
            "Validation list lengths differ from hidden 298-paper splits; scores are not directly comparable to Public scores.",
            "Same title or DOI/URL is a conservative grouping rule; repeated links need not identify identical papers.",
            "No external data or pretrained models are used. No Kaggle submission was made.",
        ],
    }
    write_json(out / "reports/results.json", results)
    root = Path(__file__).resolve().parents[1]
    write_report(out, data, results)
    code_files = [root / "main.py", root / "requirements.txt", *sorted((root / "paper_priority").glob("*.py"))]
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(),
                "machine": platform.machine(), "processor": platform.processor(),
                "threads": 1, "seed": seed,
                "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
                "input_files": data.audit["files"],
                "code_sha256": {str(path.relative_to(root)): sha256(path) for path in code_files},
                "outputs_sha256": {str(path.relative_to(out)): sha256(path)
                                   for path in out.rglob("*") if path.is_file()}}
    write_json(out / "manifest.json", manifest)
    print(f"Done: {out.resolve()}", flush=True)
    return results


def write_report(out, data, results):
    lines = ["# Kết quả thử nghiệm v1", "", "Đây là kết quả validation nội bộ, chưa phải điểm Kaggle.", "",
             "## Dữ liệu", "", f"Train: {len(data.train)}; test: {len(data.test)}.",
             f"Test.csv đã chuyển từ RTF: {data.audit['files']['test']['rtf_converted']}.",
             f"Số tác giả thiếu trong train/test: {data.audit['missing']['train']['authors']}/{data.audit['missing']['test']['authors']}.",
             f"Số dòng trùng tiêu đề sau chuẩn hóa ngoài lần xuất hiện đầu: {data.audit['duplicate_title_extra_rows']}.",
             "", "## Thiết kế đánh giá", "",
             f"Giữ riêng một fold (~20%) của train làm holdout. Dùng {results['validation']['cv_folds']}-fold StratifiedGroupKFold trong development để chọn mô hình và tỷ lệ đề xuất.",
             "Những bài có cùng tiêu đề chuẩn hóa hoặc DOI/URL được giữ cùng nhóm. Từ vựng và mô hình chỉ học từ phần training của mỗi fold.",
             "Danh sách venue của metadata test công khai được dùng để xác định phạm vi validation chính (cav, kr, lics, lpnmr). Không dùng test text để fit từ vựng hay mô hình.",
             "ICLP chỉ có trong train. Vẫn dùng ICLP khi huấn luyện nhưng không đưa vào metric chính. Metric trên tất cả venue được lưu riêng trong results.json.",
             "Chọn ứng viên bằng điểm CV development. Holdout chỉ dùng để kiểm tra sau khi chốt; không đổi ứng viên hay ngưỡng theo holdout.",
             "", "## Baseline và ứng viên", "",
             "| Mô hình | CV development | Holdout relevance | Holdout coverage | Holdout score |", "|---|---:|---:|---:|---:|"]
    for name, base in results["baselines"].items():
        m = base["holdout_target_venues"]
        lines.append(f"| {name} | {base['development_cv']['score']:.6f} | {m['relevance']:.6f} | {m['coverage']:.6f} | {m['score']:.6f} |")
    for number, record in enumerate(results["selected_candidates"], start=1):
        m = record["holdout_target_venues"]
        lines.append(f"| Ứng viên {number}: {record['candidate']} | {record['score']:.6f} | {m['relevance']:.6f} | {m['coverage']:.6f} | {m['score']:.6f} |")
    lines += ["", "Các điểm trên cùng phạm vi venue. Tỷ lệ đề xuất và độ lệch chuẩn giữa các fold có trong results.json/cv_results.csv.",
              "Baseline Public của host được quan sát là 0.60953 ngày 19/10/2026; không so sánh trực tiếp với validation vì khác tập và độ dài danh sách.",
              "", "## Giới hạn", ""]
    lines += [f"- {value}" for value in results["limitations"]]
    lines += ["", f"Thời gian chạy: {results['runtime_seconds']:.1f} giây, giới hạn 1 luồng tính toán.",
              "", "Nguồn: https://www.kaggle.com/competitions/assignment-data-mining-261/overview", ""]
    (out / "reports/summary.md").write_text("\n".join(lines), encoding="utf-8")


def predict_bundle(bundle_path: Path, data: Dataset, output: Path):
    if output.resolve() in {path.resolve() for path in data.paths.values()}:
        raise ValueError("Prediction output must not overwrite an original data file.")
    # Only load bundles produced by this project and trusted by the user.
    bundle = joblib.load(bundle_path)
    current_versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    if current_versions != bundle["versions"]:
        raise ValueError("Dependency versions differ from training. Install requirements.txt first.")
    if sha256(data.paths["test"]) != bundle["raw_test_sha256"]:
        raise ValueError("Test input hash differs from the original training run.")
    if sha256(data.paths["train"]) != bundle["train_sha256"]:
        raise ValueError("Train input hash differs from the original training run.")
    test_ids = np.asarray([r["id"] for r in data.test], dtype=str)
    if list(test_ids) != bundle["test_ids"]:
        raise ValueError("Test IDs/order changed.")
    with threadpool_limits(limits=1):
        x = rows_to_x(data.test)
        predictions = {name: predict_gain(model, x) for name, model in bundle["models"].items()}
        gain = combine(predictions, bundle["weights"])
    return write_submission(output, data.sample, test_ids, gain, bundle["fraction"])
