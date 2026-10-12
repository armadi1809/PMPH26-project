import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median


WORKLOADS = ((64, 16384), (64, 32768), (128, 8192), (128, 16384))


def median_ms(values: list[Decimal], multiplier: Decimal, path: Path) -> Decimal:
    if not values or any(not value.is_finite() or value <= 0 for value in values):
        raise ValueError(f"{path}: runtimes must be nonempty, finite, and positive")
    return median(values) * multiplier


def cuda_median(root: Path, d: int, n: int) -> Decimal:
    path = root / "cuda" / "results" / f"d{d}-N{n}.runtimes"
    try:
        values = [Decimal(line) for line in path.read_text().splitlines()]
    except InvalidOperation as error:
        raise ValueError(f"{path}: expected one numeric runtime per line") from error
    return median_ms(values, Decimal(1000), path)


def futhark_medians(root: Path, d: int) -> dict[str, Decimal]:
    path = root / "futhark" / "results" / f"attention-main{d}-cuda.json"
    try:
        data = json.loads(path.read_text(), parse_float=Decimal)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    entry = f"custom-alg1-opt.fut:main{d}"
    if not isinstance(data, dict) or not isinstance(data.get(entry), dict):
        raise ValueError(f"{path}: missing benchmark entry {entry}")
    datasets = data[entry].get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"{path}: expected a datasets object")
    results: dict[str, Decimal] = {}
    for label, dataset in datasets.items():
        if not isinstance(dataset, dict) or not isinstance(dataset.get("runtimes"), list):
            raise ValueError(f"{path}: dataset {label!r} is missing its runtimes array")
        values = dataset["runtimes"]
        if any(isinstance(value, bool) or not isinstance(value, (int, float, Decimal)) for value in values):
            raise ValueError(f"{path}: dataset {label!r} has nonnumeric runtimes")
        samples = [Decimal(str(value)) for value in values]
        results[label.strip()] = median_ms(samples, Decimal("0.001"), path)
    return results


def main() -> None:
    root = Path(__file__).resolve().parent
    try:
        futhark = {d: futhark_medians(root, d) for d in (64, 128)}
        rows = []
        for d, n in WORKLOADS:
            label = f"Class {n}-{d}"
            if label not in futhark[d]:
                raise ValueError(f"Futhark main{d} results are missing dataset {label!r}")
            cuda = cuda_median(root, d, n)
            fut = futhark[d][label]
            rows.append((str(d), str(n), f"{cuda:.3f}", f"{fut:.3f}", f"{fut / cuda:.2f}x"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"Cannot summarize results: {error}") from error

    headers = ("d", "N", "CUDA median (ms)", "Futhark median (ms)", "CUDA speedup")
    widths = [max(3, max(map(len, column))) for column in zip(headers, *rows)]
    print("| " + " | ".join(cell.rjust(width) for cell, width in zip(headers, widths)) + " |")
    print("| " + " | ".join("-" * (width - 1) + ":" for width in widths) + " |")
    for row in rows:
        print("| " + " | ".join(cell.rjust(width) for cell, width in zip(row, widths)) + " |")


if __name__ == "__main__":
    main()
