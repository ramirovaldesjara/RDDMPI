from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_PY = PROJECT_ROOT / "run_exp.py"
CONFIG_ROOT = PROJECT_ROOT / "lab" / "configs" / "imputation_pypots"
ROOT_BASE = (PROJECT_ROOT / "dataset").resolve()

# Ignore helper/default files
SKIP_FILENAMES = {"default_temp.yaml", "default.yaml"}

# ============================================================
# FILTERS



INCLUDE_DATASETS = {"ETTh1", "ETTh2", "exchange", "illness", "weather"}
INCLUDE_MODELS = {"T1", "RDDMPI"}




# Defaults when filters are not set above
INCLUDE_DATASETS = globals().get("INCLUDE_DATASETS", None)
EXCLUDE_DATASETS = globals().get("EXCLUDE_DATASETS", set())
INCLUDE_MODELS = globals().get("INCLUDE_MODELS", None)
EXCLUDE_MODELS = globals().get("EXCLUDE_MODELS", set())




# ============================================================
# HELPERS
# ============================================================

def collect_configs() -> list[str]:
    if not CONFIG_ROOT.exists():
        raise FileNotFoundError(f"Config root not found: {CONFIG_ROOT}")

    yaml_paths = [p for p in CONFIG_ROOT.rglob("*.yaml") if should_keep_config(p)]

    model_priority = {
        "T1": 0,
        "RDDMPI": 1,
    }

    def sort_key(yaml_path: Path):
        parts = yaml_path.relative_to(CONFIG_ROOT).parts
        dataset, model = parts[0], parts[1]
        priority = model_priority.get(model, 999)
        return (priority, dataset, model, yaml_path.stem)

    yaml_paths = sorted(yaml_paths, key=sort_key)

    configs = [yaml_to_config_name(yaml_path) for yaml_path in yaml_paths]
    return configs

def should_keep_config(yaml_path: Path) -> bool:
    if yaml_path.name in SKIP_FILENAMES:
        return False

    parts = yaml_path.relative_to(CONFIG_ROOT).parts
    # Expect: <dataset>/<model>/<config>.yaml
    if len(parts) < 3:
        return False

    dataset, model = parts[0], parts[1]

    if INCLUDE_DATASETS is not None and dataset not in INCLUDE_DATASETS:
        return False

    if dataset in EXCLUDE_DATASETS:
        return False

    if INCLUDE_MODELS is not None and model not in INCLUDE_MODELS:
        return False

    if model in EXCLUDE_MODELS:
        return False

    # Keep only real experiment configs like 9062_0000.yaml
    if not re.fullmatch(r"\d{4}_\d{4}", yaml_path.stem):
        return False

    return True


def yaml_to_config_name(yaml_path: Path) -> str:
    # Convert:
    # lab/configs/imputation_pypots/ETTh1/T1/9062_0000.yaml
    # to:
    # imputation_pypots/ETTh1/T1/9062_0000
    rel = yaml_path.relative_to(PROJECT_ROOT / "lab" / "configs")
    return str(rel.with_suffix("")).replace("\\", "/")


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    if not RUN_PY.exists():
        raise FileNotFoundError(f"run_exp.py not found at: {RUN_PY}")

    if not ROOT_BASE.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {ROOT_BASE}\n"
            f"Change ROOT_BASE in this script to the correct repo dataset folder."
        )

    configs = collect_configs()

    if not configs:
        print("No configs found.")
        return 1

    print("=" * 100)
    print("PROJECT ROOT :", PROJECT_ROOT)
    print("RUN FILE     :", RUN_PY)
    print("CONFIG ROOT  :", CONFIG_ROOT)
    print("DATASET ROOT :", ROOT_BASE)
    print(f"FOUND {len(configs)} CONFIG(S)")
    print("=" * 100)

    if INCLUDE_DATASETS is not None:
        print("INCLUDING DATASETS:", sorted(INCLUDE_DATASETS))
    else:
        print("INCLUDING DATASETS: ALL")

    if EXCLUDE_DATASETS:
        print("EXCLUDING DATASETS:", sorted(EXCLUDE_DATASETS))
    else:
        print("EXCLUDING DATASETS: NONE")

    if INCLUDE_MODELS is not None:
        print("INCLUDING MODELS  :", sorted(INCLUDE_MODELS))
    else:
        print("INCLUDING MODELS  : ALL")

    if EXCLUDE_MODELS:
        print("EXCLUDING MODELS  :", sorted(EXCLUDE_MODELS))
    else:
        print("EXCLUDING MODELS  : NONE")

    print("-" * 100)
    for cfg in configs:
        print("  -", cfg)
    print()

    failed = []

    for i, cfg in enumerate(configs, start=1):
        print("=" * 100)
        print(f"[{i}/{len(configs)}] RUNNING: {cfg}")
        print("=" * 100)

        cmd = [
            sys.executable,
            str(RUN_PY),
            f"--config-name={cfg}",
            f"+root_base={str(ROOT_BASE)}",
        ]

        print("Command:", " ".join(f'"{x}"' if " " in x else x for x in cmd))
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            failed.append(cfg)
            print(f"\nFAILED: {cfg}\n")
        else:
            print(f"\nDONE: {cfg}\n")

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print("Total :", len(configs))
    print("Passed:", len(configs) - len(failed))
    print("Failed:", len(failed))

    if failed:
        print("\nFailed configs:")
        for cfg in failed:
            print("  -", cfg)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())