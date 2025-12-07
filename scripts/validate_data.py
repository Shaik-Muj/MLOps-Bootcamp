import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/raw/sample.csv")
REPORT_PATH = Path("data/validation_report.json")
GE_DIR = Path("great_expectations")
GE_YML = GE_DIR / "great_expectations.yml"

checks = []
success = True


def add_result(name, ok, details=None):
    checks.append(
        {"check": name, "success": bool(ok), "details": safe_convert(details or {})}
    )
    return ok


def safe_convert(obj):
    """
    Convert numpy/pandas scalars and arrays into native Python types so json.dumps() works.
    """
    if obj is None:
        return None
    # numpy scalar
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    # numpy bool or int/float subclasses
    if isinstance(obj, (np.generic,)):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    # numpy arrays -> lists
    if isinstance(obj, (np.ndarray,)):
        return [safe_convert(x) for x in obj.tolist()]
    # pandas types that may be numpy-based
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    # dict -> convert recursively
    if isinstance(obj, dict):
        return {str(k): safe_convert(v) for k, v in obj.items()}
    # list/tuple -> convert elements
    if isinstance(obj, (list, tuple)):
        return [safe_convert(x) for x in obj]
    # default: try to return as-is if JSON can handle it, else string
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)


def main():
    global success
    if not DATA_PATH.exists():
        print(f"[ERROR] Data file not found: {DATA_PATH}")
        add_result("file_exists", False, {"path": str(DATA_PATH)})
        success = False
    else:
        df = pd.read_csv(DATA_PATH)
        # 1) Column names
        expected_cols = ["feature1", "feature2", "target"]
        cols_ok = list(df.columns) == expected_cols
        add_result(
            "column_names_match",
            cols_ok,
            {"expected": expected_cols, "found": list(df.columns)},
        )
        success = success and cols_ok

        # 2) No nulls in target
        if "target" in df.columns:
            nulls = int(df["target"].isnull().sum())
            ok = nulls == 0
            add_result("target_no_nulls", ok, {"null_count": nulls})
            success = success and ok

        # 3) target values allowed
        if "target" in df.columns:
            # convert unique values to native ints (if possible)
            try:
                vals = set(int(x) for x in pd.unique(df["target"]))
            except Exception:
                vals = set(pd.unique(df["target"]).tolist())
            allowed = set([0, 1])
            ok = vals.issubset(allowed)
            add_result(
                "target_allowed_values",
                ok,
                {"found_values": list(vals), "allowed": list(allowed)},
            )
            success = success and ok

        # 4) Basic types for features (are numeric)
        for col in ["feature1", "feature2"]:
            if col in df.columns:
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                add_result(
                    f"{col}_is_numeric", is_numeric, {"dtype": str(df[col].dtype)}
                )
                success = success and is_numeric

        # 5) Row count sanity check (>2)
        row_count = len(df)
        ok = row_count >= 3
        add_result(
            "row_count_minimum", ok, {"row_count": int(row_count), "min_required": 3}
        )
        success = success and ok

        # 6) Simple distribution check (IQR positive)
        if "feature1" in df.columns:
            p25 = df["feature1"].quantile(0.25)
            p75 = df["feature1"].quantile(0.75)
            add_result(
                "feature1_iqr_positive",
                p75 > p25,
                {"p25": safe_convert(p25), "p75": safe_convert(p75)},
            )
            success = success and (p75 > p25)

    # Create a minimal great_expectations folder scaffold
    try:
        GE_DIR.mkdir(exist_ok=True)
        if not GE_YML.exists():
            GE_YML.write_text(
                "config_version: 3\n"
                "datasources: {}\n"
                "stores: {}\n"
                "expectations_store_name: expectations_store\n"
                "validations_store_name: validations_store\n"
                "evaluation_parameter_store_name: evaluation_parameter_store\n"
                "data_docs_sites: {}\n"
                "anonymous_usage_statistics:\n  enabled: False\n"
            )
            add_result("ge_scaffold_created", True, {"path": str(GE_YML)})
    except Exception as e:
        add_result("ge_scaffold_created", False, {"error": str(e)})
        success = False

    # Write report (with safe conversion)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {"success": success, "checks": checks}
    # Apply safe_convert to entire report to handle numpy types
    report_safe = safe_convert(report)
    REPORT_PATH.write_text(json.dumps(report_safe, indent=2))
    print(json.dumps(report_safe, indent=2))

    if success:
        print("[OK] Validation passed.")
        sys.exit(0)
    else:
        print("[FAIL] Validation failed. See", REPORT_PATH)
        sys.exit(1)


if __name__ == "__main__":
    main()
