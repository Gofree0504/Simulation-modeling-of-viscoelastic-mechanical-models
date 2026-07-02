# Author: Ruan Yueheng
# Date: 2026-07-02
# Copyright (c) 2026 Ruan Yueheng.
# All rights reserved. For academic/research use, please cite or retain this notice.

from pathlib import Path
import re
import pandas as pd

RHEOLOGY_DIR = Path("raw_rheology")
STRUCTURE_FILE = Path("raw_structure/structure_metrics.xlsx")

OUTPUT_RHEOLOGY = Path("real_rheology_long.xlsx")
OUTPUT_STRUCTURE = Path("real_structure_summary.xlsx")
OUTPUT_MERGED = Path("real_model_input_summary.xlsx")

TIME_ALIASES = {
    "time",
    "time_s",
    "times",
    "t",
    "t_s",
    "Time",
    "Time_s",
    "Time (s)",
    "Time(s)",
    "Time [s]",
    "时间",
    "时间/s",
    "时间(s)",
    "时间（s）",
    "时间 [s]",
}

STRESS_ALIASES = {
    "stress",
    "stress_pa",
    "Stress",
    "Stress_Pa",
    "Stress (Pa)",
    "Stress(Pa)",
    "Stress [Pa]",
    "sigma",
    "sigma_pa",
    "Shear Stress",
    "Shear stress",
    "Shear Stress (Pa)",
    "Shear Stress(Pa)",
    "Shear Stress [Pa]",
    "应力",
    "应力/Pa",
    "应力(Pa)",
    "应力（Pa）",
    "应力 [Pa]",
    "剪切应力",
    "剪切应力/Pa",
    "剪切应力(Pa)",
    "剪切应力（Pa）",
    "剪切应力 [Pa]",
    "模量",
    "松弛模量",
    "Relaxation modulus",
    "G",
    "G(t)",
    "G_t",
}

STRESS_NORM_ALIASES = {
    "stress_norm",
    "normalized_stress",
    "norm_stress",
    "Stress_norm",
    "Normalized stress",
    "Normalized Stress",
    "Stress normalized",
    "stress/stress0",
    "stress / stress0",
    "sigma/sigma0",
    "sigma / sigma0",
    "归一化应力",
    "标准化应力",
    "归一化剪切应力",
    "归一化模量",
    "normalized modulus",
    "Normalized modulus",
}

def normalize_text(text):
    text = str(text).strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text

def canonical_sample_id(group, replicate):
    prefix = {
        "normal": "N",
        "mild": "MI",
        "severe": "SI",
    }[group]

    return f"{prefix}-{replicate}"

def infer_group_and_replicate(file_stem):
    """
    Supported file name examples:
    Normal-1
    N-1
    Mild Injury-1
    MI-1
    Severe Injury-1
    SI-1
    """

    name = normalize_text(file_stem)
    lower = name.lower()

    # Identify replicate.
    numbers = re.findall(r"\d+", lower)
    if not numbers:
        raise ValueError(f"Cannot infer replicate number from file name: {file_stem}")

    replicate = int(numbers[-1])

    # Identify group.
    compact = lower.replace(" ", "").replace("_", "").replace("-", "")

    if compact.startswith("n") or "normal" in compact:
        group = "normal"
    elif compact.startswith("mi") or "mildinjury" in compact or "mild" in compact:
        group = "mild"
    elif compact.startswith("si") or "severeinjury" in compact or "severe" in compact:
        group = "severe"
    else:
        raise ValueError(f"Cannot infer group from file name: {file_stem}")

    sample_id = canonical_sample_id(group, replicate)

    return group, replicate, sample_id

def normalize_column_name(text):
    text = str(text).strip()
    text = text.replace("_", "")
    text = text.replace("-", "")
    text = text.replace("/", "")
    text = text.replace("\\", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("（", "")
    text = text.replace("）", "")
    text = text.replace("[", "")
    text = text.replace("]", "")
    text = text.replace(" ", "")
    text = text.lower()
    return text

def find_column(columns, aliases):
    normalized_columns = {
        normalize_column_name(c): c
        for c in columns
    }

    normalized_aliases = [
        normalize_column_name(a)
        for a in aliases
    ]

    # Exact match.
    for alias in normalized_aliases:
        if alias in normalized_columns:
            return normalized_columns[alias]

    # Partial match.
    for col_norm, original_col in normalized_columns.items():
        for alias in normalized_aliases:
            if alias and alias in col_norm:
                return original_col

    return None

def read_table(path):
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(path, engine="openpyxl")
        except ImportError as exc:
            raise ImportError(
                "\nReading .xlsx files requires openpyxl.\n"
                "Please install it with:\n\n"
                "    python -m pip install openpyxl\n\n"
                "If the network is slow, you may use a mirror source, for example:\n\n"
                "    python -m pip install openpyxl "
                "-i https://pypi.tuna.tsinghua.edu.cn/simple\n"
            ) from exc

    raise ValueError(f"Unsupported file type: {path}")

def read_one_rheology_file(path):
    group, replicate, sample_id = infer_group_and_replicate(path.stem)

    df = read_table(path)

    time_col = find_column(df.columns, TIME_ALIASES)
    stress_col = find_column(df.columns, STRESS_ALIASES)
    stress_norm_col = find_column(df.columns, STRESS_NORM_ALIASES)

    if time_col is None:
        raise ValueError(
            f"No time column found in {path}\n"
            f"Available columns are:\n{list(df.columns)}"
        )

    if stress_col is None and stress_norm_col is None:
        raise ValueError(
            f"No stress or normalized stress column found in {path}\n"
            f"Available columns are:\n{list(df.columns)}"
        )

    out = pd.DataFrame(index=df.index)

    out["time_s"] = pd.to_numeric(df[time_col], errors="coerce")

    if stress_col is not None:
        out["stress_pa"] = pd.to_numeric(df[stress_col], errors="coerce")
    else:
        out["stress_pa"] = pd.NA

    if stress_norm_col is not None:
        out["stress_norm"] = pd.to_numeric(df[stress_norm_col], errors="coerce")
    else:
        stress = pd.to_numeric(df[stress_col], errors="coerce")
        valid_stress = stress.dropna()

        if valid_stress.empty:
            raise ValueError(f"No valid stress values found in {path}")

        first_valid = valid_stress.iloc[0]
        out["stress_norm"] = stress / first_valid

    out["sample_id"] = sample_id
    out["group"] = group
    out["replicate"] = replicate
    out["source_file"] = path.name

    out = out[
        [
            "sample_id",
            "group",
            "replicate",
            "time_s",
            "stress_pa",
            "stress_norm",
            "source_file",
        ]
    ]

    out = out.dropna(subset=["time_s", "stress_norm"])
    out = out.sort_values("time_s").reset_index(drop=True)

    return out

def collect_rheology_files(rheology_dir):
    files = []

    for ext in ["*.csv", "*.xlsx", "*.xls"]:
        files.extend(
            path for path in rheology_dir.glob(ext)
            if not path.name.startswith("~$")
        )

    files = sorted(files)

    if not files:
        raise FileNotFoundError(f"No rheology files found in {rheology_dir}")

    all_rows = []

    for path in files:
        print(f"Reading rheology file: {path}")
        one = read_one_rheology_file(path)
        all_rows.append(one)

    return pd.concat(all_rows, ignore_index=True)

def standardize_structure_file(path):
    """
    Read the real structure parameter file.

    The structure_metrics.xlsx file may contain only:
        sample_id, group, replicate, tortuosity, coherency

    If the following columns are missing:
        total_fiber_length_um
        mean_fiber_length_um
        fiber_segment_number
        abnormal_contact_number

    they will be filled automatically using the previous virtual fiber model
    parameters.
    """

    if not path.exists():
        print(f"Structure file not found: {path}")
        print("Only rheology long table will be generated.")
        return None

    df = read_table(path)

    # Strip leading and trailing spaces from column names.
    df.columns = [str(c).strip() for c in df.columns]

    # Try to identify required columns automatically.
    column_aliases = {
        "sample_id": ["sample_id", "sample", "Sample", "Sample ID", "样本", "样本编号"],
        "group": ["group", "Group", "组别", "分组"],
        "replicate": ["replicate", "Replicate", "rep", "重复", "编号"],
        "tortuosity": ["tortuosity", "Tortuosity", "曲折度"],
        "coherency": ["coherency", "Coherency", "orientation", "有序度", "取向有序度"],
    }

    rename_map = {}

    for target_col, aliases in column_aliases.items():
        found = find_column(df.columns, aliases)
        if found is not None:
            rename_map[found] = target_col

    df = df.rename(columns=rename_map)

    # If group/replicate are missing but sample_id exists, infer them from sample_id.
    if "sample_id" not in df.columns:
        raise ValueError(
            "structure_metrics.xlsx must contain at least sample_id, "
            "for example N-1, MI-1, or SI-1."
        )

    if "group" not in df.columns or "replicate" not in df.columns:
        inferred_groups = []
        inferred_reps = []
        inferred_ids = []

        for sid in df["sample_id"]:
            group, replicate, canonical_id = infer_group_and_replicate(str(sid))
            inferred_groups.append(group)
            inferred_reps.append(replicate)
            inferred_ids.append(canonical_id)

        df["group"] = inferred_groups
        df["replicate"] = inferred_reps
        df["sample_id"] = inferred_ids
    else:
        df["group"] = df["group"].astype(str).str.lower().str.strip()
        df["group"] = df["group"].replace({
            "normal": "normal",
            "n": "normal",
            "正常": "normal",
            "mild": "mild",
            "mild injury": "mild",
            "mi": "mild",
            "轻度": "mild",
            "轻度损伤": "mild",
            "severe": "severe",
            "severe injury": "severe",
            "si": "severe",
            "重度": "severe",
            "重度损伤": "severe",
        })

        df["replicate"] = df["replicate"].astype(int)
        df["sample_id"] = [
            canonical_sample_id(g, r)
            for g, r in zip(df["group"], df["replicate"])
        ]

    # Check required real structure columns.
    required_real_columns = [
        "sample_id",
        "group",
        "replicate",
        "tortuosity",
        "coherency",
    ]

    missing_real = [c for c in required_real_columns if c not in df.columns]

    if missing_real:
        raise ValueError(
            "structure_metrics.xlsx is missing required columns: "
            + ", ".join(missing_real)
            + "\nAt minimum, sample_id, tortuosity, and coherency are required."
        )

    # Convert real structure parameters to numeric values.
    df["tortuosity"] = pd.to_numeric(df["tortuosity"], errors="coerce")
    df["coherency"] = pd.to_numeric(df["coherency"], errors="coerce")

    # Fill missing structure constraints using the previous virtual model.
    virtual_defaults = {
        "normal": {
            "total_fiber_length_um": 1100 * 5.0,
            "mean_fiber_length_um": 60.0,
            "fiber_segment_number": 1100,
            "abnormal_contact_number": 0,
        },
        "mild": {
            "total_fiber_length_um": 1100 * 5.0,
            "mean_fiber_length_um": 48.0,
            "fiber_segment_number": 1100,
            "abnormal_contact_number": 60,
        },
        "severe": {
            "total_fiber_length_um": 2500 * 5.0,
            "mean_fiber_length_um": 28.0,
            "fiber_segment_number": 2500,
            "abnormal_contact_number": 450,
        },
    }

    optional_columns = [
        "total_fiber_length_um",
        "mean_fiber_length_um",
        "fiber_segment_number",
        "abnormal_contact_number",
    ]

    for col in optional_columns:
        if col not in df.columns:
            df[col] = [
                virtual_defaults[group][col]
                for group in df["group"]
            ]
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

            missing_mask = df[col].isna()

            if missing_mask.any():
                df.loc[missing