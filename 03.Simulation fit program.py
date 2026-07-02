# Author: Ruan Yueheng
# Date: 2026-07-02
# Copyright (c) 2026 Ruan Yueheng.
# All rights reserved. For academic/research use, please cite or retain this notice.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# 0. Paths and global parameters
# ============================================================
INPUT_FILE = Path("real_data_simulation_results/real_sample_model_summary.csv")

OUTPUT_DIR = Path("real_data_simulation_results/structure_tau_analysis")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

GROUP_COLORS = {
    "normal": "#1F77B4",   # Blue
    "mild": "#2CA02C",     # Green
    "severe": "#D62728",   # Red
}

GROUP_LABELS = {
    "normal": "Normal",
    "mild": "Mild damage",
    "severe": "Severe damage",
}

TAU_TARGETS = {
    "real_tau_s": "Experimental fitted tau / s",
    "model_tau_eff": "Model effective tau_eff / s",
    "model_tau_fit_s": "Model fitted tau / s",
}

# ============================================================
# 1. General figure saving function
# ============================================================
def save_figure_png_pdf(save_path, dpi=300):
    save_path = Path(save_path)

    png_path = save_path.with_suffix(".png")
    pdf_path = save_path.with_suffix(".pdf")

    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")

    return png_path, pdf_path

# ============================================================
# 2. Data loading and preprocessing
# ============================================================
def load_summary_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. "
            "Please run the main program first to generate real_sample_model_summary.csv."
        )

    df = pd.read_csv(INPUT_FILE)

    required_cols = [
        "sample_id",
        "group",
        "structure_tortuosity",
        "structure_coherency",
        "real_tau_s",
        "model_tau_eff",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"{INPUT_FILE} is missing required columns: {missing}")

    numeric_cols = [
        "structure_tortuosity",
        "structure_coherency",
        "real_tau_s",
        "model_tau_eff",
    ]

    if "model_tau_fit_s" in df.columns:
        numeric_cols.append("model_tau_fit_s")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["group"] = df["group"].astype(str)
    df["sample_id"] = df["sample_id"].astype(str)

    df = df.dropna(
        subset=[
            "structure_tortuosity",
            "structure_coherency",
            "real_tau_s",
            "model_tau_eff",
        ]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid data remain. Please check the structure parameters and tau "
            "parameters in the summary file."
        )

    return df

# ============================================================
# 3. Univariate linear relationship analysis
# ============================================================
def analyze_linear_relation(df, x_col, y_col):
    data = df[[x_col, y_col]].dropna().copy()

    if len(data) < 3:
        return {
            "x": x_col,
            "y": y_col,
            "n": len(data),
            "slope": np.nan,
            "intercept": np.nan,
            "r": np.nan,
            "r2": np.nan,
            "p_value": np.nan,
            "std_err": np.nan,
        }

    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)

    result = stats.linregress(x, y)

    return {
        "x": x_col,
        "y": y_col,
        "n": len(data),
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r": float(result.rvalue),
        "r2": float(result.rvalue ** 2),
        "p_value": float(result.pvalue),
        "std_err": float(result.stderr),
    }

def plot_linear_relation(df, x_col, y_col, x_label, y_label, save_path):
    data = df[[x_col, y_col, "group"]].dropna().copy()

    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)

    fit = analyze_linear_relation(data, x_col, y_col)

    plt.figure(figsize=(5.6, 4.6), dpi=220)

    for group_name in ["normal", "mild", "severe"]:
        group_data = data[data["group"] == group_name]

        if group_data.empty:
            continue

        plt.scatter(
            group_data[x_col],
            group_data[y_col],
            s=42,
            color=GROUP_COLORS.get(group_name, "gray"),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            label=GROUP_LABELS.get(group_name, group_name),
        )

    if len(data) >= 3 and np.isfinite(fit["slope"]):
        x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        y_fit = fit["slope"] * x_fit + fit["intercept"]

        plt.plot(
            x_fit,
            y_fit,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=(
                f"Linear fit: R²={fit['r2']:.3f}, "
                f"p={fit['p_value']:.3g}"
            ),
        )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()

    save_figure_png_pdf(save_path, dpi=300)
    plt.close()

    return fit

# ============================================================
# 4. Bivariate 3D relationship analysis
# ============================================================
def analyze_multiple_linear_relation(df, y_col):
    required_cols = [
        "structure_tortuosity",
        "structure_coherency",
        y_col,
    ]

    data = df[required_cols].dropna().copy()

    if len(data) < 4:
        return {
            "target": y_col,
            "n": len(data),
            "intercept": np.nan,
            "coef_tortuosity": np.nan,
            "coef_coherency": np.nan,
            "r2": np.nan,
        }, None

    X = data[["structure_tortuosity", "structure_coherency"]].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    result = {
        "target": y_col,
        "n": len(data),
        "intercept": float(model.intercept_),
        "coef_tortuosity": float(model.coef_[0]),
        "coef_coherency": float(model.coef_[1]),
        "r2": float(r2),
    }

    return result, model

def plot_3d_relation(df, y_col, y_label, save_path):
    data = df[
        [
            "structure_tortuosity",
            "structure_coherency",
            y_col,
            "group",
        ]
    ].dropna().copy()

    result, model = analyze_multiple_linear_relation(df, y_col)

    fig = plt.figure(figsize=(6.2, 5.2), dpi=220)
    ax = fig.add_subplot(111, projection="3d")

    for group_name in ["normal", "mild", "severe"]:
        group_data = data[data["group"] == group_name]

        if group_data.empty:
            continue

        ax.scatter(
            group_data["structure_tortuosity"],
            group_data["structure_coherency"],
            group_data[y_col],
            s=45,
            color=GROUP_COLORS.get(group_name, "gray"),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
            label=GROUP_LABELS.get(group_name, group_name),
        )

    if model is not None and len(data) >= 4:
        tortuosity_min = data["structure_tortuosity"].min()
        tortuosity_max = data["structure_tortuosity"].max()
        coherency_min = data["structure_coherency"].min()
        coherency_max = data["structure_coherency"].max()

        tortuosity_grid = np.linspace(tortuosity_min, tortuosity_max, 25)
        coherency_grid = np.linspace(coherency_min, coherency_max, 25)

        T_grid, C_grid = np.meshgrid(tortuosity_grid, coherency_grid)

        X_grid = np.column_stack([
            T_grid.ravel(),
            C_grid.ravel(),
        ])

        Z_grid = model.predict(X_grid).reshape(T_grid.shape)

        ax.plot_surface(
            T_grid,
            C_grid,
            Z_grid,
            color="#B0B0B0",
            alpha=0.35,
            linewidth=0,
            antialiased=True,
        )

        ax.text2D(
            0.03,
            0.95,
            f"Multiple linear fit: R²={result['r2']:.3f}",
            transform=ax.transAxes,
            fontsize=9,
        )

    ax.set_xlabel("Tortuosity")
    ax.set_ylabel("Coherency")
    ax.set_zlabel(y_label)

    ax.view_init(elev=24, azim=-58)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    plt.tight_layout()
    save_figure_png_pdf(save_path, dpi=300)
    plt.close()

    return result

# ============================================================
# 5. Main workflow
# ============================================================
def run_structure_tau_analysis():
    df = load_summary_data()

    available_tau_targets = {
        key: value
        for key, value in TAU_TARGETS.items()
        if key in df.columns
    }

    linear_rows = []
    multiple_rows = []

    # --------------------------------------------------------
    # 5.1 Univariate linear relationships between tortuosity/coherency and tau
    # --------------------------------------------------------
    for y_col, y_label in available_tau_targets.items():
        fit_tortuosity = plot_linear_relation(
            df=df,
            x_col="structure_tortuosity",
            y_col=y_col,
            x_label="Tortuosity",
            y_label=y_label,
            save_path=OUTPUT_DIR / f"tortuosity_vs_{y_col}.png",
        )

        linear_rows.append(fit_tortuosity)

        fit_coherency = plot_linear_relation(
            df=df,
            x_col="structure_coherency",
            y_col=y_col,
            x_label="Coherency",
            y_label=y_label,
            save_path=OUTPUT_DIR / f"coherency_vs_{y_col}.png",
        )

        linear_rows.append(fit_coherency)

    linear_summary = pd.DataFrame(linear_rows)
    linear_summary_file = OUTPUT_DIR / "linear_relationship_summary.csv"
    linear_summary.to_csv(linear_summary_file, index=False)

    # --------------------------------------------------------
    # 5.2 3D relationships among tortuosity, coherency, and tau
    # --------------------------------------------------------
    for y_col, y_label in available_tau_targets.items():
        multiple_fit = plot_3d_relation(
            df=df,
            y_col=y_col,
            y_label=y_label,
            save_path=OUTPUT_DIR / f"3d_tortuosity_coherency_{y_col}.png",
        )

        multiple_rows.append(multiple_fit)

    multiple_summary = pd.DataFrame(multiple_rows)
    multiple_summary_file = OUTPUT_DIR / "multiple_linear_3d_summary.csv"
    multiple_summary.to_csv(multiple_summary_file, index=False)

    # --------------------------------------------------------
    # 5.3 Export cleaned data used for plotting
    # --------------------------------------------------------
    cleaned_data_file = OUTPUT_DIR / "structure_tau_cleaned_data.csv"
    df.to_csv(cleaned_data_file, index=False)

    print("\n======================================")
    print("Structure-tau relationship analysis finished")
    print("======================================")

    print("\nSaved files:")
    print(cleaned_data_file)
    print(linear_summary_file)
    print(multiple_summary_file)

    print("\nLinear relationship summary:")
    print(linear_summary)

    print("\nMultiple linear 3D summary:")
    print(multiple_summary)

    return df, linear_summary, multiple_summary

if __name__ == "__main__":
    run_structure_tau_analysis()