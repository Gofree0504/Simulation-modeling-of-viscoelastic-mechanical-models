# Author: Ruan Yueheng
# Date: 2026-07-02
# Copyright (c) 2026 Ruan Yueheng.
# All rights reserved. For academic/research use, please cite or retain this notice.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# ============================================================
# 0. File paths and global parameters
# ============================================================
REAL_RHEOLOGY_FILE = Path("real_rheology_long.xlsx")
REAL_STRUCTURE_FILE = Path("real_structure_summary.xlsx")
REAL_MODEL_INPUT_FILE = Path("real_model_input_summary.xlsx")

OUTPUT_DIR = Path("real_data_simulation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

SHEAR_STRAIN = 0.05
RELAXATION_YLIM = (0.0, 1.2)

def save_figure_png_pdf(save_path, dpi=300):
    """
    Save the current figure in both PNG and PDF formats.

    The input save_path can be .png or .pdf.
    The function automatically generates .png and .pdf files with the same name.
    """

    save_path = Path(save_path)

    png_path = save_path.with_suffix(".png")
    pdf_path = save_path.with_suffix(".pdf")

    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")

    return png_path, pdf_path

# ============================================================
# 0.1 General table reader
# ============================================================
def read_table(path):
    """
    Automatically read csv / xlsx / xls tables.

    This also handles a common case:
    the file suffix is .xlsx, but the actual content may be CSV text.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Cannot find file: {path}")

    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception as excel_error:
            try:
                return pd.read_csv(path)
            except Exception as csv_error:
                raise ValueError(
                    f"Cannot read {path} as Excel or CSV.\n"
                    f"Excel error: {excel_error}\n"
                    f"CSV error: {csv_error}"
                )

    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="gbk")

    raise ValueError(f"Unsupported file format: {path.suffix}")

# ============================================================
# 1. Real data loading
# ============================================================
def load_real_rheology():
    if not REAL_RHEOLOGY_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {REAL_RHEOLOGY_FILE}. "
            "Please run the data preparation script first."
        )

    df = read_table(REAL_RHEOLOGY_FILE)

    required_cols = [
        "sample_id",
        "group",
        "replicate",
        "time_s",
        "stress_norm",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"{REAL_RHEOLOGY_FILE} is missing required columns: {missing}"
        )

    df["sample_id"] = df["sample_id"].astype(str)
    df["group"] = df["group"].astype(str)
    df["replicate"] = pd.to_numeric(df["replicate"], errors="coerce")
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["stress_norm"] = pd.to_numeric(df["stress_norm"], errors="coerce")

    if "stress_pa" in df.columns:
        df["stress_pa"] = pd.to_numeric(df["stress_pa"], errors="coerce")

    df = df.dropna(
        subset=[
            "sample_id",
            "group",
            "replicate",
            "time_s",
            "stress_norm",
        ]
    )

    df = df.sort_values(["group", "replicate", "time_s"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("The real rheology data is empty. Please check real_rheology_long.")

    return df

def load_real_structure():
    if REAL_STRUCTURE_FILE.exists():
        df = read_table(REAL_STRUCTURE_FILE)
    elif REAL_MODEL_INPUT_FILE.exists():
        df = read_table(REAL_MODEL_INPUT_FILE)
    else:
        print("No real structure file found. Will use default structure parameters.")
        return None

    if "sample_id" not in df.columns:
        print("Structure file has no sample_id. Will use default structure parameters.")
        return None

    df["sample_id"] = df["sample_id"].astype(str)

    return df

def get_structure_for_sample(structure_df, sample_id, group_name):
    """
    Read real structure parameters.

    If real structure parameters are missing, use the default structure values
    from the original simulation program for the corresponding group.
    """

    default_params = {
        "normal": {
            "tortuosity": 1.15,
            "coherency": 0.90,
            "total_fiber_length_um": 5500.0,
            "mean_fiber_length_um": 60.0,
            "fiber_segment_number": 1100.0,
            "abnormal_contact_number": 0.0,
        },
        "mild": {
            "tortuosity": 1.25,
            "coherency": 0.75,
            "total_fiber_length_um": 5500.0,
            "mean_fiber_length_um": 48.0,
            "fiber_segment_number": 1100.0,
            "abnormal_contact_number": 60.0,
        },
        "severe": {
            "tortuosity": 1.40,
            "coherency": 0.30,
            "total_fiber_length_um": 12500.0,
            "mean_fiber_length_um": 28.0,
            "fiber_segment_number": 2500.0,
            "abnormal_contact_number": 450.0,
        },
    }

    if group_name not in default_params:
        raise ValueError("group_name must be normal, mild, or severe")

    params = default_params[group_name].copy()

    if structure_df is None:
        return params

    rows = structure_df[structure_df["sample_id"] == sample_id]

    if rows.empty:
        return params

    row = rows.iloc[0]

    aliases = {
        "tortuosity": ["tortuosity", "tortuosity_mean", "T_mean"],
        "coherency": ["coherency", "Coherency"],
        "total_fiber_length_um": ["total_fiber_length_um"],
        "mean_fiber_length_um": ["mean_fiber_length_um", "mean_chain_length_um"],
        "fiber_segment_number": ["fiber_segment_number"],
        "abnormal_contact_number": ["abnormal_contact_number"],
    }

    for target_col, possible_cols in aliases.items():
        for col in possible_cols:
            if col in row.index and pd.notna(row[col]):
                params[target_col] = float(row[col])
                break

    return params

# ============================================================
# 2. Relaxation model and fitting functions
# ============================================================
def stretched_exp_model(t, y_inf, tau, beta):
    """
    Normalized stress relaxation fitting model:
    y(t) = y_inf + (1 - y_inf) exp[-(t / tau)^beta]

    Note: This function is used to extract apparent relaxation time.
    It is not the main physical model.
    """

    t = np.asarray(t, dtype=float)
    tau = max(float(tau), 1e-12)

    return y_inf + (1.0 - y_inf) * np.exp(-((t / tau) ** beta))

def fit_relaxation_time(time, stress_norm):
    """
    Input a real or simulated normalized stress curve and return the apparent
    relaxation time tau.
    """

    time = np.asarray(time, dtype=float)
    y = np.asarray(stress_norm, dtype=float)

    mask = np.isfinite(time) & np.isfinite(y) & (time > 0)

    time = time[mask]
    y = y[mask]

    if len(time) < 5:
        raise ValueError("Not enough valid points for relaxation fitting.")

    if abs(y[0]) <= 1e-12:
        raise ValueError("Initial stress is zero, cannot normalize.")

    y = y / y[0]

    p0 = [max(min(y[-1], 0.95), 0.01), 20.0, 0.8]
    bounds = ([0.0, 0.05, 0.2], [1.2, 300.0, 2.0])

    popt, pcov = curve_fit(
        stretched_exp_model,
        time,
        y,
        p0=p0,
        bounds=bounds,
        maxfev=30000,
    )

    y_inf, tau, beta = popt

    fitted = stretched_exp_model(time, y_inf, tau, beta)
    residual = y - fitted

    return {
        "time": time,
        "stress_norm": y,
        "y_inf": float(y_inf),
        "tau": float(tau),
        "beta": float(beta),
        "fitted": fitted,
        "residual": residual,
    }

# ============================================================
# 3. Mapping structure parameters to generalized Maxwell viscoelastic parameters
# ============================================================
def real_structure_to_viscoelastic_params(structure_params, group_name):
    """
    Map real structure parameters to generalized Maxwell viscoelastic parameters.

    Main physical model:
        G(t) = G_inf + G1 exp(-t / tau1) + G2 exp(-t / tau2)
        sigma(t) = gamma0 * G(t)

    The current G_inf, G1, and G2 are relative moduli. If real stress_pa data are
    used for fitting, they can be further converted to Pa units.
    """

    tortuosity = float(structure_params["tortuosity"])
    coherency = float(structure_params["coherency"])
    abnormal_contact_number = float(structure_params["abnormal_contact_number"])
    fiber_segment_number = float(structure_params["fiber_segment_number"])

    if group_name == "normal":
        G_inf = 0.62
        G1 = 0.25
        tau1 = 18.0
        G2 = 0.18
        tau2 = 85.0

    elif group_name == "mild":
        G_inf = 0.42
        G1 = 0.38
        tau1 = 8.0
        G2 = 0.22
        tau2 = 35.0

    elif group_name == "severe":
        G_inf = 0.22
        G1 = 0.55
        tau1 = 2.5
        G2 = 0.18
        tau2 = 11.0

    else:
        raise ValueError("group_name must be normal, mild, or severe")

    tortuosity_factor = np.clip(tortuosity / 1.20, 0.60, 1.80)
    coherency_factor = np.clip(coherency / 0.75, 0.30, 1.80)
    abnormal_factor = np.clip(1.0 + abnormal_contact_number / 500.0, 1.0, 2.5)
    segment_factor = np.clip(fiber_segment_number / 1100.0, 0.50, 3.00)

    # Empirical mapping logic:
    # higher tortuosity leads to faster relaxation; higher coherency leads to
    # slower relaxation; more abnormal contacts strengthen the fast relaxation
    # component.
    tau1 = tau1 / tortuosity_factor * coherency_factor / abnormal_factor
    tau2 = tau2 / tortuosity_factor * coherency_factor / np.sqrt(abnormal_factor)

    G1 = G1 * abnormal_factor
    G2 = G2 * np.sqrt(segment_factor)
    G_inf = G_inf * coherency_factor / np.sqrt(tortuosity_factor)

    # Derived generalized Maxwell parameters.
    G0 = G_inf + G1 + G2
    G_relaxing = G1 + G2

    eta1 = G1 * tau1
    eta2 = G2 * tau2
    eta_eff = eta1 + eta2

    if G_relaxing > 0:
        tau_eff = eta_eff / G_relaxing
    else:
        tau_eff = np.nan

    return {
        "G_inf": float(G_inf),
        "G1": float(G1),
        "tau1": float(tau1),
        "eta1": float(eta1),
        "G2": float(G2),
        "tau2": float(tau2),
        "eta2": float(eta2),
        "G0": float(G0),
        "G_relaxing": float(G_relaxing),
        "eta_eff": float(eta_eff),
        "tau_eff": float(tau_eff),
        "tortuosity_factor": float(tortuosity_factor),
        "coherency_factor": float(coherency_factor),
        "abnormal_factor": float(abnormal_factor),
        "segment_factor": float(segment_factor),
    }

def simulate_relaxation_from_real_structure(
    time,
    structure_params,
    group_name,
    shear_strain=0.05,
):
    """
    Generate the corresponding normalized relaxation curve using real structure
    parameters.
    """

    params = real_structure_to_viscoelastic_params(
        structure_params=structure_params,
        group_name=group_name,
    )

    time = np.asarray(time, dtype=float)

    G_t = (
        params["G_inf"]
        + params["G1"] * np.exp(-time / params["tau1"])
        + params["G2"] * np.exp(-time / params["tau2"])
    )

    stress = shear_strain * G_t

    stress = np.maximum(stress, 1e-12)
    stress_norm = stress / stress[0]

    return stress_norm, params

# ============================================================
# 4. Parameters derived from experimental absolute stress
# ============================================================
def calculate_experimental_modulus_params(one_sample_df, shear_strain):
    """
    If stress_pa exists, estimate apparent shear modulus parameters from
    experimental stress relaxation data.

    G_exp(t) = sigma_exp(t) / gamma0

    eta_app_pa_s is a rough apparent value:
        eta_app = G_relax_app * tau_fit
    where tau_fit is obtained from stretched exponential fitting.
    """

    result = {
        "exp_G0_pa": np.nan,
        "exp_G_inf_last_pa": np.nan,
        "exp_G_relax_pa": np.nan,
        "exp_eta_app_pa_s": np.nan,
    }

    if "stress_pa" not in one_sample_df.columns:
        return result

    stress_pa = one_sample_df["stress_pa"].to_numpy(dtype=float)
    time = one_sample_df["time_s"].to_numpy(dtype=float)

    mask = np.isfinite(stress_pa) & np.isfinite(time)

    if np.count_nonzero(mask) < 5:
        return result

    stress_pa = stress_pa[mask]
    time = time[mask]

    if abs(shear_strain) <= 1e-12:
        return result

    G_exp_t = stress_pa / shear_strain

    if len(G_exp_t) == 0 or not np.isfinite(G_exp_t[0]):
        return result

    G0_exp = float(G_exp_t[0])
    G_inf_exp = float(G_exp_t[-1])
    G_relax_exp = float(G0_exp - G_inf_exp)

    try:
        stress_norm = stress_pa / stress_pa[0]
        fit_result = fit_relaxation_time(time, stress_norm)
        eta_app = G_relax_exp * fit_result["tau"]
    except Exception:
        eta_app = np.nan

    result.update({
        "exp_G0_pa": G0_exp,
        "exp_G_inf_last_pa": G_inf_exp,
        "exp_G_relax_pa": G_relax_exp,
        "exp_eta_app_pa_s": float(eta_app) if np.isfinite(eta_app) else np.nan,
    })

    return result

# ============================================================
# 5. Error metrics
# ============================================================
def calculate_fit_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
        }

    residual = y_true - y_pred

    rmse = np.sqrt(np.mean(residual ** 2))
    mae = np.mean(np.abs(residual))

    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot <= 1e-12:
        r2 = np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

# ============================================================
# 6. Single-sample plotting
# ============================================================
def plot_one_sample_real_vs_sim(
    sample_id,
    group_name,
    time,
    stress_exp,
    stress_sim,
    fit_exp,
    save_path,
):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(6.4, 5.4),
        dpi=220,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax = axes[0]
    ax_res = axes[1]

    ax.plot(
        time,
        stress_exp,
        "o",
        markersize=3,
        alpha=0.65,
        label="Real experiment",
    )

    ax.plot(
        fit_exp["time"],
        fit_exp["fitted"],
        "--",
        color="black",
        linewidth=1.5,
        label=f"Real fit, tau={fit_exp['tau']:.2f} s",
    )

    ax.plot(
        time,
        stress_sim,
        "-",
        linewidth=2.0,
        label="Structure-driven Maxwell model",
    )

    residual = stress_exp - stress_sim

    ax_res.plot(
        time,
        residual,
        color="gray",
        linewidth=1.2,
    )

    ax_res.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8,
    )

    ax.set_ylabel("Normalized stress")
    ax.set_ylim(RELAXATION_YLIM)
    ax.set_title(f"{sample_id} ({group_name})")
    ax.legend(frameon=False, fontsize=8)

    ax_res.set_xlabel("Time / s")
    ax_res.set_ylabel("Exp - Model")

    ax.set_xlim(float(np.nanmin(time)), float(np.nanmax(time)))

    plt.tight_layout()
    save_figure_png_pdf(save_path, dpi=300)
    plt.close()

# ============================================================
# 7. Group plotting: real experimental mean
# ============================================================
def plot_group_real_curves(group_name, group_df, save_path):
    colors = {
        "normal": "#1F77B4",  # Blue
        "mild": "#2CA02C",  # Green
        "severe": "#D62728",  # Red
    }

    color = colors.get(group_name, "black")

    sample_ids = sorted(group_df["sample_id"].unique())

    replicate_curves = []
    common_time = None

    plt.figure(figsize=(6.4, 4.6), dpi=220)

    for sample_id in sample_ids:
        one = group_df[group_df["sample_id"] == sample_id].sort_values("time_s")

        time = one["time_s"].to_numpy(dtype=float)
        stress = one["stress_norm"].to_numpy(dtype=float)

        if len(time) < 3:
            continue

        if abs(stress[0]) <= 1e-12:
            continue

        stress = stress / stress[0]

        plt.plot(
            time,
            stress,
            color=color,
            alpha=0.35,
            linewidth=1.2,
        )

        if common_time is None:
            common_time = time
            replicate_curves.append(stress)
        else:
            interp_stress = np.interp(common_time, time, stress)
            replicate_curves.append(interp_stress)

    if common_time is None or len(replicate_curves) == 0:
        return None

    replicate_curves = np.array(replicate_curves)

    mean_curve = np.mean(replicate_curves, axis=0)

    if replicate_curves.shape[0] > 1:
        std_curve = np.std(replicate_curves, axis=0, ddof=1)
    else:
        std_curve = np.zeros_like(mean_curve)

    fit_result = fit_relaxation_time(common_time, mean_curve)

    plt.plot(
        common_time,
        mean_curve,
        color=color,
        linewidth=2.5,
        label=f"{group_name}, mean",
    )

    plt.fill_between(
        common_time,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color=color,
        alpha=0.22,
        linewidth=0,
        label="+/- SD",
    )

    plt.plot(
        fit_result["time"],
        fit_result["fitted"],
        "--",
        color="black",
        linewidth=1.4,
        label=f"fit, tau={fit_result['tau']:.2f} s",
    )

    plt.xlabel("Time / s")
    plt.ylabel("Normalized stress")
    plt.ylim(RELAXATION_YLIM)
    plt.title(f"Real {group_name} stress relaxation")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    save_figure_png_pdf(save_path, dpi=300)
    plt.close()

    return {
        "group": group_name,
        "n_samples": len(sample_ids),
        "tau": fit_result["tau"],
        "beta": fit_result["beta"],
        "y_inf": fit_result["y_inf"],
    }

# ============================================================
# 8. Three-group overview plot
# ============================================================
def plot_all_real_groups(real_rheology, save_path):
    colors = {
        "normal": "#1F77B4",  # Blue
        "mild": "#2CA02C",  # Green
        "severe": "#D62728",  # Red
    }

    labels = {
        "normal": "Normal",
        "mild": "Mild damage",
        "severe": "Severe damage",
    }

    plt.figure(figsize=(6.4, 4.6), dpi=220)

    summary_rows = []

    for group_name in ["normal", "mild", "severe"]:
        group_df = real_rheology[real_rheology["group"] == group_name]

        if group_df.empty:
            continue

        sample_ids = sorted(group_df["sample_id"].unique())

        common_time = None
        replicate_curves = []

        for sample_id in sample_ids:
            one = group_df[group_df["sample_id"] == sample_id].sort_values("time_s")

            time = one["time_s"].to_numpy(dtype=float)
            stress = one["stress_norm"].to_numpy(dtype=float)

            if len(time) < 3:
                continue

            if abs(stress[0]) <= 1e-12:
                continue

            stress = stress / stress[0]

            if common_time is None:
                common_time = time
                replicate_curves.append(stress)
            else:
                replicate_curves.append(np.interp(common_time, time, stress))

        if common_time is None or len(replicate_curves) == 0:
            continue

        replicate_curves = np.array(replicate_curves)
        mean_curve = np.mean(replicate_curves, axis=0)

        if replicate_curves.shape[0] > 1:
            std_curve = np.std(replicate_curves, axis=0, ddof=1)
        else:
            std_curve = np.zeros_like(mean_curve)

        fit_result = fit_relaxation_time(common_time, mean_curve)

        color = colors[group_name]

        plt.plot(
            common_time,
            mean_curve,
            color=color,
            linewidth=2.5,
            label=f"{labels[group_name]}, tau={fit_result['tau']:.1f} s",
        )

        plt.fill_between(
            common_time,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.20,
            linewidth=0,
        )

        summary_rows.append({
            "group": group_name,
            "n_samples": len(sample_ids),
            "tau": fit_result["tau"],
            "beta": fit_result["beta"],
            "y_inf": fit_result["y_inf"],
        })

    plt.xlabel("Time / s")
    plt.ylabel("Normalized stress")
    plt.ylim(RELAXATION_YLIM)
    plt.title("Real experimental stress relaxation")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    save_figure_png_pdf(save_path, dpi=300)
    plt.close()

    return pd.DataFrame(summary_rows)

# ============================================================
# 9. Main workflow: feed real data back into the structure-driven viscoelastic model
# ============================================================
def run_real_data_simulation():
    real_rheology = load_real_rheology()
    real_structure = load_real_structure()

    sample_rows = []
    curve_rows = []

    sample_ids = sorted(real_rheology["sample_id"].unique())

    for sample_id in sample_ids:
        one = real_rheology[real_rheology["sample_id"] == sample_id].copy()
        one = one.sort_values("time_s").reset_index(drop=True)

        group_name = one["group"].iloc[0]
        replicate = int(one["replicate"].iloc[0])

        time = one["time_s"].to_numpy(dtype=float)
        stress_exp = one["stress_norm"].to_numpy(dtype=float)

        if len(time) < 5:
            print(f"Skip {sample_id}: too few time points.")
            continue

        if abs(stress_exp[0]) <= 1e-12:
            print(f"Skip {sample_id}: initial normalized stress is zero.")
            continue

        stress_exp = stress_exp / stress_exp[0]

        structure_params = get_structure_for_sample(
            structure_df=real_structure,
            sample_id=sample_id,
            group_name=group_name,
        )

        stress_sim, model_params = simulate_relaxation_from_real_structure(
            time=time,
            structure_params=structure_params,
            group_name=group_name,
            shear_strain=SHEAR_STRAIN,
        )

        fit_exp = fit_relaxation_time(time, stress_exp)
        fit_sim = fit_relaxation_time(time, stress_sim)

        metrics = calculate_fit_metrics(stress_exp, stress_sim)

        exp_modulus_params = calculate_experimental_modulus_params(
            one_sample_df=one,
            shear_strain=SHEAR_STRAIN,
        )

        plot_file = OUTPUT_DIR / f"{sample_id}_real_vs_model.png"
        plot_pdf_file = plot_file.with_suffix(".pdf")

        plot_one_sample_real_vs_sim(
            sample_id=sample_id,
            group_name=group_name,
            time=time,
            stress_exp=stress_exp,
            stress_sim=stress_sim,
            fit_exp=fit_exp,
            save_path=plot_file,
        )

        curve_file = OUTPUT_DIR / f"{sample_id}_curve_real_vs_model.csv"

        curve_dict = {
            "sample_id": sample_id,
            "group": group_name,
            "replicate": replicate,
            "time_s": time,
            "stress_norm_real": stress_exp,
            "stress_norm_model": stress_sim,
            "residual_real_minus_model": stress_exp - stress_sim,
        }

        if "stress_pa" in one.columns:
            stress_pa = one["stress_pa"].to_numpy(dtype=float)
            curve_dict["stress_pa_real"] = stress_pa
            curve_dict["G_exp_pa"] = stress_pa / SHEAR_STRAIN

        curve_df = pd.DataFrame(curve_dict)

        curve_df.to_csv(curve_file, index=False)

        for _, row in curve_df.iterrows():
            curve_rows.append(row.to_dict())

        result_row = {
            "sample_id": sample_id,
            "group": group_name,
            "replicate": replicate,
            "n_time_points": len(time),
            "real_tau_s": fit_exp["tau"],
            "real_beta": fit_exp["beta"],
            "real_y_inf": fit_exp["y_inf"],
            "model_tau_fit_s": fit_sim["tau"],
            "model_beta_fit": fit_sim["beta"],
            "model_y_inf_fit": fit_sim["y_inf"],
            "rmse_real_vs_model": metrics["rmse"],
            "mae_real_vs_model": metrics["mae"],
            "r2_real_vs_model": metrics["r2"],
            "plot_file": str(plot_file),
            "plot_pdf_file": str(plot_pdf_file),
            "curve_file": str(curve_file),
        }

        for key, value in structure_params.items():
            result_row[f"structure_{key}"] = value

        for key, value in model_params.items():
            result_row[f"model_{key}"] = value

        for key, value in exp_modulus_params.items():
            result_row[key] = value

        sample_rows.append(result_row)

        print(
            f"{sample_id}: "
            f"group={group_name}, "
            f"real tau={fit_exp['tau']:.2f} s, "
            f"model tau={fit_sim['tau']:.2f} s, "
            f"G0={model_params['G0']:.3f}, "
            f"eta_eff={model_params['eta_eff']:.3f}, "
            f"RMSE={metrics['rmse']:.4f}"
        )

    sample_summary = pd.DataFrame(sample_rows)
    all_curves = pd.DataFrame(curve_rows)

    sample_summary_file = OUTPUT_DIR / "real_sample_model_summary.csv"
    all_curves_file = OUTPUT_DIR / "real_all_curves_real_vs_model.csv"

    sample_summary.to_csv(sample_summary_file, index=False)
    all_curves.to_csv(all_curves_file, index=False)

    if not sample_summary.empty:
        group_summary = (
            sample_summary
            .groupby("group")
            .agg(
                n_samples=("sample_id", "count"),
                real_tau_mean_s=("real_tau_s", "mean"),
                real_tau_std_s=("real_tau_s", "std"),
                model_tau_fit_mean_s=("model_tau_fit_s", "mean"),
                model_tau_fit_std_s=("model_tau_fit_s", "std"),
                model_G0_mean=("model_G0", "mean"),
                model_G0_std=("model_G0", "std"),
                model_G_inf_mean=("model_G_inf", "mean"),
                model_eta_eff_mean=("model_eta_eff", "mean"),
                model_eta_eff_std=("model_eta_eff", "std"),
                model_tau_eff_mean_s=("model_tau_eff", "mean"),
                model_tau_eff_std_s=("model_tau_eff", "std"),
                rmse_mean=("rmse_real_vs_model", "mean"),
                rmse_std=("rmse_real_vs_model", "std"),
                mae_mean=("mae_real_vs_model", "mean"),
                r2_mean=("r2_real_vs_model", "mean"),
                structure_tortuosity_mean=("structure_tortuosity", "mean"),
                structure_tortuosity_std=("structure_tortuosity", "std"),
                structure_coherency_mean=("structure_coherency", "mean"),
                structure_coherency_std=("structure_coherency", "std"),
                structure_abnormal_contact_number_mean=(
                    "structure_abnormal_contact_number",
                    "mean",
                ),
                structure_fiber_segment_number_mean=(
                    "structure_fiber_segment_number",
                    "mean",
                ),
                exp_G0_pa_mean=("exp_G0_pa", "mean"),
                exp_G_inf_last_pa_mean=("exp_G_inf_last_pa", "mean"),
                exp_eta_app_pa_s_mean=("exp_eta_app_pa_s", "mean"),
            )
            .reset_index()
        )
    else:
        group_summary = pd.DataFrame()

    group_summary_file = OUTPUT_DIR / "real_group_model_summary.csv"
    group_summary.to_csv(group_summary_file, index=False)

    group_fit_rows = []

    for group_name in ["normal", "mild", "severe"]:
        group_df = real_rheology[real_rheology["group"] == group_name]

        if group_df.empty:
            continue

        group_plot_file = OUTPUT_DIR / f"{group_name}_real_relaxation.png"
        group_plot_pdf_file = group_plot_file.with_suffix(".pdf")

        group_fit = plot_group_real_curves(
            group_name=group_name,
            group_df=group_df,
            save_path=group_plot_file,
        )

        if group_fit is not None:
            group_fit["plot_file"] = str(group_plot_file)
            group_fit["plot_pdf_file"] = str(group_plot_pdf_file)
            group_fit_rows.append(group_fit)

    real_group_fit = pd.DataFrame(group_fit_rows)
    real_group_fit_file = OUTPUT_DIR / "real_group_relaxation_fit_summary.csv"
    real_group_fit.to_csv(real_group_fit_file, index=False)

    all_groups_plot = OUTPUT_DIR / "all_real_groups_relaxation.png"
    all_groups_plot_pdf = all_groups_plot.with_suffix(".pdf")

    all_group_overview = plot_all_real_groups(
        real_rheology=real_rheology,
        save_path=all_groups_plot,
    )

    all_group_overview_file = OUTPUT_DIR / "all_real_groups_fit_summary.csv"
    all_group_overview.to_csv(all_group_overview_file, index=False)

    print("\n======================================")
    print("Real data model analysis finished")
    print("======================================")

    print("\nSaved files:")
    print(sample_summary_file)
    print(group_summary_file)
    print(real_group_fit_file)
    print(all_group_overview_file)
    print(all_curves_file)
    print(all_groups_plot)
    print(all_groups_plot_pdf)

    print("\nSample summary:")
    print(sample_summary)

    print("\nGroup summary:")
    print(group_summary)

    print("\nReal group relaxation fit:")
    print(real_group_fit)

    return sample_summary, group_summary, real_group_fit

if __name__ == "__main__":
    run_real_data_simulation()