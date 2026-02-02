#!/usr/bin/env python
"""Validate a 2RC battery model + EKF SOC estimator using Maryland INR18650 data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _compute_dt(time_s: pd.Series) -> pd.Series:
    dt = time_s.diff()
    if len(dt) > 1:
        default = dt.iloc[1]
    else:
        default = np.nan
    if pd.isna(default) or default <= 0:
        default = dt[dt > 0].median()
    if pd.isna(default) or default <= 0:
        default = 1.0
    return dt.fillna(default)


def load_low_current_discharge(
    file_path: Path, discharge_step: int = 3, current_threshold: float = 0.01
) -> tuple[pd.DataFrame, float]:
    data = pd.read_excel(file_path)
    discharge = data[data["Pgm step"] == discharge_step].copy()
    discharge = discharge.rename(
        columns={"Duration (sec)": "time_s", "mA": "current_ma", "mV": "voltage_mv"}
    )
    discharge["current_a"] = -discharge["current_ma"] / 1000.0
    discharge["voltage_v"] = discharge["voltage_mv"] / 1000.0
    discharge = discharge[discharge["current_a"].abs() >= current_threshold].copy()
    discharge["time_s"] = discharge["time_s"] - discharge["time_s"].iloc[0]
    discharge["dt"] = _compute_dt(discharge["time_s"])
    capacity_ah = (discharge["current_a"] * discharge["dt"]).sum() / 3600.0
    discharge["soc_cc"] = 1.0 - (
        (discharge["current_a"] * discharge["dt"]).cumsum() / 3600.0 / capacity_ah
    )
    return discharge.reset_index(drop=True), capacity_ah


def load_incremental_ocv(file_path: Path, capacity_ah: float) -> pd.DataFrame:
    xl = pd.ExcelFile(file_path)
    channels = [sheet for sheet in xl.sheet_names if sheet.startswith("Channel")]
    frames = [xl.parse(sheet) for sheet in channels]
    data = pd.concat(frames, ignore_index=True)
    rest = data[data["Current(A)"] == 0].copy()
    rest = rest[rest["Step_Index"].isin([1, 4, 6])]
    ocv = (
        rest.groupby("Discharge_Capacity(Ah)")["Voltage(V)"]
        .mean()
        .reset_index()
        .sort_values("Discharge_Capacity(Ah)")
    )
    ocv["soc"] = 1.0 - ocv["Discharge_Capacity(Ah)"] / capacity_ah
    ocv["soc"] = ocv["soc"].clip(0.0, 1.0)
    ocv = ocv.drop_duplicates(subset=["soc"]).sort_values("soc")
    return ocv.reset_index(drop=True)


def build_ocv_interpolator(ocv: pd.DataFrame) -> tuple[interp1d, interp1d]:
    soc = ocv["soc"].to_numpy()
    voltage = ocv["Voltage(V)"].to_numpy()
    ocv_func = interp1d(soc, voltage, fill_value="extrapolate")
    dvoltage = np.gradient(voltage, soc)
    docv_func = interp1d(soc, dvoltage, fill_value="extrapolate")
    return ocv_func, docv_func


def simulate_2rc(
    current: np.ndarray,
    dt: np.ndarray,
    ocv_func: interp1d,
    params: np.ndarray,
    capacity_ah: float,
    soc0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    r0, r1, c1, r2, c2 = params
    v1 = 0.0
    v2 = 0.0
    soc = soc0
    v_pred = np.zeros_like(current)
    soc_trace = np.zeros_like(current)
    for i in range(len(current)):
        a1 = np.exp(-dt[i] / (r1 * c1))
        a2 = np.exp(-dt[i] / (r2 * c2))
        soc = soc - current[i] * dt[i] / 3600.0 / capacity_ah
        soc = float(np.clip(soc, 0.0, 1.0))
        v1 = a1 * v1 + r1 * (1.0 - a1) * current[i]
        v2 = a2 * v2 + r2 * (1.0 - a2) * current[i]
        v_pred[i] = float(ocv_func(soc)) - current[i] * r0 - v1 - v2
        soc_trace[i] = soc
    return v_pred, soc_trace


def fit_2rc_params(
    current: np.ndarray,
    voltage: np.ndarray,
    dt: np.ndarray,
    ocv_func: interp1d,
    capacity_ah: float,
) -> np.ndarray:
    # Initial guesses follow typical 2RC ranges (Ohms and Farads) for 18650 cells.
    initial = np.log(np.array([0.01, 0.01, 2000.0, 0.02, 4000.0]))
    lower = np.log(np.array([1e-4, 1e-4, 100.0, 1e-4, 100.0]))
    upper = np.log(np.array([0.2, 0.2, 50000.0, 0.2, 50000.0]))

    def residuals(log_params: np.ndarray) -> np.ndarray:
        params = np.exp(log_params)
        v_pred, _ = simulate_2rc(current, dt, ocv_func, params, capacity_ah)
        return v_pred - voltage

    result = least_squares(residuals, initial, bounds=(lower, upper))
    return np.exp(result.x)


def run_ekf(
    current: np.ndarray,
    voltage: np.ndarray,
    dt: np.ndarray,
    ocv_func: interp1d,
    docv_func: interp1d,
    params: np.ndarray,
    capacity_ah: float,
) -> np.ndarray:
    r0, r1, c1, r2, c2 = params
    state = np.array([1.0, 0.0, 0.0])
    cov = np.diag([1e-4, 1e-3, 1e-3])
    # EKF noise tuning for SOC state and RC voltage states; measurement noise in volts.
    process = np.diag([1e-7, 1e-6, 1e-6])
    meas = np.array([[0.005**2]])
    soc_est = np.zeros_like(current)
    for i in range(len(current)):
        a1 = np.exp(-dt[i] / (r1 * c1))
        a2 = np.exp(-dt[i] / (r2 * c2))
        soc_pred = state[0] - current[i] * dt[i] / 3600.0 / capacity_ah
        v1_pred = a1 * state[1] + r1 * (1.0 - a1) * current[i]
        v2_pred = a2 * state[2] + r2 * (1.0 - a2) * current[i]
        state_pred = np.array([soc_pred, v1_pred, v2_pred])
        f_mat = np.array([[1.0, 0.0, 0.0], [0.0, a1, 0.0], [0.0, 0.0, a2]])
        cov = f_mat @ cov @ f_mat.T + process
        soc_clip = float(np.clip(state_pred[0], 0.0, 1.0))
        ocv = float(ocv_func(soc_clip))
        docv = float(docv_func(soc_clip))
        v_pred = ocv - current[i] * r0 - state_pred[1] - state_pred[2]
        h_mat = np.array([[docv, -1.0, -1.0]])
        s_mat = h_mat @ cov @ h_mat.T + meas
        k_gain = cov @ h_mat.T @ np.linalg.inv(s_mat)
        residual = voltage[i] - v_pred
        state = state_pred + (k_gain.flatten() * residual)
        state[0] = float(np.clip(state[0], 0.0, 1.0))
        cov = (np.eye(3) - k_gain @ h_mat) @ cov
        soc_est[i] = state[0]
    return soc_est


def build_plots(
    output_dir: Path,
    time_h: np.ndarray,
    voltage: np.ndarray,
    voltage_pred: np.ndarray,
    soc_cc: np.ndarray,
    soc_ekf: np.ndarray,
    ocv: pd.DataFrame,
    ocv_func: interp1d,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    soc_grid = np.linspace(0.0, 1.0, 200)

    axes[0, 0].plot(ocv["soc"], ocv["Voltage(V)"], "o", label="OCV points")
    axes[0, 0].plot(soc_grid, ocv_func(soc_grid), label="OCV interp")
    axes[0, 0].set_xlabel("SOC")
    axes[0, 0].set_ylabel("Voltage (V)")
    axes[0, 0].set_title("SOC-OCV Curve")
    axes[0, 0].legend()

    axes[0, 1].plot(time_h, voltage, label="Measured")
    axes[0, 1].plot(time_h, voltage_pred, label="2RC fit")
    axes[0, 1].set_xlabel("Time (h)")
    axes[0, 1].set_ylabel("Voltage (V)")
    axes[0, 1].set_title("Voltage Fit")
    axes[0, 1].legend()

    axes[1, 0].plot(time_h, soc_cc, label="Coulomb counting")
    axes[1, 0].plot(time_h, soc_ekf, label="EKF")
    axes[1, 0].set_xlabel("Time (h)")
    axes[1, 0].set_ylabel("SOC")
    axes[1, 0].set_title("SOC Comparison")
    axes[1, 0].legend()

    soc_error = soc_ekf - soc_cc
    axes[1, 1].plot(time_h, soc_error)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("Time (h)")
    axes[1, 1].set_ylabel("SOC Error")
    axes[1, 1].set_title("EKF - Coulomb Counting")

    fig.tight_layout()
    fig.savefig(output_dir / "validation_summary.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a 2RC + EKF SOC estimator.")
    parser.add_argument(
        "--low-current-file",
        type=Path,
        default=Path("11_5_2015_low current OCV test_SP20-1.xlsx"),
    )
    parser.add_argument(
        "--incremental-ocv-file",
        type=Path,
        default=Path("12_2_2015_Incremental OCV test_SP20-1.xlsx"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--downsample", type=int, default=10)
    args = parser.parse_args()

    low_current, capacity_ah = load_low_current_discharge(args.low_current_file)
    print(
        f"Low-current discharge samples: {len(low_current)} | "
        f"capacity estimate: {capacity_ah:.3f} Ah"
    )

    ocv = load_incremental_ocv(args.incremental_ocv_file, capacity_ah)
    print(f"Incremental OCV points used: {len(ocv)}")
    ocv_func, docv_func = build_ocv_interpolator(ocv)

    if args.downsample > 1:
        low_current = low_current.iloc[:: args.downsample].copy()
        low_current["dt"] = _compute_dt(low_current["time_s"])
        low_current = low_current.reset_index(drop=True)

    current = low_current["current_a"].to_numpy()
    voltage = low_current["voltage_v"].to_numpy()
    dt = low_current["dt"].to_numpy()
    time_h = low_current["time_s"].to_numpy() / 3600.0
    soc_cc = low_current["soc_cc"].to_numpy()

    params = fit_2rc_params(current, voltage, dt, ocv_func, capacity_ah)
    voltage_pred, _ = simulate_2rc(current, dt, ocv_func, params, capacity_ah)
    soc_ekf = run_ekf(current, voltage, dt, ocv_func, docv_func, params, capacity_ah)

    soc_error = soc_ekf - soc_cc
    mae = float(np.mean(np.abs(soc_error)))
    rmse = float(np.sqrt(np.mean(soc_error**2)))
    voltage_rmse = float(np.sqrt(np.mean((voltage_pred - voltage) ** 2)))

    print(
        "Fitted params: "
        f"R0={params[0]:.4f} Ω, R1={params[1]:.4f} Ω, C1={params[2]:.1f} F, "
        f"R2={params[3]:.4f} Ω, C2={params[4]:.1f} F"
    )
    print(f"Voltage RMSE: {voltage_rmse:.4f} V")
    print(f"SOC MAE: {mae:.4f}, SOC RMSE: {rmse:.4f}")

    build_plots(
        args.output_dir, time_h, voltage, voltage_pred, soc_cc, soc_ekf, ocv, ocv_func
    )
    print(f"Saved plot to {args.output_dir / 'validation_summary.png'}")


if __name__ == "__main__":
    main()
