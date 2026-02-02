#!/usr/bin/env python3
"""Validate a 2-RC battery model using provided OCV and incremental test data."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import least_squares


@dataclass(frozen=True)
class OcvCurve:
    soc: np.ndarray
    ocv: np.ndarray
    ocv_from_soc: interp1d
    soc_from_ocv: interp1d
    capacity_ah: float


@dataclass(frozen=True)
class RcParams:
    r0: float
    r1: float
    c1: float
    r2: float
    c2: float


def load_low_current_ocv(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    df = df[['Duration (sec)', 'mV', 'mA']].dropna()
    df = df.rename(columns={'Duration (sec)': 'time_s', 'mV': 'voltage_mv', 'mA': 'current_ma'})
    df['time_s'] = df['time_s'].astype(float)
    df['voltage_v'] = df['voltage_mv'] / 1000.0
    df['current_a'] = -df['current_ma'] / 1000.0  # discharge current is positive
    return df


def build_ocv_curve(df: pd.DataFrame, rest_current_a: float, bins: int) -> OcvCurve:
    time_s = df['time_s'].to_numpy()
    dt = np.diff(time_s, prepend=time_s[0])
    dt = np.where(dt < 0, 0, dt)
    current_a = df['current_a'].to_numpy()

    discharge_ah = (np.clip(current_a, 0, None) * dt).sum() / 3600.0
    if discharge_ah <= 0:
        raise ValueError('Unable to determine discharge capacity from low current data.')

    soc = 1.0 - np.cumsum(current_a * dt) / 3600.0 / discharge_ah
    soc = np.clip(soc, 0.0, 1.0)

    rest_mask = np.abs(current_a) <= rest_current_a
    soc_rest = soc[rest_mask]
    ocv_rest = df['voltage_v'].to_numpy()[rest_mask]

    if len(soc_rest) == 0:
        raise ValueError('No rest data found for OCV curve fitting.')

    bin_edges = np.linspace(0, 1, bins + 1)
    bin_ids = np.digitize(soc_rest, bin_edges) - 1
    soc_bins = []
    ocv_bins = []
    for idx in range(bins):
        mask = bin_ids == idx
        if np.any(mask):
            soc_bins.append(float(np.mean(soc_rest[mask])))
            ocv_bins.append(float(np.mean(ocv_rest[mask])))

    soc_curve = np.array(soc_bins)
    ocv_curve = np.array(ocv_bins)
    order = np.argsort(soc_curve)
    soc_curve = soc_curve[order]
    ocv_curve = ocv_curve[order]

    valid_mask = (soc_curve >= 0) & (soc_curve <= 1)
    soc_curve = soc_curve[valid_mask]
    ocv_curve = ocv_curve[valid_mask]

    ocv_from_soc = interp1d(
        soc_curve,
        ocv_curve,
        bounds_error=False,
        fill_value=(ocv_curve[0], ocv_curve[-1]),
    )

    ocv_sorted_idx = np.argsort(ocv_curve)
    ocv_sorted = ocv_curve[ocv_sorted_idx]
    soc_sorted = soc_curve[ocv_sorted_idx]
    ocv_unique, unique_idx = np.unique(ocv_sorted, return_index=True)
    soc_unique = soc_sorted[unique_idx]
    soc_from_ocv = interp1d(
        ocv_unique,
        soc_unique,
        bounds_error=False,
        fill_value=(soc_unique[0], soc_unique[-1]),
    )

    return OcvCurve(
        soc=soc_curve,
        ocv=ocv_curve,
        ocv_from_soc=ocv_from_soc,
        soc_from_ocv=soc_from_ocv,
        capacity_ah=discharge_ah,
    )


def load_incremental_data(path: str, cycle_index: int) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    frames = []
    for sheet in xl.sheet_names:
        if sheet.startswith('Channel'):
            frames.append(xl.parse(sheet))
    df = pd.concat(frames, ignore_index=True)
    df = df[df['Cycle_Index'] == cycle_index].copy()
    df = df.rename(columns={'Test_Time(s)': 'time_s', 'Current(A)': 'current_a', 'Voltage(V)': 'voltage_v'})
    df['time_s'] = df['time_s'].astype(float)
    df['current_a'] = -df['current_a'].astype(float)  # discharge current is positive
    df['voltage_v'] = df['voltage_v'].astype(float)
    return df


def compute_soc(current_a: np.ndarray, dt: np.ndarray, capacity_ah: float) -> np.ndarray:
    soc = 1.0 - np.cumsum(current_a * dt) / 3600.0 / capacity_ah
    return np.clip(soc, 0.0, 1.0)


def simulate_voltage(
    params: RcParams,
    current_a: np.ndarray,
    soc: np.ndarray,
    dt: np.ndarray,
    ocv_from_soc: interp1d,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v1 = 0.0
    v2 = 0.0
    v1_hist = np.zeros_like(current_a)
    v2_hist = np.zeros_like(current_a)
    voltage = np.zeros_like(current_a)

    for i in range(len(current_a)):
        if i > 0:
            a1 = np.exp(-dt[i] / (params.r1 * params.c1))
            a2 = np.exp(-dt[i] / (params.r2 * params.c2))
            v1 = v1 * a1 + params.r1 * (1 - a1) * current_a[i]
            v2 = v2 * a2 + params.r2 * (1 - a2) * current_a[i]
        v1_hist[i] = v1
        v2_hist[i] = v2
        voltage[i] = ocv_from_soc(soc[i]) - params.r0 * current_a[i] - v1 - v2
    return voltage, v1_hist, v2_hist


def fit_rc_parameters(
    df: pd.DataFrame,
    ocv_curve: OcvCurve,
    sample_stride: int,
    max_nfev: int,
) -> RcParams:
    time_s = df['time_s'].to_numpy()
    dt = np.diff(time_s, prepend=time_s[0])
    dt = np.where(dt < 0, 0, dt)
    current_a = df['current_a'].to_numpy()
    soc = compute_soc(current_a, dt, ocv_curve.capacity_ah)

    indices = np.arange(0, len(df), sample_stride)
    current_fit = current_a[indices]
    soc_fit = soc[indices]
    dt_fit = dt[indices]
    voltage_fit = df['voltage_v'].to_numpy()[indices]

    def residuals(x: np.ndarray) -> np.ndarray:
        params = RcParams(*x)
        pred, _, _ = simulate_voltage(params, current_fit, soc_fit, dt_fit, ocv_curve.ocv_from_soc)
        return pred - voltage_fit

    # Typical small-format Li-ion 2-RC starting guesses (Ohm/F range).
    r0_init = 0.005
    r1_init = 0.003
    c1_init = 2000.0
    r2_init = 0.008
    c2_init = 8000.0
    initial = np.array([r0_init, r1_init, c1_init, r2_init, c2_init])
    bounds = ([1e-5, 1e-5, 10.0, 1e-5, 10.0], [0.05, 0.05, 50000.0, 0.05, 50000.0])

    try:
        result = least_squares(residuals, initial, bounds=bounds, max_nfev=max_nfev)
        params = result.x
    except Exception as exc:
        print(f'Warning: parameter fit failed ({exc}). Using initial guesses.')
        params = initial

    return RcParams(*params)


def estimate_soc(
    df: pd.DataFrame,
    params: RcParams,
    ocv_curve: OcvCurve,
    soc_gain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_s = df['time_s'].to_numpy()
    dt = np.diff(time_s, prepend=time_s[0])
    dt = np.where(dt < 0, 0, dt)
    current_a = df['current_a'].to_numpy()
    voltage_v = df['voltage_v'].to_numpy()

    soc_cc = compute_soc(current_a, dt, ocv_curve.capacity_ah)

    soc_est = np.zeros_like(soc_cc)
    soc_est[0] = soc_cc[0]
    v1 = 0.0
    v2 = 0.0

    for i in range(len(current_a)):
        if i > 0:
            a1 = np.exp(-dt[i] / (params.r1 * params.c1))
            a2 = np.exp(-dt[i] / (params.r2 * params.c2))
            v1 = v1 * a1 + params.r1 * (1 - a1) * current_a[i]
            v2 = v2 * a2 + params.r2 * (1 - a2) * current_a[i]

            soc_pred = soc_est[i - 1] - current_a[i] * dt[i] / 3600.0 / ocv_curve.capacity_ah
            ocv_est = voltage_v[i] + params.r0 * current_a[i] + v1 + v2
            soc_meas = float(ocv_curve.soc_from_ocv(ocv_est))
            soc_est[i] = (1 - soc_gain) * soc_pred + soc_gain * soc_meas
            soc_est[i] = np.clip(soc_est[i], 0.0, 1.0)

    return soc_est, soc_cc, dt


def format_params(params: RcParams) -> str:
    return (
        f"R0={params.r0:.5f} Ω, R1={params.r1:.5f} Ω, C1={params.c1:.1f} F, "
        f"R2={params.r2:.5f} Ω, C2={params.c2:.1f} F"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate 2-RC battery model with provided data.')
    parser.add_argument(
        '--ocv-file',
        default='11_5_2015_low current OCV test_SP20-1.xlsx',
        help='Low current OCV Excel file.',
    )
    parser.add_argument(
        '--incremental-file',
        default='12_2_2015_Incremental OCV test_SP20-1.xlsx',
        help='Incremental OCV Excel file.',
    )
    parser.add_argument('--cycle-index', type=int, default=1, help='Cycle index to use for validation.')
    parser.add_argument('--rest-current', type=float, default=0.01, help='Rest current threshold (A).')
    parser.add_argument('--ocv-bins', type=int, default=200, help='Bins for SOC-OCV curve.')
    parser.add_argument('--fit-stride', type=int, default=5, help='Sample stride for RC fitting.')
    parser.add_argument(
        '--max-nfev',
        type=int,
        default=200,
        help='Max function evaluations for RC fitting (higher = slower, potentially more accurate).',
    )
    parser.add_argument('--soc-gain', type=float, default=0.05, help='SOC correction gain (0-1).')
    args = parser.parse_args()

    ocv_df = load_low_current_ocv(args.ocv_file)
    ocv_curve = build_ocv_curve(ocv_df, rest_current_a=args.rest_current, bins=args.ocv_bins)

    print('OCV data selection:')
    print(f"  total rows: {len(ocv_df)}, rest rows: {(np.abs(ocv_df['current_a']) <= args.rest_current).sum()}")
    print(f"  estimated capacity: {ocv_curve.capacity_ah:.3f} Ah")

    inc_df = load_incremental_data(args.incremental_file, args.cycle_index)
    if inc_df.empty:
        raise ValueError(f'No incremental data found for cycle {args.cycle_index}.')
    print('\nIncremental data selection:')
    print(f"  cycle {args.cycle_index} rows: {len(inc_df)}")
    print(
        f"  current range: {inc_df['current_a'].min():.3f} to {inc_df['current_a'].max():.3f} A"
    )

    params = fit_rc_parameters(inc_df, ocv_curve, sample_stride=args.fit_stride, max_nfev=args.max_nfev)
    print('\nFitted 2-RC parameters:')
    print(f"  {format_params(params)}")

    soc_est, soc_cc, _ = estimate_soc(inc_df, params, ocv_curve, soc_gain=args.soc_gain)
    error = soc_est - soc_cc
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_err = np.max(np.abs(error))

    print('\nSOC comparison (model vs. ampere-hour integration):')
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Max:  {max_err:.4f}")


if __name__ == '__main__':
    main()
