# Battery SOC 2RC + EKF Validation

This repository contains the Maryland INR 18650-20R 25°C low-current OCV and incremental OCV datasets. The Python script `validate_2rc_ekf.py` performs the first-stage validation of a 2RC equivalent circuit model with an extended Kalman filter (EKF):

1. Select the C/20 discharge data (step 3) from the low-current OCV file.
2. Estimate the battery capacity from the discharge segment and compute Coulomb-counting SOC.
3. Extract the 25°C SOC-OCV curve from incremental OCV rest steps.
4. Fit the 2RC parameters (R0, R1, C1, R2, C2) using nonlinear least squares.
5. Run the EKF SOC estimator and compare the result with Coulomb-counting SOC.

## Requirements

Install the Python dependencies (only needed once):

```bash
python -m pip install pandas openpyxl numpy scipy matplotlib
```

## Run the validation

From the repository root:

```bash
python validate_2rc_ekf.py --output-dir outputs
```

The script prints the fitted parameters and SOC error metrics, then writes the validation plot to `outputs/validation_summary.png`.

## Notes on data selection

- Low-current OCV data: uses **Pgm step 3** (C/20 discharge) from `11_5_2015_low current OCV test_SP20-1.xlsx`.
- Incremental OCV data: uses rest steps **Step_Index 1, 4, 6** from `12_2_2015_Incremental OCV test_SP20-1.xlsx` to build the SOC-OCV curve at 25°C.

You can change the input files or downsample rate using command-line arguments, e.g. `--downsample 20`.
