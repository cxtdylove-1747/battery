# Battery model validation

This repository contains battery test data and a small Python script to validate a 2-RC equivalent circuit model.

## Data selection

The validation uses two Excel datasets:

- `11_5_2015_low current OCV test_SP20-1.xlsx`: low current OCV test data. The script uses the rest segments (current close to 0 A) to build a SOC-OCV curve and estimates capacity from the discharge portion.
- `12_2_2015_Incremental OCV test_SP20-1.xlsx`: incremental OCV test data. The script selects one `Cycle_Index` (default `1`) for model fitting and SOC validation.

## Run the validation

Install dependencies (one-time):

```bash
python -m pip install numpy pandas scipy openpyxl
```

Run the validation script:

```bash
python model_validation.py
```

Optional arguments:

```bash
python model_validation.py \
  --cycle-index 1 \
  --rest-current 0.01 \
  --ocv-bins 200 \
  --fit-stride 5 \
  --soc-gain 0.05
```

The script outputs:

- OCV data selection summary
- Estimated 2-RC parameters
- SOC comparison metrics (MAE/RMSE/Max) between model estimation and ampere-hour integration

## Notes

- Discharge current is treated as positive in the calculations (consistent with the incremental dataset sign convention).
- The SOC-OCV curve is built from rest data (current within the `--rest-current` threshold).
- Adjust the `--cycle-index` if you want to validate on a different incremental cycle.
