## Citadel Global Datathon — Causal Modeling Project

### Overview
This repository contains our Citadel Global Datathon submission. We study the relationship between a treatment-like variable (`Salary Index`) and outcomes (e.g., `Early Career Pay`) using a causal inference workflow. We implement generalized inverse probability weighting (IPW) to balance confounders and fit a marginal structural model (MSM) via weighted least squares (WLS). We include exploratory analysis and visualizations to communicate findings.

Key ideas:
- **Treatment**: `Salary Index`
- **Outcome**: `Early Career Pay` (also explore `Mid-Career Pay`)
- **Confounders**: `upgrntp`, `noninst_per_student`, `actcm75`, `is_medical`, `stem_perc`, `admit_rate`, `inst_per_student`
- **Approach**: Generalized IPW (via normal density ratio from linear models) + MSM (WLS) for outcome vs treatment

### Repository structure
- `data/`
  - `train.csv`: modeling dataset used by notebooks
- `raw_data/`
  - `fips2county.csv`
  - `living-wage.json`
  - `PayScale Salary Data 2021.csv`
- `notebooks/`
  - `preprocess.ipynb`: raw-to-modeling data preparation (feature cleaning/creation)
  - `train.ipynb`: causal modeling workflow (IPW + WLS), figures, and diagnostics

### Environment
- Python 3.8+ (notebooks were run with Python 3.8.8)
- Jupyter (Lab or Notebook)

Recommended packages:
- pandas, numpy, scipy, statsmodels, seaborn, matplotlib, jupyterlab

Quick setup with `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy scipy statsmodels seaborn matplotlib jupyterlab
```

Optional: create a `requirements.txt` and install via `pip install -r requirements.txt`.

### Data preparation (summary)
- Currency and percentage strings are parsed to numeric (e.g., `Early Career Pay`, `Mid-Career Pay`, `% High Meaning`, `% STEM Degrees`).
- Derived features per-student spend:
  - `inst_per_student = Total Instruction / Total Undergrads`
  - `noninst_per_student = (Current Year Total Cost - Total Instruction) / Total Undergrads`
- Imputation: `actcm75` filled with its mean when missing.
- Optional filtering for robustness/sanity:
  - drop `instnm` containing "Columbia" (to reduce leverage)
  - drop extreme `Salary Index` outliers (e.g., `Salary Index` < 100000 kept)

All transformation logic is visible in `notebooks/train.ipynb` (and complementary steps in `preprocess.ipynb`).

### Methodology
1. **Generalized IPW weights**
   - Fit OLS for `treatment ~ 1 + confounders` to obtain fitted mean and residual std.
   - Compute normal PDF of observed treatment given fitted mean/std (denominator).
   - Fit a null OLS for `treatment ~ 1` (no confounders) and compute PDF (numerator).
   - Weight = numerator / denominator.
2. **Marginal Structural Model (MSM)**
   - Fit `WLS(outcome ~ 1 + treatment, weights=IPW)`.
   - Use the fitted model for counterfactual predictions and confidence intervals.
3. **Diagnostics**
   - Spearman correlation heatmap for outcome/treatment/confounders.
   - Weighted scatter visualization of `outcome` vs `treatment` (`s='weight'`).
   - Prediction curve with 95% CI across a treatment grid.

Notes and limitations:
- WLS standard errors assume correctly specified covariance; for more conservative uncertainty, consider GEE/robust SEs.
- Linear-Gaussian assumptions in density specification and MSM may be misspecified; consider alternative propensity models or doubly robust estimators.
- Results are sensitive to outlier handling; we provide flags controlling exclusions.

### Reproducing results
1. Ensure the environment is set up (see Environment section) and data is present in `data/` and `raw_data/`.
2. Launch Jupyter:
```bash
jupyter lab  # or: jupyter notebook
```
3. Run notebooks in order:
   - `notebooks/preprocess.ipynb` (if needed for your run)
   - `notebooks/train.ipynb`

The `train.ipynb` will:
- Load `data/train.csv`
- Build features, confounder set, and IPW weights
- Fit MSM (WLS)
- Plot outcome-vs-treatment with confidence intervals
- Demonstrate a school-level counterfactual (e.g., add $10,000 to treatment and predict)

### Example result (from the included notebook)
- Weighted linear model estimates a positive slope between `Salary Index` and `Early Career Pay` (example coefficient ≈ 0.68 in the snapshot), implying a 10k increase in treatment is associated with ≈ $6.8k in early career pay, under model assumptions.
- Please interpret directionally; conclusions depend on specification, confounders, and data cleaning choices.

### How to extend
- Swap outcome to `Mid-Career Pay` and re-run.
- Add or revise confounders (e.g., school selectivity, geography) and recompute weights.
- Replace OLS-based density with alternative treatment models (e.g., GLMs, nonparametric estimators) for IPW.
- Use doubly robust learners (AIPW/TMLE) to reduce reliance on either model being correct.

### Project tips
- Keep raw sources in `raw_data/` untouched; write any new derived datasets to `data/`.
- Version notebooks or export scripts to ensure exact reproducibility.
- Save random seeds when introducing stochastic steps.

### Acknowledgements
- Public data sources summarized in `raw_data/`.
- Python ecosystem: pandas, statsmodels, scipy, numpy, seaborn, matplotlib.

### License
This repository is provided for the Datathon. Add a license if you plan to share or extend beyond the event. 