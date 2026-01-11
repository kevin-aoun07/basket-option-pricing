# Basket Option Pricing (Monte Carlo + Variance Reduction)

This repository contains a quantitative finance project on **pricing a European basket call option** in a **two-asset Black–Scholes** framework.

The project includes:
- A **log-normal approximation** of the basket distribution
- **Monte Carlo simulation** with correlated Brownian motions
- **Variance reduction** techniques (conditioning + control variates)
- Sensitivity analyses and confidence intervals

---

## Repository content

### ✅ Code
- **`original_code.py`**  
  The original academic script used for the project report.  
  It runs *all questions/experiments sequentially* and displays figures as it goes.  
  ⚠️ Some sections can take a long time to run.

- **`basket_option_pricing.py`**  
  A **GitHub-friendly version** of the code (same logic, reorganized for usability):  
  - `FAST` mode (default): reduced Monte Carlo sample sizes and smaller grids  
  - `FULL` mode: reproduces the full experiments  
  - Run **one section only** using `--section`  
  - Option to **save figures** to `./figures` using `--savefig`  

> This is the recommended script to run for a quick overview.

### 📄 Report
- **`rapport.pdf`**  
  Full mathematical derivations, methodology, and numerical results.

(Optional) If you also uploaded slides, you can list them here:
- `presentation.pdf` (optional)

---

## Installation

```bash
pip install numpy scipy matplotlib
