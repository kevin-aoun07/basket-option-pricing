Basket Option Pricing (Monte Carlo + Variance Reduction)

This repository contains a quantitative finance project on pricing a European basket call option in a two-asset Black–Scholes framework.

The project includes:

A log-normal approximation of the basket distribution

Monte Carlo simulation with correlated Brownian motions

Variance reduction techniques (conditioning + control variates)

Sensitivity analyses and confidence intervals

Repository Content
✅ Code

original_code.py
The original academic script used for the project report.
It runs all questions/experiments sequentially and displays figures as it goes.
⚠️ Some sections can take a long time to run.

basket_option_pricing.py
A GitHub-friendly version of the code (same logic, reorganized for usability):

FAST mode (default): reduced Monte Carlo sample sizes and smaller grids

FULL mode: reproduces the full experiments

Run one section only using --section

Option to save figures to ./figures using --savefig

This is the recommended script to run for a quick overview.

📄 Report

rapport.pdf
Full mathematical derivations, methodology, and numerical results.

(Optional)

presentation.pdf – Slides summarizing the project (if available)

Installation
pip install numpy scipy matplotlib

How to Run
1) Run the Original Script (Academic Version)
python original_code.py


This will execute all experiments sequentially and display the figures as they are generated.
⚠️ Some parts may take a long time to run.

2) Run the GitHub-Friendly Version (Recommended)
Fast demo (default)

Runs quickly with reduced Monte Carlo sample sizes and smaller parameter grids:

python basket_option_pricing.py

Full run (slow)

Reproduces the full experiments:

python basket_option_pricing.py --full

Run a specific section only

Available sections:

basics, mc, q6, q7, q8, q9, q10, q11, q12


Example (run only the correlation ρ study):

python basket_option_pricing.py --section q7

Save figures instead of displaying them

Figures will be saved into a figures/ folder:

python basket_option_pricing.py --savefig


You can combine options, for example:

python basket_option_pricing.py --full --savefig

Notes on Performance

Some parts of the project are computationally expensive (large Monte Carlo sample sizes + parameter sweeps).
For this reason, the GitHub-friendly version runs in FAST mode by default.
