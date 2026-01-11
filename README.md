# Basket Option Pricing Project

This project studies the pricing of European basket options in a two-asset Black-Scholes framework.

## Methods implemented
- Log-normal approximation of the basket distribution  
- Monte Carlo simulation with correlated Brownian motions  
- Variance reduction using conditioning  
- Variance reduction using control variates  

## Features
- Box-Muller Gaussian simulation  
- Cholesky decomposition for correlation  
- Confidence intervals for Monte Carlo estimators  
- Sensitivity analysis with respect to:
  - Correlation ρ  
  - Basket weight α  
  - Strike K  

## Files
- `basket_option_pricing.py`: Python implementation  
- `rapport.pdf`: Full mathematical report  

## Technologies
- Python  
- NumPy  
- SciPy  
- Matplotlib  

