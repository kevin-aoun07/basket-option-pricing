# basket_option_pricing.py
import argparse
import os
import time

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import cholesky

np.random.seed(0)

# =========================================================
# Utils: figures
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_or_show(fig_name: str, savefig: bool, fig_dir: str = "figures") -> None:
    """
    If savefig=True: saves current figure to fig_dir/fig_name.png and closes it.
    Else: shows it.
    """
    if savefig:
        ensure_dir(fig_dir)
        plt.savefig(os.path.join(fig_dir, f"{fig_name}.png"), dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

# =========================================================
# 1) Log-normal approximation functions
# =========================================================
def aprox(x: float) -> float:
    # Abramowitz & Stegun approximation for N(x)
    b0 = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    if x > 0:
        t = 1 / (1 + b0 * x)
        cdf = b1*t + b2*t**2 + b3*t**3 + b4*t**4 + b5*t**5
        return 1 - np.exp(-0.5 * x**2) * cdf / np.sqrt(2 * np.pi)
    elif x < 0:
        return 1 - aprox(-x)
    else:
        return 0.5

def call_price(S_0: float, sigma: float, T: float, r: float, K: float) -> float:
    d = (np.log(K/S_0) - (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S_0 * aprox(-d + sigma * np.sqrt(T)) - K * np.exp(-r * T) * aprox(-d)

def call_price_1(S_0: float, sigma: float, T: float, r: float, K: float) -> float:
    d = (np.log(K/S_0) - (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S_0 * norm.cdf(-d + sigma * np.sqrt(T)) - K * np.exp(-r * T) * norm.cdf(-d)

def sigma_basket(alpha: float, beta: float, S1_0: float, S2_0: float,
                 sigma1: float, sigma2: float, rho: float, T: float) -> float:
    SB_0 = alpha * S1_0 + beta * S2_0
    num = (
        (alpha**2 * S1_0**2 / SB_0**2) * np.exp(sigma1**2 * T) +
        (beta**2 * S2_0**2 / SB_0**2) * np.exp(sigma2**2 * T) +
        (2 * alpha * beta * S1_0 * S2_0 / SB_0**2) * np.exp(sigma1 * sigma2 * rho * T)
    )
    sigma_b_sq = (1 / T) * np.log(num)
    return float(np.sqrt(sigma_b_sq))

# =========================================================
# 2) Monte Carlo primitives
# =========================================================
def Box_Muller(n: int):
    U = npr.random(n)
    V = npr.random(n)
    X = np.sqrt(-2 * np.log(U)) * np.cos(2 * np.pi * V)
    Y = np.sqrt(-2 * np.log(U)) * np.sin(2 * np.pi * V)
    return X, Y

def Euro_call(t: float, alpha_S_1_0: float, beta_S_2_0: float, r: float,
              sigma_1: float, sigma_2: float, rho: float, N: int, K: float):
    X, Y = Box_Muller(N)
    W1 = X
    W2 = rho * X + np.sqrt(1 - rho**2) * Y

    S_T = (
        alpha_S_1_0 * np.exp((r - 0.5 * sigma_1**2) * t + sigma_1 * np.sqrt(t) * W1) +
        beta_S_2_0 * np.exp((r - 0.5 * sigma_2**2) * t + sigma_2 * np.sqrt(t) * W2)
    )

    payoff = np.exp(-r * t) * np.maximum(S_T - K, 0)
    MC_price = float(np.mean(payoff))
    MC_error = float(1.96 * np.std(payoff) / np.sqrt(N))
    IC_sup = MC_price + MC_error
    IC_inf = MC_price - MC_error
    return MC_price, MC_error, IC_inf, IC_sup

def Gauss_Correl(N: int, Gamma: np.ndarray):
    A = cholesky(Gamma)
    X, Y = Box_Muller(N)
    G = np.vstack([X, Y])
    Z = np.dot(A, G)
    return Z

def Euro_call_chole(t: float, alpha_S_1_0: float, beta_S_2_0: float, r: float,
                    sigma_1: float, sigma_2: float, rho: float, N: int, K: float, Gamma: np.ndarray):
    Z = Gauss_Correl(N, Gamma)
    W1, W2 = Z[0], Z[1]

    S_T = (
        alpha_S_1_0 * np.exp((r - 0.5 * sigma_1**2) * t + sigma_1 * np.sqrt(t) * W1) +
        beta_S_2_0 * np.exp((r - 0.5 * sigma_2**2) * t + sigma_2 * np.sqrt(t) * W2)
    )

    payoff = np.exp(-r * t) * np.maximum(S_T - K, 0)
    MC_price = float(np.mean(payoff))
    MC_error = float(1.96 * np.std(payoff) / np.sqrt(N))
    IC_sup = MC_price + MC_error
    IC_inf = MC_price - MC_error
    return MC_price, MC_error, IC_inf, IC_sup

# =========================================================
# 3) Conditioning variance reduction
# =========================================================
def phi(x: float, t: float, sigma1: float, sigma2: float,
        beta: float, alpha: float, r: float, K: float, rho: float) -> float:
    S_cond = beta * np.exp(-0.5 * (sigma2 * rho)**2 * t + sigma2 * rho * x)
    sigma_cond = sigma2 * np.sqrt(1 - rho**2)
    K_cond = K - alpha * np.exp((r - 0.5 * sigma1**2) * t + sigma1 * x)

    if K_cond > 0:
        return np.exp(r * t) * call_price(S_cond, sigma_cond, t, r, K_cond)
    else:
        return S_cond * np.exp(r * t) - K_cond

def Euro_call_red_var(t: float, sigma1: float, sigma2: float,
                      beta: float, alpha: float, r: float, K: float, rho: float, N: int):
    X, _ = Box_Muller(N)
    W1 = np.sqrt(t) * X

    # NOTE: this loop is the slow part in your original code.
    Z = np.array([phi(W1[i], t, sigma1, sigma2, beta, alpha, r, K, rho) for i in range(N)])
    MC_price = float(np.exp(-r * t) * np.mean(Z))
    MC_error = float(1.96 * np.std(Z) / np.sqrt(N))
    IC_sup = MC_price + MC_error
    IC_inf = MC_price - MC_error
    return MC_price, MC_error, IC_inf, IC_sup

# =========================================================
# 4) Control variate using call-put parity
# =========================================================
def Euro_call_controle(t: float, alpha_S_1_0: float, beta_S_2_0: float, r: float,
                       sigma_1: float, sigma_2: float, rho: float, N: int, K: float):
    X, Y = Box_Muller(N)
    W1 = X
    W2 = rho * X + np.sqrt(1 - rho**2) * Y

    S_T = (
        alpha_S_1_0 * np.exp((r - 0.5 * sigma_1**2) * t + sigma_1 * np.sqrt(t) * W1) +
        beta_S_2_0 * np.exp((r - 0.5 * sigma_2**2) * t + sigma_2 * np.sqrt(t) * W2)
    )

    payoff_put = np.exp(-r * t) * np.maximum(K - S_T, 0)

    # Call = S1+S2 - K e^{-rt} + Put  (here with weights)
    MC_price = float(alpha_S_1_0 + beta_S_2_0 - K * np.exp(-r * t) + np.mean(payoff_put))
    MC_error = float(1.96 * np.std(payoff_put) / np.sqrt(N))
    IC_sup = MC_price + MC_error
    IC_inf = MC_price - MC_error
    return MC_price, MC_error, IC_sup, IC_inf

# =========================================================
# 5) Delta estimation
# =========================================================
def delta_approx_log_normale(S1_0: float, S2_0: float, alpha: float, beta: float,
                             sigma1: float, sigma2: float, rho: float, T: float, r: float, K: float) -> float:
    SB_0 = alpha * S1_0 + beta * S2_0

    f = (
        (alpha**2 * S1_0**2 / SB_0**2) * np.exp(sigma1**2 * T) +
        (beta**2 * S2_0**2 / SB_0**2) * np.exp(sigma2**2 * T) +
        (2 * alpha * beta * S1_0 * S2_0 / SB_0**2) * np.exp(sigma1 * sigma2 * rho * T)
    )
    sigma_b_sq = (1 / T) * np.log(f)
    sigma_b = np.sqrt(sigma_b_sq)

    d1 = (np.log(SB_0 / K) + (r + 0.5 * sigma_b**2) * T) / (sigma_b * np.sqrt(T))
    d2 = d1 - sigma_b * np.sqrt(T)

    def df(S1_0_, S2_0_, alpha_, beta_, sigma1_, sigma2_, rho_, T_):
        SB_0_ = alpha_ * S1_0_ + beta_ * S2_0_
        term1 = -2 * alpha_**3 * S1_0_**2 * np.exp(sigma1_**2 * T_) / SB_0_**3
        term2 = -4 * alpha_**2 * beta_ * S1_0_ * S2_0_ * np.exp(sigma1_ * sigma2_ * rho_ * T_) / SB_0_**3
        term3 =  2 * alpha_**2 * S1_0_ * np.exp(sigma1_**2 * T_) / SB_0_**2
        term4 = -2 * alpha_ * beta_**2 * S2_0_**2 * np.exp(sigma2_**2 * T_) / SB_0_**3
        term5 =  2 * alpha_ * beta_ * S2_0_ * np.exp(sigma1_ * sigma2_ * rho_ * T_) / SB_0_**2
        return term1 + term2 + term3 + term4 + term5

    f_der = df(S1_0, S2_0, alpha, beta, sigma1, sigma2, rho, T)
    sigma_der = (f_der) / (2 * f * sigma_b * T)

    d1_der = (alpha / SB_0 + T * sigma_b * sigma_der) / (sigma_b * np.sqrt(T)) - d1 * sigma_der / sigma_b
    d2_der = (alpha / SB_0 - T * sigma_b * sigma_der) / (sigma_b * np.sqrt(T)) - d2 * sigma_der / sigma_b

    delta = (
        alpha * norm.cdf(d1)
        + SB_0 * norm.pdf(d1) * d1_der
        - K * np.exp(-r * T) * norm.pdf(d2) * d2_der
    )
    return float(delta)

def delta_MC_red_var(t: float, sigma1: float, sigma2: float, beta: float, alpha: float,
                     r: float, K: float, rho: float, N: int, S1_0: float, S2_0: float, h: float = 0.01):
    X, _ = Box_Muller(N)
    W1 = np.sqrt(t) * X

    Z_diff = np.array([
        (
            phi(W1[i], t, sigma1, sigma2, beta * S2_0, alpha * (S1_0 + h), r, K, rho)
            - phi(W1[i], t, sigma1, sigma2, beta * S2_0, alpha * S1_0, r, K, rho)
        ) / h
        for i in range(N)
    ])

    delta_estimate = float(np.mean(Z_diff) * np.exp(-r * t))
    delta_error = float(1.96 * np.std(Z_diff) / np.sqrt(N))
    IC_sup = delta_estimate + delta_error
    IC_inf = delta_estimate - delta_error
    return delta_estimate, delta_error, IC_sup, IC_inf

# =========================================================
# 6) New control variate (FT)
# =========================================================
def Euro_call_new_control(t: float, alpha: float, beta: float, S1_0: float, S2_0: float,
                          alpha_S1_0: float, beta_S2_0: float, r: float,
                          sigma1: float, sigma2: float, rho: float, N: int, K: float):
    # Exact E[(e^{F_T} - K)^+] with F_T Gaussian
    esp = alpha * np.log(S1_0) + beta * np.log(S2_0) \
        + alpha * (r - 0.5 * sigma1**2) * t + beta * (r - 0.5 * sigma2**2) * t

    var = t * ((alpha * sigma1)**2 + (beta * sigma2)**2 + 2 * rho * alpha * sigma1 * beta * sigma2)
    std = np.sqrt(var)
    d = (esp - np.log(K)) / std

    exact_control = np.exp(-r * t) * (
        np.exp(esp + 0.5 * var) * norm.cdf(d + std) - K * norm.cdf(d)
    )

    # MC simulation
    X, Y = Box_Muller(N)
    W1 = np.sqrt(t) * X
    W2 = np.sqrt(t) * (rho * X + np.sqrt(1 - rho**2) * Y)

    S1_T = np.exp((r - 0.5 * sigma1**2) * t + sigma1 * W1)
    S2_T = np.exp((r - 0.5 * sigma2**2) * t + sigma2 * W2)

    F_T = alpha * np.log(S1_0 * S1_T) + beta * np.log(S2_0 * S2_T)

    X_payoff = np.maximum(alpha_S1_0 * S1_T + beta_S2_0 * S2_T - K, 0)
    Y_payoff = np.maximum(np.exp(F_T) - K, 0)

    Yc = X_payoff - Y_payoff

    MC_price = float(np.exp(-r * t) * np.mean(Yc) + exact_control)
    MC_error = float(1.96 * np.std(Yc) / np.sqrt(N))
    IC_sup = MC_price + MC_error
    IC_inf = MC_price - MC_error
    return MC_price, MC_error, IC_inf, IC_sup

# =========================================================
# Runners (sections)
# =========================================================
def run_basics(params, savefig: bool):
    alpha, beta = params["alpha"], params["beta"]
    S1_0, S2_0 = params["S1_0"], params["S2_0"]
    sigma1, sigma2 = params["sigma1"], params["sigma2"]
    rho, T, K, r = params["rho"], params["T"], params["K"], params["r"]

    SB_0 = alpha * S1_0 + beta * S2_0
    sigb = sigma_basket(alpha, beta, S1_0, S2_0, sigma1, sigma2, rho, T)

    prix_approx = call_price(SB_0, sigb, T, r, K)
    prix_exact = call_price_1(SB_0, sigb, T, r, K)

    print("=== Log-normal approximation ===")
    print("Sigma_basket :", sigb)
    print("Prix (A&S aprox) :", prix_approx)
    print("Prix (norm.cdf)  :", prix_exact)
    return prix_approx, sigb

def run_mc_compare(params):
    S1_0, S2_0 = params["S1_0"], params["S2_0"]
    sigma1, sigma2 = params["sigma1"], params["sigma2"]
    rho, T, K, r, N = params["rho"], params["T"], params["K"], params["r"], params["N"]

    Gamma = np.array([[1, rho], [rho, 1]])

    p1, e1, i1, s1 = Euro_call(T, S1_0, S2_0, r, sigma1, sigma2, rho, N, K)
    p2, e2, i2, s2 = Euro_call_chole(T, S1_0, S2_0, r, sigma1, sigma2, rho, N, K, Gamma)
    p3, e3, i3, s3 = Euro_call_red_var(T, sigma1, sigma2, params["beta"], params["alpha"], r, K, rho, N)

    print("\n=== Monte Carlo comparisons ===")
    print(f"MC (rho direct)         : {p1:.6f}  IC95% [{i1:.4f}, {s1:.4f}]")
    print(f"MC (Cholesky)           : {p2:.6f}  IC95% [{i2:.4f}, {s2:.4f}]")
    print(f"MC (Conditioning)       : {p3:.6f}  IC95% [{i3:.4f}, {s3:.4f}]")
    return (p1, p2, p3)

def run_q6_convergence(params, ref_price: float, savefig: bool):
    S1_0, S2_0 = params["S1_0"], params["S2_0"]
    sigma1, sigma2 = params["sigma1"], params["sigma2"]
    rho, T, K, r = params["rho"], params["T"], params["K"], params["r"]

    n_values = params["n_values"]

    mc_prices, mc_infs, mc_sups = [], [], []
    red_prices, red_infs, red_sups = [], [], []

    for n in n_values:
        pmc, _, imc, smc = Euro_call(T, S1_0, S2_0, r, sigma1, sigma2, rho, int(n), K)
        prd, _, ird, srd = Euro_call_red_var(T, sigma1, sigma2, params["beta"], params["alpha"], r, K, rho, int(n))
        mc_prices.append(pmc); mc_infs.append(imc); mc_sups.append(smc)
        red_prices.append(prd); red_infs.append(ird); red_sups.append(srd)

    plt.figure(figsize=(12, 7))
    plt.errorbar(n_values, mc_prices,
                 yerr=[np.array(mc_prices) - np.array(mc_infs), np.array(mc_sups) - np.array(mc_prices)],
                 fmt='o-', capsize=4, label="MC classique (rho direct)")
    plt.errorbar(n_values, red_prices,
                 yerr=[np.array(red_prices) - np.array(red_infs), np.array(red_sups) - np.array(red_prices)],
                 fmt='s-', capsize=4, label="MC réduction de variance (conditionnement)")
    plt.axhline(ref_price, linestyle='--', label=f"Approx. log-normale (≈ {ref_price:.4f})")
    plt.xlabel("Nombre de simulations (N)")
    plt.ylabel("Prix estimé")
    plt.title("Convergence MC vs N (avec IC à 95%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_or_show("q6_convergence_vs_N", savefig)

def run_q7_rho_study(params, savefig: bool):
    alpha, beta = params["alpha"], params["beta"]
    S1_0, S2_0 = params["S1_0"], params["S2_0"]
    sigma1, sigma2 = params["sigma1"], params["sigma2"]
    T, K, r, N = params["T"], params["K"], params["r"], params["N"]

    rho_values = params["rho_values"]

    mc_prices, log_prices, diff = [], [], []
    for rho_i in rho_values:
        pmc, _, _, _ = Euro_call_red_var(T, sigma1, sigma2, beta, alpha, r, K, float(rho_i), N)
        SB0 = alpha * S1_0 + beta * S2_0
        sigb = sigma_basket(alpha, beta, S1_0, S2_0, sigma1, sigma2, float(rho_i), T)
        pln = call_price(SB0, sigb, T, r, K)
        mc_prices.append(pmc); log_prices.append(pln); diff.append(pmc - pln)

    plt.figure(figsize=(12, 6))
    plt.plot(rho_values, mc_prices, label="MC (conditionnement)")
    plt.plot(rho_values, log_prices, linestyle='--', label="Approx. log-normale")
    plt.xlabel("ρ")
    plt.ylabel("Prix")
    plt.title("Prix en fonction de ρ")
    plt.legend(); plt.grid(True); plt.tight_layout()
    save_or_show("q7_price_vs_rho", savefig)

    plt.figure(figsize=(12, 6))
    plt.plot(rho_values, diff, label="Différence (MC - log-normale)")
    plt.axhline(0, linestyle='--')
    plt.xlabel("ρ")
    plt.ylabel("Différence")
    plt.title("Différence (MC - log-normale) en fonction de ρ")
    plt.legend(); plt.grid(True); plt.tight_layout()
    save_or_show("q7_diff_vs_rho", savefig)

def run_q8_alpha_study(params, savefig: bool):
    beta = params["beta"]
    S1_0, S2_0 = params["S1_0"], params["S2_0"]
    sigma1, sigma2 = params["sigma1"], params["sigma2"]
    rho, T, K, r, N = params["rho"], params["T"], params["K"], params["r"], params["N"]

    alpha_values = params["alpha_values"]

    mc_prices, log_prices, diff = [], [], []
    for alpha_i in alpha_values:
        pmc, _, _, _ = Euro_call_red_var(T, sigma1, sigma2, beta, float(alpha_i), r, K, rho, N)

        SB0 = float(alpha_i) * S1_0 + beta * S2_0
        # sigma_b depends on alpha too
        sigb = sigma_basket(float(alpha_i), beta, S1_0, S2_0, sigma1, sigma2, rho, T)
        pln = call_price(SB0, sigb, T, r, K)

        mc_prices.append(pmc); log_prices.append(pln); diff.append(pmc - pln)

    plt.figure(figsize=(12, 6))
    plt.plot(alpha_values, mc_prices, label="MC (conditionnement)")
    plt.plot(alpha_values, log_prices, linestyle='--', label="Approx. log-normale")
    plt.xlabel("α")
    plt.ylabel("Prix")
    plt.title("Prix en fonction de α")
    plt.legend(); plt.grid(True); plt.tight_layout()
    save_or_show("q8_price_vs_alpha", savefig)

    plt.figure(figsize=(12, 6))
    plt.plot(alpha_values, diff, label="Différence (MC - log-normale)")
    plt.axhline(0, linestyle='--')
    plt.xlabel("α")
    plt.ylabel("Différence")
    plt.title("Différence (MC - log-normale) en fonction de α")
    plt.legend(); plt.grid(True); plt.tight_layout()
    save_or_show("q8_diff_vs_alpha", savefig)

def run_q9_K_study(params, sigma_b: float, savefig: bool):
    alpha, beta = params["alpha"], params["beta"]
    S1_0, S2_0 = params["S1_0"], params["S2_0"]
    sigma1, sigma2 = params["sigma1"], params["sigma2"]
    rho, T, r, N = params["rho"], params["T"], params["r"], params["N"]

    K_values = params["K_values"]
    SB_0 = alpha * S1_0 + beta * S2_0

    prices_K, IC_inf, IC_sup = [], [], []
    for K_i in K_values:
        p, _, inf, sup = Euro_call_red_var(T, sigma1, sigma2, beta, alpha, r, float(K_i), rho, N)
        prices_K.append(p); IC_inf.append(inf); IC_sup.append(sup)

    log_prices = [call_price(SB_0, sigma_b, T, r, float(K_i)) for K_i in K_values]

    plt.figure(figsize=(12, 6))
    plt.errorbar(K_values, prices_K,
                yerr=[np.array(prices_K) - np.array(IC_inf), np.array(IC_sup) - np.array(prices_K)],
                fmt='o-', capsize=4, label="MC (conditionnement)")
    plt.plot(K_values, log_prices, linestyle='--', label="Approx. log-normale")
    plt.xlabel("K")
    plt.ylabel("Prix")
    plt.title("Prix en fonction de K (avec IC 95%)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    save_or_show("q9_price_vs_K", savefig)

def run_q10_control_var(params, savefig: bool):
    # Note: here we keep your idea but we reduce grids in FAST mode via params values.
    sigma1, sigma2 = params["sigma1"], params["sigma2"]
    rho, T, K, r, N = params["rho"], params["T"], params["K"], params["r"], params["N"]

    print("\n=== Q10: Control variate (call-put parity) ===")
    MC_price_ctrl, MC_error_ctrl, IC_sup_ctrl, IC_inf_ctrl = Euro_call_controle(
        T, params["alpha"], params["beta"], r, sigma1, sigma2, rho, N, K
    )
    print("Prix (contrôle call-put) :", MC_price_ctrl)
    print("Erreur MC                :", MC_error_ctrl)
    print("IC95%                    :", (IC_inf_ctrl, IC_sup_ctrl))

    # variance vs alpha
    alpha_values = params["alpha_values_q10"]
    var_put, var_call = [], []
    for a in alpha_values:
        _, err_put, _, _ = Euro_call_controle(T, float(a), params["beta"], r, sigma1, sigma2, rho, N, K)
        _, err_call, _, _ = Euro_call(T, float(a), params["beta"], r, sigma1, sigma2, rho, N, K)
        # invert the 1.96/sqrt(N) scaling to get stdev proxy (like your code)
        var_put.append(err_put * np.sqrt(N) / 1.96)
        var_call.append(err_call * np.sqrt(N) / 1.96)

    plt.figure()
    plt.plot(alpha_values, var_put, label="Std proxy put Y")
    plt.plot(alpha_values, var_call, label="Std proxy call X")
    plt.title("Q10 - Variance proxy vs α")
    plt.xlabel("α")
    plt.ylabel("Std proxy")
    plt.legend(); plt.grid(True)
    save_or_show("q10_variance_vs_alpha", savefig)

    # variance vs K
    K_values = params["K_values_q10"]
    var_put_k, var_call_k = [], []
    for k in K_values:
        _, err_put, _, _ = Euro_call_controle(T, params["alpha"], params["beta"], r, sigma1, sigma2, rho, N, float(k))
        _, err_call, _, _ = Euro_call(T, params["alpha"], params["beta"], r, sigma1, sigma2, rho, N, float(k))
        var_put_k.append(err_put * np.sqrt(N) / 1.96)
        var_call_k.append(err_call * np.sqrt(N) / 1.96)

    plt.figure()
    plt.plot(K_values, var_put_k, label="Std proxy put Y")
    plt.plot(K_values, var_call_k, label="Std proxy call X")
    plt.title("Q10 - Variance proxy vs K")
    plt.xlabel("K")
    plt.ylabel("Std proxy")
    plt.legend(); plt.grid(True)
    save_or_show("q10_variance_vs_K", savefig)

    # variance vs rho
    rho_values = params["rho_values_q10"]
    var_put_r, var_call_r = [], []
    for rr in rho_values:
        _, err_put, _, _ = Euro_call_controle(T, params["alpha"], params["beta"], r, sigma1, sigma2, float(rr), N, K)
        _, err_call, _, _ = Euro_call(T, params["alpha"], params["beta"], r, sigma1, sigma2, float(rr), N, K)
        var_put_r.append(err_put * np.sqrt(N) / 1.96)
        var_call_r.append(err_call * np.sqrt(N) / 1.96)

    plt.figure()
    plt.plot(rho_values, var_put_r, label="Std proxy put Y")
    plt.plot(rho_values, var_call_r, label="Std proxy call X")
    plt.title("Q10 - Variance proxy vs ρ")
    plt.xlabel("ρ")
    plt.ylabel("Std proxy")
    plt.legend(); plt.grid(True)
    save_or_show("q10_variance_vs_rho", savefig)

def run_q11_delta(params, savefig: bool):
    sigma1, sigma2 = params["sigma1"], params["sigma2"]
    T, r, N = params["T"], params["r"], params["N"]
    S2_0 = params["S2_0"]
    alpha, beta = params["alpha_delta"], params["beta_delta"]
    K = params["K_delta"]

    S1_values = params["S1_grid_delta"]
    rho_pos, rho_neg = params["rho_pos_delta"], params["rho_neg_delta"]

    # log-normal delta
    dpos_ln, dneg_ln = [], []
    for s1 in S1_values:
        dpos_ln.append(delta_approx_log_normale(float(s1), S2_0, alpha, beta, sigma1, sigma2, rho_pos, T, r, K))
        dneg_ln.append(delta_approx_log_normale(float(s1), S2_0, alpha, beta, sigma1, sigma2, rho_neg, T, r, K))

    plt.figure()
    plt.plot(S1_values, dpos_ln, label="log-normal, ρ=+")
    plt.plot(S1_values, dneg_ln, label="log-normal, ρ=-")
    plt.title("Q11 - Δ1 (log-normal)")
    plt.xlabel("S1,0")
    plt.ylabel("Delta")
    plt.legend(); plt.grid(True)
    save_or_show("q11_delta_lognormal", savefig)

    # MC delta (finite diff)
    dpos_mc, dneg_mc = [], []
    for s1 in S1_values:
        d1, _, _, _ = delta_MC_red_var(T, sigma1, sigma2, beta, alpha, r, K, rho_pos, N, float(s1), S2_0)
        d2, _, _, _ = delta_MC_red_var(T, sigma1, sigma2, beta, alpha, r, K, rho_neg, N, float(s1), S2_0)
        dpos_mc.append(d1); dneg_mc.append(d2)

    plt.figure()
    plt.plot(S1_values, dpos_mc, label="MC cond, ρ=+")
    plt.plot(S1_values, dneg_mc, label="MC cond, ρ=-")
    plt.title("Q11 - Δ1 (MC conditionnement, dérivée numérique)")
    plt.xlabel("S1,0")
    plt.ylabel("Delta")
    plt.legend(); plt.grid(True)
    save_or_show("q11_delta_mc", savefig)

    # diff
    plt.figure()
    plt.plot(S1_values, np.array(dpos_ln) - np.array(dpos_mc), label="Diff ρ=+")
    plt.plot(S1_values, np.array(dneg_ln) - np.array(dneg_mc), label="Diff ρ=-")
    plt.axhline(0, linestyle='--')
    plt.title("Q11 - Différence (log-normal - MC)")
    plt.xlabel("S1,0")
    plt.ylabel("Différence")
    plt.legend(); plt.grid(True)
    save_or_show("q11_delta_diff", savefig)

def run_q12_new_control(params):
    print("\n=== Q12: New control variate (F_T) ===")
    sigma1, sigma2 = params["sigma1"], params["sigma2"]
    rho, T, r, N = params["rho"], params["T"], params["r"], params["N"]

    # Typical config from your report/code
    alpha = params["alpha_q12"]
    beta = params["beta_q12"]
    S1_0 = params["S1_0_q12"]
    S2_0 = params["S2_0_q12"]
    alphaS10 = params["alphaS10_q12"]
    betaS20 = params["betaS20_q12"]
    K = params["K_q12"]

    p, e, inf, sup = Euro_call_new_control(T, alpha, beta, S1_0, S2_0, alphaS10, betaS20, r, sigma1, sigma2, rho, N, K)
    print("MC_price :", p)
    print("MC_error :", e)
    print(f"IC95%    : [{inf:.4f}, {sup:.4f}]")

# =========================================================
# Main
# =========================================================
def build_params(fast: bool):
    # Base params (your defaults)
    params = dict(
        alpha=1.0, beta=1.0,
        S1_0=1.0, S2_0=1.0,
        sigma1=0.35, sigma2=0.4,
        rho=0.3, T=2.0, K=2.0, r=0.01
    )

    # Runtime knobs
    if fast:
        params["N"] = 20000
        params["n_values"] = np.arange(2000, 30001, 4000)
        params["rho_values"] = np.linspace(-0.99, 0.99, 25)
        params["alpha_values"] = np.linspace(0.0, 2.0, 25)
        params["K_values"] = np.linspace(1.0, 3.0, 11)

        params["alpha_values_q10"] = np.linspace(0.0, 2.0, 25)
        params["K_values_q10"] = np.linspace(1.0, 3.0, 11)
        params["rho_values_q10"] = np.linspace(-0.99, 0.99, 25)

        params["S1_grid_delta"] = np.linspace(0.0, 4.0, 25)
        params["alpha_delta"] = 0.5
        params["beta_delta"] = 0.5
        params["K_delta"] = 2.0
        params["rho_pos_delta"] = 0.5
        params["rho_neg_delta"] = -0.5

    else:
        params["N"] = 100000
        params["n_values"] = np.arange(1000, 100001, 5000)
        params["rho_values"] = np.linspace(-0.99, 0.99, 100)
        params["alpha_values"] = np.linspace(0.0, 2.0, 100)
        params["K_values"] = np.linspace(1.0, 3.0, 15)

        params["alpha_values_q10"] = np.linspace(0.0, 2.0, 100)
        params["K_values_q10"] = np.linspace(1.0, 3.0, 15)
        params["rho_values_q10"] = np.linspace(-0.99, 0.99, 100)

        params["S1_grid_delta"] = np.linspace(0.0, 4.0, 100)
        params["alpha_delta"] = 0.5
        params["beta_delta"] = 0.5
        params["K_delta"] = 2.0
        params["rho_pos_delta"] = 0.5
        params["rho_neg_delta"] = -0.5

    # Q12 config (as in your report example)
    params["alpha_q12"] = 0.5
    params["beta_q12"] = 0.5
    params["S1_0_q12"] = 2.0
    params["S2_0_q12"] = 2.0
    params["alphaS10_q12"] = 1.0
    params["betaS20_q12"] = 1.0
    params["K_q12"] = 2.0

    return params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run the full (slow) experiment.")
    parser.add_argument("--fast", action="store_true", help="Run the fast demo (default).")
    parser.add_argument("--savefig", action="store_true", help="Save figures to ./figures instead of showing them.")
    parser.add_argument("--section", type=str, default="all",
                        help="Which section to run: all, basics, mc, q6, q7, q8, q9, q10, q11, q12")
    args = parser.parse_args()

    # Default to fast unless --full is specified
    fast = True
    if args.full:
        fast = False
    if args.fast:
        fast = True

    params = build_params(fast=fast)

    t0 = time.time()
    print(f"Mode: {'FAST' if fast else 'FULL'} | N={params['N']} | savefig={args.savefig} | section={args.section}")

    ref_price, sigb = run_basics(params, savefig=args.savefig)

    if args.section in ("all", "mc"):
        run_mc_compare(params)

    if args.section in ("all", "q6"):
        run_q6_convergence(params, ref_price=ref_price, savefig=args.savefig)

    if args.section in ("all", "q7"):
        run_q7_rho_study(params, savefig=args.savefig)

    if args.section in ("all", "q8"):
        run_q8_alpha_study(params, savefig=args.savefig)

    if args.section in ("all", "q9"):
        run_q9_K_study(params, sigma_b=sigb, savefig=args.savefig)

    if args.section in ("all", "q10"):
        run_q10_control_var(params, savefig=args.savefig)

    if args.section in ("all", "q11"):
        run_q11_delta(params, savefig=args.savefig)

    if args.section in ("all", "q12"):
        run_q12_new_control(params)

    print(f"Done in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
