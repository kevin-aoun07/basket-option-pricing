import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import cholesky
np.random.seed(0)
###########################################################################
######################## Prix via approximation log-normale ###############
###########################################################################

def aprox(x):
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

def call_price(S_0, sigma, T, r, K):
    d = (np.log(K/S_0) - (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S_0 * aprox(-d + sigma * np.sqrt(T)) - K * np.exp(-r * T) * aprox(-d)

def call_price_1(S_0, sigma_b, T, r, K):
    d = (np.log(K/S_0) - (r - 0.5 * sigma_b**2) * T) / (sigma_b * np.sqrt(T))
    return S_0 * norm.cdf(-d + sigma_b * np.sqrt(T)) - K * np.exp(-r * T) * norm.cdf(-d)

###################################################################
########################## Paramètres #############################
###################################################################

alpha = 1
beta = 1
S1_0 = 1
S2_0 = 1
sigma1 = 0.35
sigma2 = 0.4
rho = 0.3
T = 2
K = 2
r = 0.01
N = 100000

SB_0 = alpha * S1_0 + beta * S2_0

numerateur = (
    (alpha**2 * S1_0**2 / SB_0**2) * np.exp(sigma1**2 * T) +
    (beta**2 * S2_0**2 / SB_0**2) * np.exp(sigma2**2 * T) +
    (2 * alpha * beta * S1_0 * S2_0 / SB_0**2) * np.exp(sigma1 * sigma2 * rho * T)
)
sigma_b_squared = (1 / T) * np.log(numerateur)
sigma_b = np.sqrt(sigma_b_squared)

prix_approx = call_price(SB_0, sigma_b, T, r, K)
prix_exact = call_price_1(SB_0, sigma_b, T, r, K)

print("Prix (approximation Abramowitz & Stegun) :", prix_approx)
print("Prix (avec norm.cdf)                     :", prix_exact)


#################################################################################
######## Monte Carlo - Corrélation analytique (Box-Muller + rho) ################
#################################################################################

def Box_Muller(n):
    U = npr.random(n)
    V = npr.random(n)
    X = np.sqrt(-2 * np.log(U)) * np.cos(2 * np.pi * V)
    Y = np.sqrt(-2 * np.log(U)) * np.sin(2 * np.pi * V)
    return X, Y

def Euro_call(t, alpha_S_1_0, beta_S_2_0, r, sigma_1, sigma_2, rho, N, K):
    X, Y = Box_Muller(N)
    W1 = X
    W2 = rho * X + np.sqrt(1 - rho**2) * Y

    S_T = (
        alpha_S_1_0 * np.exp((r - 0.5 * sigma_1**2) * t + sigma_1 * np.sqrt(t) * W1) +
        beta_S_2_0 * np.exp((r - 0.5 * sigma_2**2) * t + sigma_2 * np.sqrt(t) * W2)
    )

    payoff = np.exp(-r * t) * np.maximum(S_T - K, 0)
    MC_price = np.mean(payoff)
    MC_error = 1.96 * np.std(payoff) / np.sqrt(N)
    IC_sup = MC_price + MC_error
    IC_inf = MC_price - MC_error

    return MC_price, MC_error, IC_inf, IC_sup

###########################################################################
################ Monte Carlo - Cholesky + Box-Muller ######################
###########################################################################

def Gauss_Correl(N, Gamma):
    A = cholesky(Gamma)
    X, Y = Box_Muller(N)
    G = np.vstack([X, Y])
    Z = np.dot(A,G)
    return Z

def Euro_call_chole(t, alpha_S_1_0, beta_S_2_0, r, sigma_1, sigma_2, rho, N, K, Gamma):
    Z = Gauss_Correl(N, Gamma)
    W1, W2 = Z[0], Z[1]

    S_T = (
        alpha_S_1_0 * np.exp((r - 0.5 * sigma_1**2) * t + sigma_1 * np.sqrt(t) * W1) +
        beta_S_2_0 * np.exp((r - 0.5 * sigma_2**2) * t + sigma_2 * np.sqrt(t) * W2)
    )

    payoff = np.exp(-r * t) * np.maximum(S_T - K, 0)
    MC_price = np.mean(payoff)
    MC_error = 1.96 * np.std(payoff) / np.sqrt(N)
    IC_sup = MC_price + MC_error
    IC_inf = MC_price - MC_error

    return MC_price, MC_error, IC_inf, IC_sup

Gamma = np.array([[1, rho], [rho, 1]])

prix_MC1, err1, inf1, sup1 = Euro_call(T, S1_0, S2_0, r, sigma1, sigma2, rho, N, K)
prix_MC2, err2, inf2, sup2 = Euro_call_chole(T, S1_0, S2_0, r, sigma1, sigma2, rho, N, K, Gamma)

print("Prix (Monte Carlo - rho direct)         :", prix_MC1)
print("IC à 95% : [{:.4f}, {:.4f}]".format(inf1, sup1))
print("Prix (Monte Carlo - Cholesky)           :", prix_MC2)
print("IC à 95% : [{:.4f}, {:.4f}]".format(inf2, sup2))


##############################################################################################
### Calcul du prix par la méthode de MONTE CARLO par REDUCTION DE VARIANCE sur CONDITIONNEMENT
##############################################################################################

def phi(x, t, sigma1, sigma2, beta, alpha, r, K, rho):
    S_cond = beta * np.exp(-0.5 * (sigma2 * rho)**2 * t + sigma2 * rho * x)
    sigma_cond = sigma2 * np.sqrt(1 - rho**2)
    K_cond = K - alpha * np.exp((r - 0.5 * sigma1**2) * t + sigma1 * x)

    if K_cond > 0:
        return np.exp(r * t) * call_price(S_cond, sigma_cond, t, r, K_cond)
    else:
        return S_cond * np.exp(r * t) - K_cond

def Euro_call_red_var(t, sigma1, sigma2, beta, alpha, r, K, rho, N):
    X, Y = Box_Muller(N)
    W1 = np.sqrt(t) * X
    Z = np.array([phi(W1[i], t, sigma1, sigma2, beta, alpha, r, K, rho) for i in range(N)])
    MC_price = np.exp(-r * t) * np.mean(Z)
    MC_error = 1.96 * np.std(Z) / np.sqrt(N)
    IC_sup = MC_price + MC_error
    IC_inf = MC_price - MC_error
    return MC_price, MC_error, IC_inf, IC_sup


prix_MC3, err3, inf3, sup3 = Euro_call_red_var(T, sigma1, sigma2, beta, alpha, r, K, rho, N)

print("Prix (Monte Carlo - réduction variance) :", prix_MC3)
print("IC à 95% : [{:.4f}, {:.4f}]".format(inf3, sup3))

######################################################################
##################  COMPARAISON DES METHODES MC ######################
######################################################################

print("\nRésumé des méthodes Monte Carlo :")
print("Méthode                        Prix        Erreur MC     Intervalle 95%")
print(f"Corrélation directe        : {prix_MC1:.6f}  {err1:.6f}    [{inf1:.4f}, {sup1:.4f}]")
print(f"Cholesky                   : {prix_MC2:.6f}  {err2:.6f}    [{inf2:.4f}, {sup2:.4f}]")
print(f"Réduction de variance      : {prix_MC3:.6f}  {err3:.6f}    [{inf3:.4f}, {sup3:.4f}]")

#########################################################################################################
##################  Graphique des Estimations avec IC des differentes methodes MC  ######################
#########################################################################################################

import matplotlib.pyplot as plt

# Pour Q6 – analyse en fonction de N
n_values = np.arange(1000, 100001, 5000)

# Stockage des résultats
mc_prices = []
mc_errors = []
mc_infs = []
mc_sups = []

redvar_prices = []
redvar_errors = []
redvar_infs = []
redvar_sups = []

# Boucle sur les différentes tailles d’échantillons
for n in n_values:
    # Méthode Monte Carlo classique
    price_mc, error_mc, inf_mc, sup_mc = Euro_call(T, S1_0, S2_0, r, sigma1, sigma2, rho, n, K)
    mc_prices.append(price_mc)
    mc_errors.append(error_mc)
    mc_infs.append(inf_mc)
    mc_sups.append(sup_mc)

    # Méthode réduction de variance
    price_red, error_red, inf_red, sup_red = Euro_call_red_var(T, sigma1, sigma2, beta, alpha, r, K, rho, n)
    redvar_prices.append(price_red)
    redvar_errors.append(error_red)
    redvar_infs.append(inf_red)
    redvar_sups.append(sup_red)

# Prix de référence par approximation log-normale
ref_price = prix_approx

# Tracé
plt.figure(figsize=(12, 7))

# Monte Carlo classique
plt.errorbar(n_values, mc_prices, yerr=[np.array(mc_prices) - np.array(mc_infs), np.array(mc_sups) - np.array(mc_prices)],
             fmt='o-', capsize=5, label="MC classique (corrélation directe)", color='blue')

# Monte Carlo réduction de variance
plt.errorbar(n_values, redvar_prices, yerr=[np.array(redvar_prices) - np.array(redvar_infs), np.array(redvar_sups) - np.array(redvar_prices)],
             fmt='s-', capsize=5, label="MC réduction de variance", color='green')

# Prix théorique
plt.axhline(ref_price, color='red', linestyle='--', label=f"Approx. log-normale (prix ≈ {ref_price:.4f})")

# Mise en forme
plt.xlabel("Nombre de simulations (N)")
plt.ylabel("Prix estimé")
plt.title("Prix estimé de l'option en fonction du nombre de trajectoires")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


##############################################################################################
### Question 7 – Étude du prix en fonction de ρ avec MC réduction de variance et log-normale
##############################################################################################

# Liste des valeurs de rho
rho_values = np.linspace(-0.99, 0.99, 100)

# Stockage des prix
mc_red_prices = []
lognorm_prices = []
differences = []

for rho_i in rho_values:
    # Prix Monte Carlo avec réduction de variance
    price_mc_red, _, _, _ = Euro_call_red_var(T, sigma1, sigma2, beta, alpha, r, K, rho_i, N)
    mc_red_prices.append(price_mc_red)

    # Calcul de SB_0 (reste constant) et sigma_b pour cette valeur de rho
    SB_0_i = alpha * S1_0 + beta * S2_0
    num = (
        (alpha**2 * S1_0**2 / SB_0_i**2) * np.exp(sigma1**2 * T) +
        (beta**2 * S2_0**2 / SB_0_i**2) * np.exp(sigma2**2 * T) +
        (2 * alpha * beta * S1_0 * S2_0 / SB_0_i**2) * np.exp(sigma1 * sigma2 * rho_i * T)
    )
    sigma_b_sq = (1 / T) * np.log(num)
    sigma_b_i = np.sqrt(sigma_b_sq)

    price_lognorm = call_price(SB_0_i, sigma_b_i, T, r, K)
    lognorm_prices.append(price_lognorm)

    differences.append(price_mc_red - price_lognorm)

# Tracé des prix
plt.figure(figsize=(12, 6))
plt.plot(rho_values, mc_red_prices, label="MC réduction de variance", color='green')
plt.plot(rho_values, lognorm_prices, label="Approximation log-normale", color='red', linestyle='--')
plt.xlabel("ρ (corrélation)")
plt.ylabel("Prix de l'option")
plt.title("Prix estimé de l'option en fonction de ρ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Tracé de la différence entre les deux
plt.figure(figsize=(12, 6))
plt.plot(rho_values, differences, label="Différence (MC - log-normale)", color='blue')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("ρ (corrélation)")
plt.ylabel("Différence des estimations")
plt.title("Différence entre MC (réduction variance) et approximation log-normale")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



##############################################################################################
### Question 8 – Étude du prix en fonction de α (S1_0), avec MC réduction de variance et log-normale
##############################################################################################

# Plage de valeurs pour alpha
alpha_values = np.linspace(0, 2, 100)

# Stockage des prix
mc_red_prices_alpha = []
lognorm_prices_alpha = []
differences_alpha = []

for alpha_i in alpha_values:
    # Prix Monte Carlo avec réduction de variance (alpha change, le reste reste constant)
    price_mc_red, _, _, _ = Euro_call_red_var(T, sigma1, sigma2, beta, alpha_i, r, K, rho, N)
    mc_red_prices_alpha.append(price_mc_red)

    # Nouveau SB_0
    SB_0_i = alpha_i * S1_0 + beta * S2_0

    # Nouveau sigma_b pour ce alpha
    num = (
        (alpha_i**2 * S1_0**2 / SB_0_i**2) * np.exp(sigma1**2 * T) +
        (beta**2 * S2_0**2 / SB_0_i**2) * np.exp(sigma2**2 * T) +
        (2 * alpha_i * beta * S1_0 * S2_0 / SB_0_i**2) * np.exp(sigma1 * sigma2 * rho * T)
    )
    sigma_b_sq = (1 / T) * np.log(num)
    sigma_b_i = np.sqrt(sigma_b_sq)

    price_lognorm = call_price(SB_0_i, sigma_b_i, T, r, K)
    lognorm_prices_alpha.append(price_lognorm)

    differences_alpha.append(price_mc_red - price_lognorm)

# Tracé des prix
plt.figure(figsize=(12, 6))
plt.plot(alpha_values, mc_red_prices_alpha, label="MC réduction de variance", color='green')
plt.plot(alpha_values, lognorm_prices_alpha, label="Approximation log-normale", color='red', linestyle='--')
plt.xlabel("α (coefficient de S1_0)")
plt.ylabel("Prix de l'option")
plt.title("Prix estimé de l'option en fonction de α")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Tracé de la différence
plt.figure(figsize=(12, 6))
plt.plot(alpha_values, differences_alpha, label="Différence (MC - log-normale)", color='blue')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("α (coefficient de S1_0)")
plt.ylabel("Différence des estimations")
plt.title("Différence entre MC (réduction variance) et approximation log-normale selon α")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



##############################################################################################
### Question 9 – Étude du prix estimé (réduction de variance) + IC en fonction de K
##############################################################################################

# Plage de valeurs pour K
K_values = np.linspace(1, 3, 15)

# Stockage des résultats
prices_K = []
errors_K = []
IC_inf_K = []
IC_sup_K = []

# Estimations Monte Carlo avec réduction de variance
for K_i in K_values:
    price, error, inf, sup = Euro_call_red_var(T, sigma1, sigma2, beta, alpha, r, K_i, rho, N)
    prices_K.append(price)
    errors_K.append(error)
    IC_inf_K.append(inf)
    IC_sup_K.append(sup)

# Estimations log-normales (utilisent SB_0 et sigma_b fixés par alpha, beta, etc.)
call_prices_K = [call_price(SB_0, sigma_b, T, r, K_i) for K_i in K_values]

# Tracé
plt.figure(figsize=(12, 6))
plt.errorbar(K_values, prices_K,
             yerr=[np.array(prices_K) - np.array(IC_inf_K), np.array(IC_sup_K) - np.array(prices_K)],
             fmt='o-', capsize=5, label="MC réduction de variance", color='green')

plt.plot(K_values, call_prices_K, 'r--', label="Approximation log-normale")

plt.xlabel("K")
plt.ylabel("Prix estimé")
plt.title("Prix estimé de l'option et IC à 95% en fonction de K")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



##############################################################################################
### Question 10 – Réduction de variance avec variable de contrôle (call-put)
##############################################################################################

# ----------------------------------------------------------------------------------------
# Fonction : Estimation du prix de l'option via variable de contrôle (relation call-put)
# Utilise : Y = payoff_put comme variable de contrôle pour X = payoff_call
# ----------------------------------------------------------------------------------------
def Euro_call_controle(t, alpha_S_1_0, beta_S_2_0, r, sigma_1, sigma_2, rho, N, K):
    X, Y = Box_Muller(N)
    W1 = X
    W2 = rho * X + np.sqrt(1 - rho**2) * Y
   
    S_T = (
        alpha_S_1_0 * np.exp((r - 0.5 * sigma_1**2) * t + sigma_1 * np.sqrt(t) * W1) +
        beta_S_2_0 * np.exp((r - 0.5 * sigma_2**2) * t + sigma_2 * np.sqrt(t) * W2)
    )
   
    payoff_put = np.exp(-r * t) * np.maximum(K - S_T, 0)
   
    # Prix ajusté via parité call-put
    MC_price = alpha_S_1_0 + beta_S_2_0 - K * np.exp(-r * t) + np.mean(payoff_put)
   
    # Intervalle de confiance à 95%
    MC_error = 1.96 * np.std(payoff_put) / np.sqrt(N)
    IC_sup = MC_price + MC_error
    IC_inf = MC_price - MC_error

    return MC_price, MC_error, IC_sup, IC_inf


# ----------------------------------------------------------------------------------------
# Affichage du prix obtenu avec réduction de variance par variable de contrôle
# ----------------------------------------------------------------------------------------
MC_price_ctrl, MC_error_ctrl, IC_sup_ctrl, IC_inf_ctrl = Euro_call_controle(T, alpha, beta, r, sigma1, sigma2, rho, N, K)

print("Résultats Monte Carlo Réduction de variance avec variable de contrôle")
print("Prix MC Red Var Contrôle :", MC_price_ctrl)
print("Erreur Red Var Contrôle  :", MC_error_ctrl)
print("IC_sup :", IC_sup_ctrl)
print("IC_inf :", IC_inf_ctrl)


# ----------------------------------------------------------------------------------------
# Analyse de la variance en fonction de alpha (α)
# Objectif : vérifier si Var(put) < Var(call) pour différentes valeurs de α
# ----------------------------------------------------------------------------------------
alpha_values = np.linspace(0, 2, 100)
MC_error_VC1 = []
MC_error_VC3 = []

for alpha_S_1_0 in alpha_values:
    _, MC_error_pt, _, _ = Euro_call_controle(T, alpha_S_1_0, beta, r, sigma1, sigma2, rho, N, K)
    _, MC_error_tt, _, _ = Euro_call(T, alpha_S_1_0, beta, r, sigma1, sigma2, rho, N, K)
   
    MC_error_VC1.append(MC_error_pt * np.sqrt(N) / 1.96)
    MC_error_VC3.append(MC_error_tt * np.sqrt(N) / 1.96)

plt.plot(alpha_values, MC_error_VC1, label='Variance du payoff du put Y')
plt.plot(alpha_values, MC_error_VC3, label='Variance du payoff du call X')
plt.title('Évolution de la variance en fonction de α (alpha_S_1_0)')
plt.xlabel('α (alpha_S_1_0)')
plt.ylabel('Variance')
plt.legend()
plt.grid(True)
plt.show()


# ----------------------------------------------------------------------------------------
# Analyse de la variance en fonction de K (strike)
# Objectif : vérifier si Var(put) < Var(call) selon K
# ----------------------------------------------------------------------------------------
K_values = np.linspace(1, 3, 15)
MC_error_VCK1 = []
MC_error_VCK3 = []

for KV in K_values:
    _, MC_error_pt, _, _ = Euro_call_controle(T, alpha, beta, r, sigma1, sigma2, rho, N, KV)
    _, MC_error_tt, _, _ = Euro_call(T, alpha, beta, r, sigma1, sigma2, rho, N, KV)
   
    MC_error_VCK1.append(MC_error_pt * np.sqrt(N) / 1.96)
    MC_error_VCK3.append(MC_error_tt * np.sqrt(N) / 1.96)

plt.plot(K_values, MC_error_VCK1, label='Variance du payoff du put Y')
plt.plot(K_values, MC_error_VCK3, label='Variance du payoff du call X')
plt.title('Évolution de la variance en fonction de K')
plt.xlabel('K')
plt.ylabel('Variance')
plt.legend()
plt.grid(True)
plt.show()


# ----------------------------------------------------------------------------------------
# Analyse de la variance en fonction de ρ (corrélation)
# Objectif : vérifier si Var(put) < Var(call) selon ρ
# ----------------------------------------------------------------------------------------
rho_values = np.linspace(-0.99, 0.99, 100)
MC_error_VCrho1 = []
MC_error_VCrho3 = []

for i in rho_values:
    _, MC_error_pt, _, _ = Euro_call_controle(T, alpha, beta, r, sigma1, sigma2, i, N, K)
    _, MC_error_tt, _, _ = Euro_call(T, alpha, beta, r, sigma1, sigma2, i, N, K)
   
    MC_error_VCrho1.append(MC_error_pt * np.sqrt(N) / 1.96)
    MC_error_VCrho3.append(MC_error_tt * np.sqrt(N) / 1.96)

plt.plot(rho_values, MC_error_VCrho1, label='Variance du payoff du put Y')
plt.plot(rho_values, MC_error_VCrho3, label='Variance du payoff du call X')
plt.title('Évolution de la variance en fonction de ρ')
plt.xlabel('ρ')
plt.ylabel('Variance')
plt.legend()
plt.grid(True)
plt.show()


##############################################################################################
### Question 11 – Calcule de delta
##############################################################################################


# ----------------------------------------------------------------------------------------
# calcule pour l approximation log-normale
# ----------------------------------------------------------------------------------------

def delta_approx_log_normale(S1_0, S2_0, alpha, beta, sigma1, sigma2, rho, T, r, K):

    # SB(0)
    SB_0 = alpha * S1_0 + beta * S2_0

    # σ_B^2
    f = (
        (alpha**2 * S1_0**2 / SB_0**2) * np.exp(sigma1**2 * T) +
        (beta**2 * S2_0**2 / SB_0**2) * np.exp(sigma2**2 * T) +
        (2 * alpha * beta * S1_0 * S2_0 / SB_0**2) * np.exp(sigma1 * sigma2 * rho * T)
    )
    sigma_b_sq = (1 / T) * np.log(f)
    sigma_b = np.sqrt(sigma_b_sq)

    # d1, d2
    d1 = (np.log(SB_0 / K) + (r + 0.5 * sigma_b**2) * T) / (sigma_b * np.sqrt(T))
    d2 = d1 - sigma_b * np.sqrt(T)

    # dérivée de f (intermédiaire pour dσ_B/dS1_0)
    def df(S1_0, S2_0, alpha, beta, sigma1, sigma2, rho, T):
        SB_0 = alpha * S1_0 + beta * S2_0
        term1 = -2 * alpha**3 * S1_0**2 * np.exp(sigma1**2 * T) / SB_0**3
        term2 = -4 * alpha**2 * beta * S1_0 * S2_0 * np.exp(sigma1 * sigma2 * rho * T) / SB_0**3
        term3 =  2 * alpha**2 * S1_0 * np.exp(sigma1**2 * T) / SB_0**2
        term4 = -2 * alpha * beta**2 * S2_0**2 * np.exp(sigma2**2 * T) / SB_0**3
        term5 =  2 * alpha * beta * S2_0 * np.exp(sigma1 * sigma2 * rho * T) / SB_0**2
        return term1 + term2 + term3 + term4 + term5

    # dérivée de σ_B
    f_der = df(S1_0, S2_0, alpha, beta, sigma1, sigma2, rho, T)
    sigma_der = (f_der) / (2 * f * sigma_b * T)

    # dérivées de d1 et d2
    d1_der = (alpha / SB_0 + T * sigma_b * sigma_der) / (sigma_b * np.sqrt(T)) - d1 * sigma_der / sigma_b
    d2_der = (alpha / SB_0 - T * sigma_b * sigma_der) / (sigma_b * np.sqrt(T)) - d2 * sigma_der / sigma_b

    # Calcul du delta (∂P/∂S1_0)
    delta = (
        alpha * norm.cdf(d1)
        + SB_0 * norm.pdf(d1) * d1_der
        - K * np.exp(-r * T) * norm.pdf(d2) * d2_der
    )

    return delta

####création de liste
S_1_0_values = np.linspace(0, 4, 100)
delta_values1 = []
delta_values2 = []
rho1 = 0.5
rho2 = -0.5

for S_1_0 in S_1_0_values:

    delta_rho1 =delta_approx_log_normale(S_1_0, 1, 0.5, 0.5, 0.35, 0.4, rho1, 2, 0.01, 2)

    delta_rho2 =delta_approx_log_normale(S_1_0, 1, 0.5, 0.5, 0.35, 0.4, rho2, 2, 0.01, 2)

    delta_values1.append(delta_rho1)
    delta_values2.append(delta_rho2)

plt.plot(S_1_0_values, delta_values1, label='rho=0.5')
plt.plot(S_1_0_values, delta_values2, label='rho=-0.5')
plt.title('Variation delta_1 en fonction de S_1_0 par méthode log-normale')
plt.xlabel('S_1_0')
plt.ylabel('Delta')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------------------------
# calcule pour la methode MC avec conditionnement
# ----------------------------------------------------------------------------------------


def delta_MC_red_var(t, sigma1, sigma2, beta, alpha, r, K, rho, N, S1_0, S2_0, h=0.01):
    X, Y = Box_Muller(N)
    W1 = np.sqrt(t) * X

    # Calcul de la dérivée numérique ∂P/∂S1_0 ≈ (P(S1_0 + h) - P(S1_0)) / h
    Z_diff = np.array([
        (
            phi(W1[i], t, sigma1, sigma2, beta * S2_0, alpha * (S1_0 + h), r, K, rho)
            - phi(W1[i], t, sigma1, sigma2, beta * S2_0, alpha * S1_0, r, K, rho)
        ) / h
        for i in range(N)
    ])

    delta_estimate = np.mean(Z_diff) * np.exp(-r * t)
    delta_error = 1.96 * np.std(Z_diff) / np.sqrt(N)
    IC_sup = delta_estimate + delta_error
    IC_inf = delta_estimate - delta_error

    return delta_estimate, delta_error, IC_sup, IC_inf

#Initialisation
S_1_0_values= np.linspace(0, 4, 100)
delta_values_1 = []
delta_values_2 = []

#Données imposées
rho1= 0.5
rho2=-0.5
K=2
T=2

#Appeler la fonction pour chaque valeur de rho et stocker delta
for S_1_0 in S_1_0_values:
    delta_1, MC_error, IC_sup, IC_inf= delta_MC_red_var(T, sigma1, sigma2, 0.5, 0.5, r, K, rho1, N, S_1_0, S2_0)
    delta_2, MC_error, IC_sup, IC_inf= delta_MC_red_var(T, sigma1, sigma2, 0.5, 0.5, r, K, rho2, N, S_1_0, S2_0)
   
    delta_values_1.append(delta_1)
    delta_values_2.append(delta_2)
   

# Tracer MC_price en fonction de rho
plt.plot(S_1_0_values, delta_values_1, label='delta avec rho = 0.5')
plt.plot(S_1_0_values, delta_values_2, label='delta avec rho = -0.5')

plt.title('Variation de delta_1 en fonction de S_1_0 ')
plt.xlabel('S_1_0')
plt.ylabel('Delta')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------------------------
# Comparaison des deux méthodes : différence des estimateurs de delta
# ----------------------------------------------------------------------------------------

# Calcul des différences (log-normale - MC)
diff_rho_pos = np.array(delta_values1) - np.array(delta_values_1)
diff_rho_neg = np.array(delta_values2) - np.array(delta_values_2)

# Tracé des différences
plt.plot(S_1_0_values, diff_rho_pos, label='Différence pour ρ = 0.5', color='blue')
plt.plot(S_1_0_values, diff_rho_neg, label='Différence pour ρ = -0.5', color='orange')
plt.axhline(0, color='black', linestyle='--')
plt.title("Différence entre delta log-normale et MC conditionnement")
plt.xlabel("S_1_0")
plt.ylabel("Différence des estimateurs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



########################################################################################
### Question 12 – Réduction de variance avec nouvelle variable de contrôle (F_T)
########################################################################################

def Euro_call_new_control(t, alpha, beta, S1_0, S2_0,alpha_S1_0, beta_S2_0, r, sigma1, sigma2, rho, N, K):
    # Valeur exacte de la variable de contrôle (basée sur F_T ~ N)
    esp = alpha * np.log(S1_0) + beta * np.log(S2_0) \
        + alpha * (r - 0.5 * sigma1**2) * t + beta * (r - 0.5 * sigma2**2) * t

    var = t * ((alpha * sigma1)**2 + (beta * sigma2)**2 + 2 * rho * alpha * sigma1 * beta * sigma2)
    std = np.sqrt(var)
    d = (esp - np.log(K)) / std

    # Approximation analytique de E[(e^{F_T} - K)^+]
    exact_control = np.exp(-r * t) * (
        np.exp(esp + 0.5 * var) * norm.cdf(d + std) - K * norm.cdf(d)
    )

    # Simulation Monte Carlo
    X, Y = Box_Muller(N)
    W1 = np.sqrt(t) * X
    W2 = np.sqrt(t) * (rho * X + np.sqrt(1 - rho**2) * Y)

    S1_T = np.exp((r - 0.5 * sigma1**2) * t + sigma1 * W1)
    S2_T = np.exp((r - 0.5 * sigma2**2) * t + sigma2 * W2)

    F_T = alpha * np.log(S1_0*S1_T) + beta * np.log(S2_0*S2_T)

    X_payoff = np.maximum(alpha_S1_0 * S1_T + beta_S2_0 * S2_T - K, 0)
    Y_payoff = np.maximum(np.exp(F_T) - K, 0)

    # Nouvelle variable de contrôle : Y^c = X - Y
    Yc = X_payoff - Y_payoff

    # Estimateur de variance réduite
    MC_price = np.exp(-r * t) * np.mean(Yc) + exact_control
    MC_error = 1.96 * np.std(Yc) / np.sqrt(N)
    IC_sup = MC_price + MC_error
    IC_inf = MC_price - MC_error

    return MC_price, MC_error, IC_inf, IC_sup

T=2
r=0.01
N=100000
alpha=0.5
beta=0.5
S1_0=2
S2_0=2

# Exemple d’appel avec les mêmes paramètres globaux
MC_price_nv, MC_error_nv, IC_inf_nv, IC_sup_nv = Euro_call_new_control(
    T, alpha, beta, S1_0, S2_0,1,1, r, 0.35, 0.4, 0.3, N, 2
)

print("Prix MC Réduction de variance (nouvelle VC) :", MC_price_nv)
print("Erreur MC :", MC_error_nv)
print("IC à 95% : [{:.4f}, {:.4f}]".format(IC_inf_nv, IC_sup_nv))



sigma1 = 0.35
sigma2 = 0.4
T=2
r=0.01
N=100000
beta=0.5
alpha=0.5
S1_0=2
S2_0=2

alpha_values = np.linspace(0, 2, 100)


# Initialiser une liste pour stocker les valeurs de MC_price
#MC_price_values_control = []
MC_error_VC1=[]
MC_error_VC2=[]
MC_error_VC3=[]


# Appeler la fonction pour chaque valeur de rho et stocker MC_price
for alpha_S1_0 in alpha_values:
   MC_price_1, MC_error_pt, IC_sup_1, IC_inf_1 = Euro_call_controle(T, alpha_S1_0, 1, r, sigma1, sigma2, rho, N, K)
   MC_price_2, MC_error_cl, IC_sup_2, IC_inf_2 = Euro_call_new_control(T, alpha, beta, S1_0, S2_0,alpha_S1_0, 1, r, sigma1, sigma2, rho, N, K)
   MC_price_3, MC_error_tt, IC_sup_3, IC_inf_3 = Euro_call(T, alpha_S1_0, 1, r, sigma1, sigma2, rho, N, K)
 
   
   MC_error_VC1.append(MC_error_pt*np.sqrt(N)/1.96)
   MC_error_VC2.append(MC_error_cl*np.sqrt(N)/1.96)
   MC_error_VC3.append(MC_error_tt*np.sqrt(N)/1.96)

plt.plot(alpha_values, MC_error_VC1, label='Variance avec la première VC')
plt.plot(alpha_values, MC_error_VC2, label='Variance avec la deuxième VC')
plt.plot(alpha_values, MC_error_VC3, label='Variance MC')
plt.title('Variation de la variance en fonction de alpha_S_1_0')
plt.xlabel('alpha_S_1_0')
plt.ylabel('Variance')
plt.legend()
plt.grid(True)
plt.show()