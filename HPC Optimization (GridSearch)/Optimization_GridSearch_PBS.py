############################################## Libraries and data #############################################

import sys
import pickle
import itertools

import numpy as np
import pandas as pd

from scipy.optimize import root
from scipy.integrate import quad
from IPython.display import display, Math

# --------------------------------------------------------------------------------
# Load simulated primitives (emissions, means, stds, cost parameters, etc.)
# These are created in the calibration/simulation notebook and saved as pickles.
# --------------------------------------------------------------------------------
with open('WSimul.pkl', 'rb') as pickle_file:
    W = pickle.load(pickle_file)

with open('MuSimul.pkl', 'rb') as pickle_file:
    mu = pickle.load(pickle_file)

with open('MuSimulUnadjusted.pkl', 'rb') as pickle_file:
    mu_Unadjusted = pickle.load(pickle_file)

with open('StdSimul.pkl', 'rb') as pickle_file:
    std = pickle.load(pickle_file)

with open('StdSimulUnadjusted.pkl', 'rb') as pickle_file:
    std_Unadjusted = pickle.load(pickle_file)

with open('EtaSimul.pkl', 'rb') as pickle_file:
    eta = pickle.load(pickle_file)

with open('hSimul.pkl', 'rb') as pickle_file:
    h = pickle.load(pickle_file)


########################################## Parameters (Pierrehumbert) #########################################

# Gas lifetimes in years
tau_CH4 = 12.4
tau_N2O = 114.0
tau_SF6 = 3200.0
tau_CO2 = 1e5

# Radiative efficiency (W/m^2/ppb)
a_CH4 = 3.7e-4
a_N2O = 3e-3
a_SF6 = 0.52
a_CO2 = 1.4e-5

# Molecular weights (g/mol)
moles_CH4 = 16.04
moles_N2O = 44.0
moles_SF6 = 146.0
moles_CO2 = 44.0
moles_C = 12.0

# Two-box energy model parameters
lamda = 1.2    # W/m^2/K
gamma = 1.2    # (kept for reference; not used in the new volatility spec)
mu_mix = 0.75

# Time step (for interpretation only – integrals are continuous in this script)
dt_T = 1

# Scaling factor (GtCO2 -> tCO2, etc.)
adj = 1e9

# Multiplier for CH4 (calibrated)
d_CH4 = 0.05859375


############################################   Parameters (All)   #############################################

# Horizon and model dimensions
T = 26          # Firm horizon (years)
K = 2           # Number of sectors
N = 1           # Number of firms per sector (in this implementation)

# -----------------------------------------------------------------------------
# Cost-function helpers:
#   c_fun(h, eta) = h * eta
#   nu_fun(h, eta) = h^2 * eta
# Both handle scalar and list-valued h, eta (for multiple (h, eta) scenarios).
# -----------------------------------------------------------------------------
def c_fun(h, eta):
    if isinstance(h, list) and isinstance(eta, list):
        heta = []
        combinations = list(itertools.product(h, eta))
        for x in combinations:
            heta.append(x[0] * x[1])
        return heta, combinations
    else:
        return h * eta


def nu_fun(h, eta):
    if isinstance(h, list) and isinstance(eta, list):
        heta = []
        combinations = list(itertools.product(h, eta))
        for x in combinations:
            heta.append(x[0] ** 2 * x[1])
        return heta, combinations
    else:
        return h ** 2 * eta


# Sector-specific cost coefficients
c_1 = c_fun(h=h['0'], eta=eta['0'])
c_2 = c_fun(h=h['1'], eta=eta['1'])
c = {"0": c_1, "1": c_2}

nu_1 = nu_fun(h=h['0'], eta=eta['0'])
nu_2 = nu_fun(h=h['1'], eta=eta['1'])
nu = {"0": nu_1, "1": nu_2}

# Allowed deviation (delta) in the tolerance constraint
delta = {"0": 0.1, "1": 0.1}


# Radiative forcing parameter for CH4 (Pierrehumbert formula)
def k_gas(a, m, tau):
    return (5500.34 / m) * a


k_CH4 = k_gas(a=a_CH4, m=moles_CH4, tau=tau_CH4)


############################################        Optimization(code)       ##############################################


def Optimization(
    mu_opt,
    c,
    nu,
    std,
    W,
    T,
    T_tilde,
    eta,
    delta,
    y_initial_guess,
    lambda_,
    gamma_,   # kept in signature for compatibility, not used after switching vol to sqrt(T-t)
    U,
    ul=20,
    d=d_CH4,
    adj=1e9,
):
    """
    Grid-search based optimization over (b_0, b_1) with an expected-temperature constraint.

    For each grid point b = (b_0, b_1):
      1. Solve for Lagrange multipliers y^j from the tolerance equation.
      2. Compute prices P^j(b).
      3. Compute the expected temperature index E[Î(T_tilde)] and unconstrained upper bound U_0.
      4. If E[Î(T_tilde)] <= U, include the point in the constrained set.
      5. Record social cost J(b) and export results as pickles.
    """

    # ---------------------------------------------------------------------
    # Compute epsilon_min per sector using the volatility spec:
    #     sigma(t) = sigma_0 * sqrt(T - t)
    # ---------------------------------------------------------------------
    def tolerance_AllSectors(T, mu, c, std, adj, dt, W, delta):

        # Count number of sectors present in W["1"]
        sectors = 0
        Steps = len(W["1"]["Firm_00"])  # Assuming "Firm_00" exists

        for key in W["1"].keys():
            if key.startswith("Sector"):
                sectors += 1

        print(f"There are {sectors} sectors!")

        def toler(T, mu, c, std, adj, dt, delta):
            """
            Compute epsilon_min for a *single* sector with:
                sigma(t) = std * adj * sqrt(T - t)
            and tolerance parameter delta.
            """

            def integrand(t):
                # New volatility spec: sigma(t) = sigma_0 * sqrt(T - t)
                vol = (std * adj) * np.sqrt(max(T - t, 0.0))
                return vol ** 2

            # Integrate sigma^2 over [0, T]
            integral_result, _ = quad(integrand, 0, T)

            # Minimal epsilon consistent with the tolerance inequality
            epsilon_min = np.sqrt(delta * (((mu * adj) + c) * T) ** 2 + integral_result)
            return epsilon_min

        epsilons = {}

        # Compute epsilon_min sector-by-sector
        for i in range(sectors):
            if not isinstance(c[f"{i}"], list):
                epsilons[f"{i}"] = toler(
                    T=T,
                    mu=mu[f"{i}"],
                    c=c[f"{i}"],
                    std=std[f"{i}"],
                    adj=adj,
                    dt=dt_T,
                    delta=delta[f"{i}"],
                )
            else:
                epsilons[f"{i}"] = []
                for k in range(len(c[f"{i}"])):
                    epsilons[f"{i}"].append(
                        toler(
                            T=T,
                            mu=mu[f"{i}"],
                            c=c[f"{i}"][k],
                            std=std[f"{i}"],
                            adj=adj,
                            dt=dt_T,
                            delta=delta[f"{i}"],
                        )
                    )

        # Report sector-wise minima; handle scalar vs list-valued c
        if not isinstance(c["0"], list):
            epsilon_min = min(epsilons.values())
            print(f"Epsilon should be in the range (0, {epsilon_min})")
        else:
            epsilon_min = []
            for k in range(len(c["0"])):
                epsilon_min_k = min(
                    [epsilons[f"{i}"][k] for i in range(sectors) if isinstance(epsilons[f"{i}"], list)]
                )
                epsilon_min.append(epsilon_min_k)
                display(
                    Math(
                        f"\\varepsilon\\text{{ should be in the range }} (0, {epsilon_min_k})"
                        f"\\text{{ for }} \\bar{{c}}_1 = {c['0'][k]} \\text{{ and }} "
                        f"\\bar{{c}}_2 = {c['1'][k]}!"
                    )
                )

        return epsilons, epsilon_min

    # ---------------------------------------------------------------------
    # Grid over (b_0, b_1) in [0, (1-δ^j)(μ^j + c^j/adj)T)
    # ---------------------------------------------------------------------
    def b1bkGrid(mu, c, T, delta, ul=20, adj=1e9):
        max_, x_ = {}, {}
        for key in mu.keys():
            max_[key] = (1 - delta[key]) * (mu[key] + c[key] / adj) * T
            x_[key] = np.linspace(0, max_[key], ul, endpoint=False)

        x_values = [x_[key] for key in mu.keys()]
        grids = np.meshgrid(*x_values, indexing='ij')
        cartesian_prod = np.stack([grid.flatten() for grid in grids], axis=-1)
        return cartesian_prod

    # ---------------------------------------------------------------------
    # Solve for Lagrange multipliers y^j(T; b) given epsilon and b
    # using the volatility sigma(t) = sigma_0 * sqrt(T - t).
    # ---------------------------------------------------------------------
    def solveY(W, std, T, eta, mu, c, adj, eps, b, y_initial_guess, gamma_, combinations=None):

        sectors = 0
        for key in W["1"].keys():
            if key.startswith("Sector"):
                sectors += 1

        if isinstance(std, dict):
            y = {}
            for i in range(sectors):

                def integrand(t, y):
                    # New volatility spec
                    vol = (std[f"{i}"] * adj) * np.sqrt(max(T - t, 0.0))
                    return (vol ** 2) / (1 + 2 * y * eta[f"{i}"] * (T - t)) ** 2

                def equation(y):
                    integral_result, _ = quad(integrand, 0, T, args=(y,))
                    term_1 = (
                        ((mu[f"{i}"] * adj + c[f"{i}"]) * T) - (b[f"{i}"] * adj)
                    ) / (1 + (2 * y * eta[f"{i}"] * T))
                    return (term_1 ** 2) + integral_result - (eps ** 2)

                sol = root(equation, y_initial_guess)
                if sol.success:
                    y[f"{i}"] = sol.x[0]
                else:
                    print(f"Solver failed to converge for sector {i}")
                    y[f"{i}"] = None
        return y

    # ---------------------------------------------------------------------
    # Pricing function: P^j(b) under the quadratic penalty structure.
    # ---------------------------------------------------------------------
    def Pricing(y, mu, c, T, b, eta, adj=1e9):
        P_null = {}
        for key in y.keys():
            P_null[key] = (
                2 * y[key] * ((((mu[key] * adj) + c[key]) * T) - (b[key] * adj))
            ) / (1 + 2 * y[key] * eta[key] * T)
        return P_null

    # ---------------------------------------------------------------------
    # Social cost functional J(b) given y, P, and volatility sigma(t)=sigma_0 sqrt(T-t).
    # ---------------------------------------------------------------------
    def socialCost(T, std, y, eta, nu, P, adj=1e9):
        def inner_integrand(s, y, eta, T, adj, std):
            # New volatility spec
            vol = (std * adj) * np.sqrt(max(T - s, 0.0))
            return (vol ** 2) / (1 + 2 * y * eta * (T - s)) ** 2

        def outer_integrand(t, y, eta, T, adj, std):
            inner_result, _ = quad(inner_integrand, 0, t, args=(y, eta, T, adj, std))
            return inner_result

        result = 0
        for key in y:
            outer_result, _ = quad(
                outer_integrand, 0, T, args=(y[key], eta[key], T, adj, std[key])
            )
            result += (
                (eta[key] / 2) * outer_result
                + (eta[key] / 2) * P[key] ** 2 * T
                - (nu[key] / 2) * T
            )

        return result

    # ---------------------------------------------------------------------
    # Temperature functional I^a(T_tilde) for CO2 & CH4 under linear response.
    # ---------------------------------------------------------------------
    def Temp(T_firm, T_regulator, N, mu_bar, c_bar, eta_bar, P_hat_0, y_bnull, lambda_, mu_mix, d, adj):

        alpha = lambda_ / mu_mix

        def f_CO2(
            u_s,
            d1=0.004223 * 0.27218,
            d2=0.004223 * 0.14621,
            d3=0.004223 * 0.13639,
            d4=0.004223 * 0.44422,
        ):
            return (
                d1 * np.exp(u_s / 8.696)
                + d2 * np.exp(u_s / 93.33)
                + d3 * np.exp(u_s / 645.87)
                + d4
            )

        def f_OtherGas(u, d=d, tau_gas=tau_CH4):
            return d * np.exp(u / tau_gas)

        def integrand_inner_CO2(u, s):
            return f_CO2(u - s)

        def integrand_inner_CH4(u, s, d):
            return f_OtherGas(u - s, d)

        def integrand_outer_CO2(s):
            return np.exp(alpha * (s - T_regulator)) * quad(
                integrand_inner_CO2, 0, min(T_firm, s), args=(s,)
            )[0]

        def integrand_outer_CH4(s):
            return np.exp(alpha * (s - T_regulator)) * quad(
                integrand_inner_CH4, 0, min(T_firm, s), args=(s, d)
            )[0]

        # Shared term_2 is overwritten inside loop but harmless; kept for clarity
        term_2 = quad(integrand_outer_CO2, 0, T_regulator)[0]

        result_E, result_U = 0, 0

        for dic in mu_bar.keys():

            if (int(dic) % 2) == 0:  # CO2 sector

                term_2 = quad(integrand_outer_CO2, 0, T_regulator)[0]

                result_E += (
                    N
                    * (
                        ((mu_bar[dic] * adj) + c_bar[dic] - (eta_bar[dic] * P_hat_0[dic]))
                        / adj
                    )
                    * term_2
                )

                result_U += N * mu_bar[dic] * term_2

            else:  # CH4 sector

                term_2 = quad(integrand_outer_CH4, 0, T_regulator)[0]

                result_E += (
                    N
                    * (
                        (mu_bar[dic] * adj)
                        + (c_bar[dic] - (eta_bar[dic] * P_hat_0[dic]))
                    )
                    / adj
                    * term_2
                )

                result_U += N * (mu_bar[dic] * adj) / adj * term_2

        return result_E, result_U

    # =====================================================================
    # MAIN BODY OF Optimization()
    # =====================================================================

    # 1) Compute epsilon_min (volatility consistent) and construct b-grid
    _, epsilon_min = tolerance_AllSectors(
        T=T, mu=mu_opt, c=c, std=std, adj=adj, dt=dt_T, W=W, delta=delta
    )

    b_grid = b1bkGrid(mu=mu_opt, c=c, T=T, delta=delta, ul=ul, adj=adj)

    # Containers for constrained vs unconstrained results
    J_constrained = []
    grid_constrained_df = []
    prices_constrained_df = []
    y_constrained_df = []

    J = []
    grid_df = []
    prices_df = []
    y_df = []
    Temps_df = []

    # 2) Loop over all grid points b = (b_0, b_1)
    for i in range(len(b_grid)):
        timer = np.round((i / len(b_grid)) * 100)
        grid = {"0": b_grid[i][0], "1": b_grid[i][1]}

        # Solve for Lagrange multipliers given epsilon_min
        y = solveY(
            W=W,
            std=std,
            T=T,
            eta=eta,
            mu=mu_opt,
            c=c,
            adj=adj,
            eps=epsilon_min / adj,
            b=grid,
            y_initial_guess=y_initial_guess,
            gamma_=gamma_,
        )

        # Check sign of multipliers (sanity check)
        for key in y.keys():
            if y[key] is not None and y[key] < 0:
                print(f"The Lagrange multiplier for {grid[key]} is less than 0!")

        # Compute prices and temperature
        Null_prices = Pricing(y=y, mu=mu_opt, c=c, T=T, b=grid, eta=eta, adj=adj)

        E_U, U_U = Temp(
            T_firm=T,
            T_regulator=T_tilde,
            N=N,
            mu_bar=mu_opt,
            c_bar=c,
            eta_bar=eta,
            P_hat_0=Null_prices,
            y_bnull=y,
            lambda_=lambda_,
            mu_mix=mu_mix,
            d=d,
            adj=adj,
        )

        sys.stdout.write(
            f"\r Grid loop: {timer}% and the expected terminal temperature is {E_U} (with {U_U} as upperbound)!"
        )
        sys.stdout.flush()

        # If temperature constraint satisfied, store in constrained set
        if E_U <= U:
            J_constrained.append(
                socialCost(T=T, std=std, y=y, eta=eta, nu=nu, P=Null_prices, adj=adj)
            )
            grid_constrained_df.append([grid["0"], grid["1"]])
            prices_constrained_df.append([Null_prices["0"], Null_prices["1"]])
            y_constrained_df.append([y["0"], y["1"]])

        # Always store unconstrained social cost / temps
        J.append(
            socialCost(T=T, std=std, y=y, eta=eta, nu=nu, P=Null_prices, adj=adj)
        )
        grid_df.append([grid["0"], grid["1"]])
        prices_df.append([Null_prices["0"], Null_prices["1"]])
        y_df.append([y["0"], y["1"]])
        Temps_df.append(E_U)

    # ---------------------------------------------------------------------
    # Collect results into DataFrames and save to disk
    # ---------------------------------------------------------------------
    J_constrained_df = pd.DataFrame(J_constrained)
    grid_constrained_df = pd.DataFrame(grid_constrained_df)
    prices_constrained_df = pd.DataFrame(prices_constrained_df)
    y_constrained_df = pd.DataFrame(y_constrained_df)
    c_df = pd.DataFrame(list(c.items()), columns=['Key', 'c'])
    nu_df = pd.DataFrame(list(nu.items()), columns=['Key', 'nu'])
    Temps_df = pd.DataFrame(Temps_df)

    J_df = pd.DataFrame(J)
    grid_df = pd.DataFrame(grid_df)
    prices_df = pd.DataFrame(prices_df)
    y_df = pd.DataFrame(y_df)
    c_df = pd.DataFrame(list(c.items()), columns=['Key', 'c'])
    nu_df = pd.DataFrame(list(nu.items()), columns=['Key', 'nu'])

    J_constrained_df.columns = ["Constrained social cost"]
    grid_constrained_df.columns = ["b_0", "b_1"]
    prices_constrained_df.columns = ["P_0", "P_1"]
    y_constrained_df.columns = ["y_0", "y_1"]
    c_df.columns = ["c_0", "c_1"]
    nu_df.columns = ["nu_0", "nu_1"]

    J_df.columns = ["Social cost"]
    grid_df.columns = ["b_0", "b_1"]
    prices_df.columns = ["P_0", "P_1"]
    y_df.columns = ["y_0", "y_1"]
    c_df.columns = ["c_0", "c_1"]
    nu_df.columns = ["nu_0", "nu_1"]
    Temps_df.columns = ["Temps"]

    Constrained_results = pd.concat(
        [J_df, grid_df, prices_df, y_df, c_df, nu_df, Temps_df], axis=1
    )
    Constrained_results.to_pickle(f"Unconstrained_results_{U}_{T_tilde}_{ul}.pkl")

    Unconstrained_results = pd.concat(
        [J_constrained_df, grid_constrained_df, prices_constrained_df, y_constrained_df, c_df, nu_df],
        axis=1,
    )
    Unconstrained_results.to_pickle(f"Constrained_results_{U}_{T_tilde}_{ul}.pkl")

    return Constrained_results


###########################################     Optimization (Execution)      #############################################

# Single regulator horizon (can be looped over as needed)
T_tilde = [26]

J = []

for T_Tilde in T_tilde:
    print(T_Tilde)
    J.append(
        Optimization(
            W=W,
            mu_opt=mu,
            c=c,
            nu=nu,
            std=std,
            T=T,
            T_tilde=T_Tilde,
            eta=eta,
            delta=delta,
            U=0.4,
            y_initial_guess=1e-10,
            lambda_=lamda,
            gamma_=0.25,
            ul=5,
            d=0.565,   # CH4 scaling used in this particular run
            adj=1e9,
        )
    )
