############################################## Libraries and data #############################################

import sys
import pickle
import itertools

import numpy as np
import pandas as pd

from scipy.optimize import root
from scipy.integrate import quad
from IPython.display import display, Math

########################################## Load simulated primitives ##########################################

# These pickles are produced by the calibration/simulation notebook.
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
gamma = 1.2    # kept for reference; not used in new σ(t) spec
mu_mix = 0.75

# Time step (only for interpretation – integrals are continuous)
dt_T = 1

# Scaling factor (GtCO2 -> tCO2, etc.)
adj = 1e9

# CH4 multiplier (calibrated)
d_CH4 = 0.05859375


############################################   Parameters (All)   #############################################

# Horizon and model dimensions
T = 26          # Firm horizon (years)
K = 2           # Number of sectors
N = 1           # Number of firms per sector (here effectively scaling)

# --------------------------------------------------------------------------------
# Cost-function helpers:
#   c_fun(h, eta)  = h * eta
#   nu_fun(h, eta) = h^2 * eta
# Both allow list-valued h, eta (multiple (h,η) scenarios), but here we use scalars.
# --------------------------------------------------------------------------------
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


# Radiative forcing parameter for CH4 (Pierrehumbert-style scaling)
def k_gas(a, m, tau):
    return (5500.34 / m) * a


k_CH4 = k_gas(a=a_CH4, m=moles_CH4, tau=tau_CH4)


############################################  Optimization via SGD  ############################################

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
    gamma_,   # kept in signature for compatibility (σ now uses sqrt(T-t) instead)
    U,
    ul=20,          # only used to label output files (kept for compatibility)
    d=d_CH4,
    adj=1e9,
    max_iters=250,  # SGD iterations
    lr=5e-4,        # learning rate
    fd_step=1e-3,   # finite-difference step (relative to b_max)
    penalty_weight=1e4,  # penalty weight on temperature violation
):
    """
    Stochastic-gradient-based optimization over (b_0, b_1), replacing the
    original grid search.

    For each iteration t, we:
      1. Given b_t = (b_0, b_1), solve for Lagrange multipliers y^j from the
         tolerance equation using σ(t) = σ_0 * sqrt(T - t).
      2. Compute prices P^j(b_t).
      3. Compute the expected temperature index E[I(T_tilde; b_t)] and
         unconstrained upper bound U_0(b_t) via Temp().
      4. Form the penalised objective
             J_pen(b_t) = J(b_t) + penalty_weight * max(0, E - U)^2.
      5. Approximate ∇J_pen(b_t) by finite differences and update b_t with
         a projected gradient step, enforcing:
             0 ≤ b_j ≤ (1 - δ_j)(μ_j + c_j/adj)T.

    We record:
      - Unconstrained trajectory of all iterates (J, b, P, y, Temps),
      - Constrained subset where E[I(T_tilde; b)] ≤ U.

    Results are exported as pickles:
      - Unconstrained_results_SGD_{U}_{T_tilde}_{ul}.pkl
      - Constrained_results_SGD_{U}_{T_tilde}_{ul}.pkl
    """

    # ---------------------------------------------------------------------
    # 1. Compute epsilon_min per sector using σ(t) = σ_0 * sqrt(T - t)
    # ---------------------------------------------------------------------
    def tolerance_AllSectors(T, mu, c, std, adj, dt, W, delta):

        # Count number of sectors present in W["1"]
        sectors = 0
        Steps = len(W["1"]["Firm_00"])  # Assuming "Firm_00" exists

        for key in W["1"].keys():
            if key.startswith("Sector"):
                sectors += 1

        print(f"There are {sectors} sectors!")

        def toler(T, mu_val, c_val, std_val, adj_val, dt_val, delta_val):
            """
            Compute epsilon_min for a *single* sector with:
                σ(t) = std_val * adj_val * sqrt(T - t),
            and tolerance parameter delta_val.
            """

            def integrand(t):
                vol = (std_val * adj_val) * np.sqrt(max(T - t, 0.0))
                return vol ** 2

            # ∫_0^T σ(t)^2 dt
            integral_result, _ = quad(integrand, 0, T)

            epsilon_min_val = np.sqrt(
                delta_val * (((mu_val * adj_val) + c_val) * T) ** 2 + integral_result
            )
            return epsilon_min_val

        epsilons = {}

        # Compute epsilon_min sector-by-sector
        for i in range(sectors):
            if not isinstance(c[f"{i}"], list):
                epsilons[f"{i}"] = toler(
                    T=T,
                    mu_val=mu[f"{i}"],
                    c_val=c[f"{i}"],
                    std_val=std[f"{i}"],
                    adj_val=adj,
                    dt_val=dt_T,
                    delta_val=delta[f"{i}"],
                )
            else:
                epsilons[f"{i}"] = []
                for k_idx in range(len(c[f"{i}"])):
                    epsilons[f"{i}"].append(
                        toler(
                            T=T,
                            mu_val=mu[f"{i}"],
                            c_val=c[f"{i}"][k_idx],
                            std_val=std[f"{i}"],
                            adj_val=adj,
                            dt_val=dt_T,
                            delta_val=delta[f"{i}"],
                        )
                    )

        # Scalar case (ours): just print the minimum epsilon across sectors
        if not isinstance(c["0"], list):
            epsilon_min = min(epsilons.values())
            print(f"Epsilon should be in the range (0, {epsilon_min})")
        else:
            epsilon_min = []
            for k_idx in range(len(c["0"])):
                epsilon_min_k = min(
                    [
                        epsilons[f"{i}"][k_idx]
                        for i in range(sectors)
                        if isinstance(epsilons[f"{i}"], list)
                    ]
                )
                epsilon_min.append(epsilon_min_k)
                display(
                    Math(
                        f"\\varepsilon\\text{{ should be in the range }} (0, {epsilon_min_k})"
                        f"\\text{{ for }} \\bar{{c}}_1 = {c['0'][k_idx]} \\text{{ and }} "
                        f"\\bar{{c}}_2 = {c['1'][k_idx]}!"
                    )
                )

        return epsilons, epsilon_min

    # ---------------------------------------------------------------------
    # 2. Solve for Lagrange multipliers y^j(T; b)
    #    given epsilon and b, under σ(t) = σ_0 * sqrt(T - t)
    # ---------------------------------------------------------------------
    def solveY(W, std, T, eta, mu, c, adj, eps, b, y_initial_guess, gamma_, combinations=None):

        sectors = 0
        for key in W["1"].keys():
            if key.startswith("Sector"):
                sectors += 1

        y = {}
        for i in range(sectors):

            def integrand(t, y_val):
                vol = (std[f"{i}"] * adj) * np.sqrt(max(T - t, 0.0))
                return (vol ** 2) / (1 + 2 * y_val * eta[f"{i}"] * (T - t)) ** 2

            def equation(y_val):
                integral_result, _ = quad(integrand, 0, T, args=(y_val,))
                term_1 = (
                    ((mu[f"{i}"] * adj + c[f"{i}"]) * T) - (b[f"{i}"] * adj)
                ) / (1 + (2 * y_val * eta[f"{i}"] * T))
                return (term_1 ** 2) + integral_result - (eps ** 2)

            sol = root(equation, y_initial_guess)
            if sol.success:
                y[f"{i}"] = sol.x[0]
            else:
                print(f"Solver failed to converge for sector {i}")
                y[f"{i}"] = None

        return y

    # ---------------------------------------------------------------------
    # 3. Pricing function P^j(b)
    # ---------------------------------------------------------------------
    def Pricing(y, mu, c, T, b, eta, adj=1e9):
        P_null = {}
        for key in y.keys():
            P_null[key] = (
                2 * y[key] * ((((mu[key] * adj) + c[key]) * T) - (b[key] * adj))
            ) / (1 + 2 * y[key] * eta[key] * T)
        return P_null

    # ---------------------------------------------------------------------
    # 4. Social cost functional J(b) with σ(t)=σ_0√(T−t)
    # ---------------------------------------------------------------------
    def socialCost(T, std, y, eta, nu, P, adj=1e9):

        def inner_integrand(s, y_val, eta_val, T_val, adj_val, std_val):
            vol = (std_val * adj_val) * np.sqrt(max(T_val - s, 0.0))
            return (vol ** 2) / (1 + 2 * y_val * eta_val * (T_val - s)) ** 2

        def outer_integrand(t, y_val, eta_val, T_val, adj_val, std_val):
            inner_result, _ = quad(
                inner_integrand, 0, t, args=(y_val, eta_val, T_val, adj_val, std_val)
            )
            return inner_result

        result = 0.0
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
    # 5. Temperature functional I(T_tilde; b)
    # ---------------------------------------------------------------------
    def Temp(T_firm, T_regulator, N, mu_bar, c_bar, eta_bar, P_hat_0, y_bnull,
             lambda_, mu_mix, d, adj):

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

        def f_OtherGas(u, d_val=d, tau_gas=tau_CH4):
            return d_val * np.exp(u / tau_gas)

        def integrand_inner_CO2(u, s):
            return f_CO2(u - s)

        def integrand_inner_CH4(u, s, d_val):
            return f_OtherGas(u - s, d_val)

        def integrand_outer_CO2(s):
            return np.exp(alpha * (s - T_regulator)) * quad(
                integrand_inner_CO2, 0, min(T_firm, s), args=(s,)
            )[0]

        def integrand_outer_CH4(s):
            return np.exp(alpha * (s - T_regulator)) * quad(
                integrand_inner_CH4, 0, min(T_firm, s), args=(s, d)
            )[0]

        result_E, result_U = 0.0, 0.0

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
    # MAIN BODY: SGD over b = (b_0, b_1)
    # =====================================================================

    # 1) Compute epsilon_min (same as in grid-search version, but with new σ)
    _, epsilon_min = tolerance_AllSectors(
        T=T, mu=mu_opt, c=c, std=std, adj=adj, dt=dt_T, W=W, delta=delta
    )

    # 2) Box constraints for b_j as in original grid
    b_max = {
        key: (1 - delta[key]) * (mu_opt[key] + c[key] / adj) * T
        for key in mu_opt.keys()
    }

    # Start SGD from the middle of the box
    b = {key: 0.5 * b_max[key] for key in mu_opt.keys()}

    # Containers: treat each SGD iterate as a "grid point" in the output
    J_constrained = []
    grid_constrained_df = []
    prices_constrained_df = []
    y_constrained_df = []

    J_all = []
    grid_all = []
    prices_all = []
    y_all = []
    Temps_all = []

    # Track best feasible point (E ≤ U)
    best_feasible = None
    best_feasible_J = np.inf

    # Helper: evaluate J(b), E(b), J_pen(b)
    def eval_obj_and_temp(b_dict):
        y_curr = solveY(
            W=W,
            std=std,
            T=T,
            eta=eta,
            mu=mu_opt,
            c=c,
            adj=adj,
            eps=epsilon_min / adj,
            b=b_dict,
            y_initial_guess=y_initial_guess,
            gamma_=gamma_,
        )

        P_curr = Pricing(y=y_curr, mu=mu_opt, c=c, T=T, b=b_dict, eta=eta, adj=adj)

        E_curr, U_curr = Temp(
            T_firm=T,
            T_regulator=T_tilde,
            N=N,
            mu_bar=mu_opt,
            c_bar=c,
            eta_bar=eta,
            P_hat_0=P_curr,
            y_bnull=y_curr,
            lambda_=lambda_,
            mu_mix=mu_mix,
            d=d,
            adj=adj,
        )

        J_curr = socialCost(T=T, std=std, y=y_curr, eta=eta, nu=nu, P=P_curr, adj=adj)

        penalty = penalty_weight * max(0.0, E_curr - U) ** 2
        J_pen = J_curr + penalty

        return J_curr, E_curr, J_pen, y_curr, P_curr

    # 3) SGD loop
    for it in range(max_iters):
        timer = np.round((it / max_iters) * 100)

        # Current penalised objective and temperature
        J_curr, E_curr, J_pen, y_curr, P_curr = eval_obj_and_temp(b)

        sys.stdout.write(
            f"\r SGD iter {it+1}/{max_iters} ({timer}%), "
            f"E[T] = {E_curr:.4f}, J = {J_curr:.4f}"
        )
        sys.stdout.flush()

        # Store unconstrained trajectory
        J_all.append(J_curr)
        grid_all.append([b["0"], b["1"]])
        prices_all.append([P_curr["0"], P_curr["1"]])
        y_all.append([y_curr["0"], y_curr["1"]])
        Temps_all.append(E_curr)

        # If feasible, also store in constrained set and update best feasible
        if E_curr <= U:
            J_constrained.append(J_curr)
            grid_constrained_df.append([b["0"], b["1"]])
            prices_constrained_df.append([P_curr["0"], P_curr["1"]])
            y_constrained_df.append([y_curr["0"], y_curr["1"]])

            if J_curr < best_feasible_J:
                best_feasible_J = J_curr
                best_feasible = (b["0"], b["1"])

        # Gradient of penalised objective via finite differences
        grad = {}
        for key in b.keys():
            step = fd_step * max(b_max[key], 1.0)  # relative step
            b_pert = b.copy()
            b_pert[key] = float(np.clip(b[key] + step, 0.0, b_max[key]))

            _, _, J_pen_eps, _, _ = eval_obj_and_temp(b_pert)
            grad[key] = (J_pen_eps - J_pen) / step

        # Gradient step with box projection
        for key in b.keys():
            b[key] -= lr * grad[key]
            b[key] = float(np.clip(b[key], 0.0, b_max[key]))

    print("\nSGD finished.")
    if best_feasible is not None:
        print(f"Best feasible b found: {best_feasible} with J = {best_feasible_J}")
    else:
        print("No feasible point with E[T] <= U was found during SGD.")

    # ---------------------------------------------------------------------
    # Collect results into DataFrames and save as pickles
    # ---------------------------------------------------------------------
    J_constrained_df = pd.DataFrame(J_constrained)
    grid_constrained_df = pd.DataFrame(grid_constrained_df)
    prices_constrained_df = pd.DataFrame(prices_constrained_df)
    y_constrained_df = pd.DataFrame(y_constrained_df)

    J_df = pd.DataFrame(J_all)
    grid_df = pd.DataFrame(grid_all)
    prices_df = pd.DataFrame(prices_all)
    y_df = pd.DataFrame(y_all)
    Temps_df = pd.DataFrame(Temps_all)

    c_df = pd.DataFrame(list(c.items()), columns=['Key', 'c'])
    nu_df = pd.DataFrame(list(nu.items()), columns=['Key', 'nu'])

    if len(J_constrained_df) > 0:
        J_constrained_df.columns = ["Constrained social cost"]
        grid_constrained_df.columns = ["b_0", "b_1"]
        prices_constrained_df.columns = ["P_0", "P_1"]
        y_constrained_df.columns = ["y_0", "y_1"]

    J_df.columns = ["Social cost"]
    grid_df.columns = ["b_0", "b_1"]
    prices_df.columns = ["P_0", "P_1"]
    y_df.columns = ["y_0", "y_1"]
    Temps_df.columns = ["Temps"]

    c_df.columns = ["c_0", "c_1"]
    nu_df.columns = ["nu_0", "nu_1"]

    # NOTE: filenames have _SGD_ to avoid overwriting the grid-search results.
    Unconstrained_results = pd.concat(
        [J_df, grid_df, prices_df, y_df, c_df, nu_df, Temps_df],
        axis=1,
    )
    Unconstrained_results.to_pickle(
        f"Unconstrained_results_SGD_{U}_{T_tilde}_{ul}.pkl"
    )

    Constrained_results = pd.concat(
        [J_constrained_df, grid_constrained_df, prices_constrained_df,
         y_constrained_df, c_df, nu_df],
        axis=1,
    )
    Constrained_results.to_pickle(
        f"Constrained_results_SGD_{U}_{T_tilde}_{ul}.pkl"
    )

    return Unconstrained_results


###########################################     Optimization (Execution)      #############################################

# You can set T_tilde externally in a PBS script, or just hard-code as below.
# For HPC, it's natural to run ONE T_tilde per job and pass it via env or argv.

T_tilde_list = [26]   # e.g. [26, 31, 36, ...] if you want to loop

J = []

for T_Tilde in T_tilde_list:
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
            d=0.565,      # CH4 scaling for this run (as in your grid version)
            adj=1e9,
            # SGD hyperparams (tune if needed):
            max_iters=250,
            lr=5e-4,
            fd_step=1e-3,
            penalty_weight=1e4,
        )
    )
