import numpy as np
import matplotlib.pyplot as plt
from models import (
    xi_min,
    get_model_functions,
    candidate_mu,
    candidate_mu_derivatives,
)


def plot_muT(model_name, mu_pairs, A_values, n=400):
    muT_func, _, _, _ = get_model_functions(model_name)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for mu1, mu2 in mu_pairs:
        for A in A_values:
            x0 = xi_min(A)
            xi = np.linspace(x0, 1.0, n)
            ax.plot(xi, muT_func(xi, mu1, mu2, A), linewidth=2, label=f"mu1={mu1:.3g}, mu2={mu2:.3g}, A={A:g}")
    ax.set_xlabel(r"$x_i$")
    ax.set_ylabel(r"$\mu_T$")
    ax.set_title("Effective friction coefficient")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_FD(model_name, mu_pairs, A_values, n=400):
    _, fd_func, _, _ = get_model_functions(model_name)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for mu1, mu2 in mu_pairs:
        for A in A_values:
            x0 = xi_min(A)
            xi = np.linspace(x0, 1.0, n)
            ax.plot(xi, fd_func(xi, mu1, mu2, A), linewidth=2, label=f"mu1={mu1:.3g}, mu2={mu2:.3g}, A={A:g}")
    ax.set_xlabel(r"$x_i$")
    ax.set_ylabel(r"$F_D/(W_{bp}L)$")
    ax.set_title("Normalized drag force")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_dfd_mu1(model_name, A_values, n=400):
    _, _, s1_func, _ = get_model_functions(model_name)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for A in A_values:
        x0 = xi_min(A)
        xi = np.linspace(x0, 1.0, n)
        ax.plot(xi, s1_func(xi, A), linewidth=2, label=f"A={A:g}")
    ax.set_xlabel(r"$x_i$")
    ax.set_ylabel(r"$\partial [F_D/(W_{bp}L)]/\partial \mu_1$")
    ax.set_title(r"Sensitivity to $\mu_1$")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_dfd_mu2(model_name, A_values, n=400):
    _, _, _, s2_func = get_model_functions(model_name)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for A in A_values:
        x0 = xi_min(A)
        xi = np.linspace(x0, 1.0, n)
        ax.plot(xi, s2_func(xi, A), linewidth=2, label=f"A={A:g}")
    ax.set_xlabel(r"$x_i$")
    ax.set_ylabel(r"$\partial [F_D/(W_{bp}L)]/\partial \mu_2$")
    ax.set_title(r"Sensitivity to $\mu_2$")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_sensitivity(model_name, A_values, n=400):
    _, _, s1_func, s2_func = get_model_functions(model_name)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for A in A_values:
        x0 = xi_min(A)
        xi = np.linspace(x0, 1.0, n)
        ax.plot(xi, s1_func(xi, A), linewidth=2, label=f"dF/dmu1, A={A:g}")
        ax.plot(xi, s2_func(xi, A), "--", linewidth=2, label=f"dF/dmu2, A={A:g}")
    ax.set_xlabel(r"$x_i$")
    ax.set_ylabel("Sensitivity")
    ax.set_title(r"Sensitivity to $\mu_1$ and $\mu_2$")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_sensitivity_ratio(model_name, A_values, n=400):
    _, _, s1_func, s2_func = get_model_functions(model_name)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for A in A_values:
        x0 = xi_min(A)
        xi = np.linspace(x0, 1.0, n)
        s1 = np.abs(s1_func(xi, A))
        s2 = np.abs(s2_func(xi, A))
        ratio = np.divide(s2, s1, out=np.zeros_like(s2), where=s1 > 1e-12)
        ax.plot(xi, ratio, linewidth=2, label=f"A={A:g}")
    ax.set_xlabel(r"$x_i$")
    ax.set_ylabel(r"$R_\mu=|\partial F_D/\partial\mu_2| / |\partial F_D/\partial\mu_1|$")
    ax.set_title("Relative sensitivity ratio")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def scenario_values(model_name, A, xi_eval, H, V, d, coeffs1, coeffs2):
    muT_func, fd_func, _, _ = get_model_functions(model_name)
    mu1 = candidate_mu(H, V, d, coeffs1)
    mu2 = candidate_mu(H, V, d, coeffs2)
    fd = fd_func(np.array([xi_eval]), mu1, mu2, A)[0]
    mut = muT_func(np.array([xi_eval]), mu1, mu2, A)[0]
    return fd, mut, mu1, mu2


def plot_physical_variable(model_name, variable, response, A, xi_eval, H0, V0, d0, coeffs1, coeffs2, n=120):
    ranges = {
        "H*": (0.0, 1.0, "H* = Hbed/Dh"),
        "V*": (0.0, 2.0, "V* = Vpull/Vref"),
        "d*": (0.0, 0.5, "d* = dp/Dh"),
    }
    lo, hi, label = ranges[variable]
    x = np.linspace(lo, hi, n)
    y = []
    for val in x:
        H, V, d = H0, V0, d0
        if variable == "H*":
            H = val
        elif variable == "V*":
            V = val
        else:
            d = val
        fd, mut, _, _ = scenario_values(model_name, A, xi_eval, H, V, d, coeffs1, coeffs2)
        y.append(mut if response == "muT" else fd)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(x, y, linewidth=2.5)
    ax.set_xlabel(label)
    ax.set_ylabel(r"$\mu_T$" if response == "muT" else r"$F_D/(W_{bp}L)$")
    ax.set_title(f"Operational sensitivity: {label} -> {'mu_T' if response == 'muT' else 'FD'}")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    return fig


def plot_tornado(model_name, response, A, xi_eval, H0, V0, d0, coeffs1, coeffs2, perturbation=0.2):
    base_fd, base_mut, _, _ = scenario_values(model_name, A, xi_eval, H0, V0, d0, coeffs1, coeffs2)
    base_value = base_mut if response == "muT" else base_fd
    variables = [("H*", H0, 0.0, 1.0), ("V*", V0, 0.0, 2.0), ("d*", d0, 0.0, 0.5)]
    rows = []
    for name, base, lo_bound, hi_bound in variables:
        low_val = max(lo_bound, base * (1.0 - perturbation))
        high_val = min(hi_bound, base * (1.0 + perturbation))
        H_low, V_low, d_low = H0, V0, d0
        H_high, V_high, d_high = H0, V0, d0
        if name == "H*":
            H_low, H_high = low_val, high_val
        elif name == "V*":
            V_low, V_high = low_val, high_val
        else:
            d_low, d_high = low_val, high_val
        fd_low, mut_low, _, _ = scenario_values(model_name, A, xi_eval, H_low, V_low, d_low, coeffs1, coeffs2)
        fd_high, mut_high, _, _ = scenario_values(model_name, A, xi_eval, H_high, V_high, d_high, coeffs1, coeffs2)
        low_resp = mut_low if response == "muT" else fd_low
        high_resp = mut_high if response == "muT" else fd_high
        rows.append((name, low_resp - base_value, high_resp - base_value, abs(high_resp - low_resp)))
    rows.sort(key=lambda r: r[3], reverse=True)
    labels = [r[0] for r in rows]
    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for i, r in enumerate(rows):
        ax.barh(i, r[1], left=0)
        ax.barh(i, r[2], left=0)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Change in {'mu_T' if response == 'muT' else 'FD/(WbpL)'} relative to base case")
    ax.set_title(f"Tornado chart for {'mu_T' if response == 'muT' else 'FD'}, perturbation ±{int(perturbation*100)}%")
    ax.grid(True, axis="x", alpha=0.35)
    fig.tight_layout()
    return fig


def muT_operational(H, V, d, coeffs1, coeffs2, model_name, A, xi_eval):
    muT_func, _, _, _ = get_model_functions(model_name)
    mu1 = candidate_mu(H, V, d, coeffs1)
    mu2 = candidate_mu(H, V, d, coeffs2)
    return muT_func(np.array([xi_eval]), mu1, mu2, A)[0]


def _surface_grid(model_name, pair, A, xi_eval, H0, V0, d0, coeffs1, coeffs2, n=70):
    if pair == "H-V":
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 2, n)
        xlabel, ylabel = "H* = Hbed/Dh", "V* = Vpull/Vref"
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(n):
            for j in range(n):
                Z[i, j] = muT_operational(X[i, j], Y[i, j], d0, coeffs1, coeffs2, model_name, A, xi_eval)
    elif pair == "H-d":
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 0.5, n)
        xlabel, ylabel = "H* = Hbed/Dh", "d* = dp/Dh"
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(n):
            for j in range(n):
                Z[i, j] = muT_operational(X[i, j], V0, Y[i, j], coeffs1, coeffs2, model_name, A, xi_eval)
    else:
        x = np.linspace(0, 2, n)
        y = np.linspace(0, 0.5, n)
        xlabel, ylabel = "V* = Vpull/Vref", "d* = dp/Dh"
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(n):
            for j in range(n):
                Z[i, j] = muT_operational(H0, X[i, j], Y[i, j], coeffs1, coeffs2, model_name, A, xi_eval)
    return X, Y, Z, xlabel, ylabel


def plot_muT_surface(model_name, pair, A, xi_eval, H0, V0, d0, coeffs1, coeffs2):
    X, Y, Z, xlabel, ylabel = _surface_grid(model_name, pair, A, xi_eval, H0, V0, d0, coeffs1, coeffs2)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    cf = ax.contourf(X, Y, Z, levels=18)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"$\mu_T$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Response surface of mu_T: {pair}")
    fig.tight_layout()
    return fig


def plot_muT_surface_3d(model_name, pair, A, xi_eval, H0, V0, d0, coeffs1, coeffs2):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    X, Y, Z, xlabel, ylabel = _surface_grid(model_name, pair, A, xi_eval, H0, V0, d0, coeffs1, coeffs2, n=60)
    fig = plt.figure(figsize=(9, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(r"$\mu_T$")
    ax.set_title(f"3D response surface of mu_T: {pair}")
    fig.colorbar(surf, ax=ax, shrink=0.65, aspect=14, label=r"$\mu_T$")
    fig.tight_layout()
    return fig


def normalized_muT_sensitivities(model_name, A, xi_eval, H, V, d, coeffs1, coeffs2):
    muT_func, _, _, _ = get_model_functions(model_name)
    mu1 = candidate_mu(H, V, d, coeffs1)
    mu2 = candidate_mu(H, V, d, coeffs2)
    mut = muT_func(np.array([xi_eval]), mu1, mu2, A)[0]

    dmu1_dH, dmu1_dV, dmu1_dd = candidate_mu_derivatives(H, V, d, coeffs1)
    dmu2_dH, dmu2_dV, dmu2_dd = candidate_mu_derivatives(H, V, d, coeffs2)

    eps = 1e-6
    dmut_dmu1 = (muT_func(np.array([xi_eval]), mu1 + eps, mu2, A)[0] - muT_func(np.array([xi_eval]), mu1 - eps, mu2, A)[0]) / (2 * eps)
    dmut_dmu2 = (muT_func(np.array([xi_eval]), mu1, mu2 + eps, A)[0] - muT_func(np.array([xi_eval]), mu1, mu2 - eps, A)[0]) / (2 * eps)

    dmut_dH = dmut_dmu1 * dmu1_dH + dmut_dmu2 * dmu2_dH
    dmut_dV = dmut_dmu1 * dmu1_dV + dmut_dmu2 * dmu2_dV
    dmut_dd = dmut_dmu1 * dmu1_dd + dmut_dmu2 * dmu2_dd

    SH = dmut_dH * H / mut if abs(mut) > 1e-12 else 0
    SV = dmut_dV * V / mut if abs(mut) > 1e-12 else 0
    Sd = dmut_dd * d / mut if abs(mut) > 1e-12 else 0
    return {"H*": SH, "V*": SV, "d*": Sd}, {"H*": dmut_dH, "V*": dmut_dV, "d*": dmut_dd}, mut


def plot_normalized_muT_sensitivity(model_name, A, xi_eval, H, V, d, coeffs1, coeffs2):
    S, raw, mut = normalized_muT_sensitivities(model_name, A, xi_eval, H, V, d, coeffs1, coeffs2)
    labels = list(S.keys())
    vals = [S[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(labels, vals)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Normalized sensitivity")
    ax.set_title(r"Normalized sensitivity of $\mu_T$")
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    return fig
