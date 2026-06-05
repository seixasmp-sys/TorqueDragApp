import numpy as np


def xi_min(A: float) -> float:
    return A / (1.0 + A)


# ============================================================
# Friction and drag model library
# ============================================================

def mu_t_logarithmic(xi, mu1, mu2, A):
    xi = np.asarray(xi)
    return mu1 + A * (mu2 - mu1) * (1.0 - xi) / xi


def fd_logarithmic_positive(xi, mu1, mu2, A):
    xi = np.asarray(xi)
    return mu1 * (1.0 - xi) + A * (mu2 - mu1) * ((xi - 1.0) - np.log(xi))


def mu_t_quadratic(xi, mu1, mu2, A=None):
    xi = np.asarray(xi)
    return mu1 + mu2 * (1.0 - xi) ** 2


def fd_quadratic_positive(xi, mu1, mu2, A=None):
    xi = np.asarray(xi)
    return mu1 * (1.0 - xi) + (mu2 / 3.0) * (1.0 - xi) ** 3


def mu_t_linear_decay(xi, mu1, mu2, A=None):
    xi = np.asarray(xi)
    return mu1 + mu2 * (1.0 - xi)


def fd_linear_decay_positive(xi, mu1, mu2, A=None):
    xi = np.asarray(xi)
    return (mu1 + mu2 / 2.0) - (mu1 + mu2) * xi + (mu2 / 2.0) * xi**2


def mu_t_linear_mixture(xi, mu1, mu2, A=None):
    xi = np.asarray(xi)
    return mu1 * xi + mu2 * (1.0 - xi)


def fd_linear_mixture_positive(xi, mu1, mu2, A=None):
    xi = np.asarray(xi)
    return ((mu1 + mu2) / 2.0) - mu2 * xi + ((mu2 - mu1) / 2.0) * xi**2


# ============================================================
# Direct sensitivity of normalized drag to mu1 and mu2
# ============================================================

def sensitivity_log_mu1(xi, A):
    xi = np.asarray(xi)
    return (1.0 - xi) - A * ((xi - 1.0) - np.log(xi))


def sensitivity_log_mu2(xi, A):
    xi = np.asarray(xi)
    return A * ((xi - 1.0) - np.log(xi))


def sensitivity_quad_mu1(xi, A=None):
    xi = np.asarray(xi)
    return 1.0 - xi


def sensitivity_quad_mu2(xi, A=None):
    xi = np.asarray(xi)
    return (1.0 / 3.0) * (1.0 - xi) ** 3


def sensitivity_linear_decay_mu1(xi, A=None):
    xi = np.asarray(xi)
    return 1.0 - xi


def sensitivity_linear_decay_mu2(xi, A=None):
    xi = np.asarray(xi)
    return 0.5 * (1.0 - xi) ** 2


def sensitivity_linear_mixture_mu1(xi, A=None):
    xi = np.asarray(xi)
    return 0.5 * (1.0 - xi**2)


def sensitivity_linear_mixture_mu2(xi, A=None):
    xi = np.asarray(xi)
    return 0.5 * (1.0 - xi) ** 2


# ============================================================
# Model selector
# ============================================================

MODEL_NAMES = [
    "Logarithmic soft-string correction",
    "Quadratic friction correction",
    "Linear decay correction",
    "Linear mixture correction",
]


def get_model_functions(model_name):
    if model_name == "Logarithmic soft-string correction":
        return mu_t_logarithmic, fd_logarithmic_positive, sensitivity_log_mu1, sensitivity_log_mu2
    if model_name == "Quadratic friction correction":
        return mu_t_quadratic, fd_quadratic_positive, sensitivity_quad_mu1, sensitivity_quad_mu2
    if model_name == "Linear decay correction":
        return mu_t_linear_decay, fd_linear_decay_positive, sensitivity_linear_decay_mu1, sensitivity_linear_decay_mu2
    if model_name == "Linear mixture correction":
        return mu_t_linear_mixture, fd_linear_mixture_positive, sensitivity_linear_mixture_mu1, sensitivity_linear_mixture_mu2
    raise ValueError(f"Unknown model name: {model_name}")


def model_latex(model_name):
    if model_name == "Logarithmic soft-string correction":
        return (
            r"\mu_T=\mu_1+A(\mu_2-\mu_1)\frac{1-x_i}{x_i}",
            r"\frac{F_D}{W_{bp}L}=\mu_1(1-x_i)+A(\mu_2-\mu_1)\left[(x_i-1)-\ln(x_i)\right]",
        )
    if model_name == "Quadratic friction correction":
        return (
            r"\mu_T=\mu_1+\mu_2(1-x_i)^2",
            r"\frac{F_D}{W_{bp}L}=\mu_1(1-x_i)+\frac{\mu_2}{3}(1-x_i)^3",
        )
    if model_name == "Linear decay correction":
        return (
            r"\mu_T=\mu_1+\mu_2(1-x_i)",
            r"\frac{F_D}{W_{bp}L}=\left(\mu_1+\frac{\mu_2}{2}\right)-(\mu_1+\mu_2)x_i+\mu_2\frac{x_i^2}{2}",
        )
    if model_name == "Linear mixture correction":
        return (
            r"\mu_T=\mu_1x_i+\mu_2(1-x_i)",
            r"\frac{F_D}{W_{bp}L}=\frac{\mu_1+\mu_2}{2}-\mu_2x_i+(\mu_2-\mu_1)\frac{x_i^2}{2}",
        )
    raise ValueError(f"Unknown model name: {model_name}")


def derivative_latex(model_name):
    if model_name == "Logarithmic soft-string correction":
        return (
            r"\frac{\partial}{\partial \mu_1}\left(\frac{F_D}{W_{bp}L}\right)=(1-x_i)-A\left[(x_i-1)-\ln(x_i)\right]",
            r"\frac{\partial}{\partial \mu_2}\left(\frac{F_D}{W_{bp}L}\right)=A\left[(x_i-1)-\ln(x_i)\right]",
        )
    if model_name == "Quadratic friction correction":
        return (
            r"\frac{\partial}{\partial \mu_1}\left(\frac{F_D}{W_{bp}L}\right)=1-x_i",
            r"\frac{\partial}{\partial \mu_2}\left(\frac{F_D}{W_{bp}L}\right)=\frac{1}{3}(1-x_i)^3",
        )
    if model_name == "Linear decay correction":
        return (
            r"\frac{\partial}{\partial \mu_1}\left(\frac{F_D}{W_{bp}L}\right)=1-x_i",
            r"\frac{\partial}{\partial \mu_2}\left(\frac{F_D}{W_{bp}L}\right)=\frac{(1-x_i)^2}{2}",
        )
    if model_name == "Linear mixture correction":
        return (
            r"\frac{\partial}{\partial \mu_1}\left(\frac{F_D}{W_{bp}L}\right)=\frac{1-x_i^2}{2}",
            r"\frac{\partial}{\partial \mu_2}\left(\frac{F_D}{W_{bp}L}\right)=\frac{(1-x_i)^2}{2}",
        )
    raise ValueError(f"Unknown model name: {model_name}")


def symbolic_endpoint_expressions(model_name):
    if model_name == "Logarithmic soft-string correction":
        return {
            "xi_min": r"\frac{A}{1+A}",
            "xi_max": r"1",
            "muT_min": r"\mu_2",
            "fd_min": r"\frac{\mu_1}{1+A}+A(\mu_2-\mu_1)\left[-\frac{1}{1+A}-\ln\left(\frac{A}{1+A}\right)\right]",
            "muT_max": r"\mu_1",
            "fd_max": r"0",
        }
    if model_name == "Quadratic friction correction":
        return {
            "xi_min": r"\frac{A}{1+A}",
            "xi_max": r"1",
            "muT_min": r"\mu_1+\frac{\mu_2}{(1+A)^2}",
            "fd_min": r"\frac{\mu_1}{1+A}+\frac{\mu_2}{3(1+A)^3}",
            "muT_max": r"\mu_1",
            "fd_max": r"0",
        }
    if model_name == "Linear decay correction":
        return {
            "xi_min": r"\frac{A}{1+A}",
            "xi_max": r"1",
            "muT_min": r"\mu_1+\frac{\mu_2}{1+A}",
            "fd_min": r"\frac{\mu_1}{1+A}+\frac{\mu_2}{2(1+A)^2}",
            "muT_max": r"\mu_1",
            "fd_max": r"0",
        }
    if model_name == "Linear mixture correction":
        return {
            "xi_min": r"\frac{A}{1+A}",
            "xi_max": r"1",
            "muT_min": r"\frac{A\mu_1+\mu_2}{1+A}",
            "fd_min": r"\frac{(2A+1)\mu_1+\mu_2}{2(1+A)^2}",
            "muT_max": r"\mu_1",
            "fd_max": r"0",
        }
    raise ValueError(f"Unknown model name: {model_name}")


# ============================================================
# Candidate mu1/mu2 equation as operational-variable function
# ============================================================

def candidate_mu(H, V, d, c):
    return (
        c["c0"]
        + c["cH"] * H
        + c["cV"] * V
        + c["cd"] * d
        + c["cHH"] * H**2
        + c["cVV"] * V**2
        + c["cdd"] * d**2
        + c["cHV"] * H * V
        + c["cHd"] * H * d
        + c["cVd"] * V * d
    )


def candidate_mu_derivatives(H, V, d, c):
    dmu_dH = c["cH"] + 2 * c["cHH"] * H + c["cHV"] * V + c["cHd"] * d
    dmu_dV = c["cV"] + 2 * c["cVV"] * V + c["cHV"] * H + c["cVd"] * d
    dmu_dd = c["cd"] + 2 * c["cdd"] * d + c["cHd"] * H + c["cVd"] * V
    return dmu_dH, dmu_dV, dmu_dd
