import io
import json
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

try:
    from scipy.optimize import least_squares, differential_evolution, lsq_linear
    SCIPY_OK = True
except Exception:
    least_squares = None
    differential_evolution = None
    lsq_linear = None
    SCIPY_OK = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

APP_DIR = Path(__file__).parent
LIBRARY_PATH = APP_DIR / "td_models_library.json"
CAMPAIGN_PATH = APP_DIR / "td_campaign_library.json"
EXPERIMENT_PATH = APP_DIR / "td_experiment_library.json"
ANALYSIS_PATH = APP_DIR / "td_analysis_library.json"
xi, A, mu1, mu2 = sp.symbols("x_i A mu_1 mu_2", positive=True)
sym_s = sp.symbols("s", positive=True)

st.set_page_config(page_title="T&D Complete App", page_icon="📊", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 980px;}
div.stButton > button, div.stDownloadButton > button {font-weight: 700; border-radius: 10px; min-height: 2.6rem;}
.small-note {font-size: 0.92rem; color: #60646c;}
.eq-box {background: #ffffff; border: 1px solid #e6e8ef; border-radius: 12px; padding: 0.65rem 0.9rem; margin-top: 0.4rem; margin-bottom: 0.75rem;}
.eq-title {font-weight: 700; margin-bottom: 0.15rem;}
[data-testid="stMetricValue"] {font-size: 1.3rem;}
</style>
""", unsafe_allow_html=True)

# ----------------------------- chart style helpers -----------------------------
PLOT_FONT = dict(size=15, color="#000000")
PLOT_TITLE_FONT = dict(size=18, color="#000000")
PLOT_AXIS_TITLE_FONT = dict(size=17, color="#000000")
PLOT_TICK_FONT = dict(size=13, color="#000000")
PLOT_LEGEND_FONT = dict(size=13, color="#000000")

def apply_plotly_text_style(fig):
    """Make Plotly chart text larger and black across the whole app."""
    fig.update_layout(
        font=PLOT_FONT,
        title_font=PLOT_TITLE_FONT,
        legend=dict(font=PLOT_LEGEND_FONT),
    )
    fig.update_xaxes(title_font=PLOT_AXIS_TITLE_FONT, tickfont=PLOT_TICK_FONT, color="#000000")
    fig.update_yaxes(title_font=PLOT_AXIS_TITLE_FONT, tickfont=PLOT_TICK_FONT, color="#000000")
    return fig

# ----------------------------- core helpers -----------------------------
def ensure_library():
    if not LIBRARY_PATH.exists():
        LIBRARY_PATH.write_text("{}", encoding="utf-8")


def sort_models_alpha(models):
    """Return model dictionary ordered alphabetically by model name."""
    if not isinstance(models, dict):
        return {}
    return {k: models[k] for k in sorted(models.keys(), key=lambda v: str(v).casefold())}


def load_models():
    ensure_library()
    try:
        data = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
        return sort_models_alpha(data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_models(models):
    LIBRARY_PATH.write_text(json.dumps(sort_models_alpha(models), ensure_ascii=False, indent=2), encoding="utf-8")




def load_json_dict(path):
    try:
        if not Path(path).exists():
            Path(path).write_text("{}", encoding="utf-8")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_json_dict(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def new_id(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def safe_float(value, default=0.0):
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def clean_name(value):
    return str(value or "").strip()


def name_key(value):
    return clean_name(value).casefold()


def duplicate_campaign_name(campaigns, name, current_id=None):
    nk = name_key(name)
    return any(cid != current_id and name_key(c.get("name")) == nk for cid, c in campaigns.items())


def duplicate_experiment_name(experiments, campaign_id, name, current_id=None):
    nk = name_key(name)
    return any(
        eid != current_id
        and e.get("campaign_id") == campaign_id
        and name_key(e.get("name")) == nk
        for eid, e in experiments.items()
    )


def dataframe_from_uploaded_table(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    return normalize_experimental_df(df)


def dataframe_from_pasted_text(text):
    text = str(text or "").strip()
    if not text:
        return None
    # Excel copy/paste usually comes as tab-separated text. Comma and semicolon files are also accepted.
    sep = "\t" if "\t" in text else (";" if ";" in text and "," not in text.splitlines()[0] else ",")
    return normalize_experimental_df(pd.read_csv(io.StringIO(text), sep=sep))


def normalize_experimental_df(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["x_ft", "FD_N"])
    out = df.copy()
    lowered = {str(c).strip().lower(): c for c in out.columns}

    def find_col(candidates, fallback_index):
        for cand in candidates:
            if cand in lowered:
                return lowered[cand]
        return out.columns[fallback_index] if len(out.columns) > fallback_index else None

    x_col = find_col(["x_ft", "x", "x (ft)", "x_ft ", "position", "position_ft"], 0)
    fd_col = find_col(["fd_n", "fd", "f_d", "force", "force_n", "fd (n)"], 1)
    if x_col is None or fd_col is None:
        return pd.DataFrame(columns=["x_ft", "FD_N"])
    clean = pd.DataFrame({
        "x_ft": pd.to_numeric(out[x_col], errors="coerce"),
        "FD_N": pd.to_numeric(out[fd_col], errors="coerce"),
    }).dropna(subset=["x_ft", "FD_N"])
    return clean.reset_index(drop=True)


def stable_signature(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def analysis_signature(experiment, selected_models, models):
    model_payload = []
    for name in sorted(selected_models):
        m = models.get(name, {})
        model_payload.append({
            "name": name,
            "equation": m.get("equation"),
            "x_min": m.get("x_min"),
            "x_max": m.get("x_max"),
            "integration_type": m.get("integration_type"),
            "integration_limits": m.get("integration_limits"),
            "force_direction": m.get("force_direction", "opposite"),
        })
    return stable_signature({
        "experiment_id": experiment.get("experiment_id"),
        "campaign_id": experiment.get("campaign_id"),
        "A": experiment.get("A"),
        "w_bp_Nm": experiment.get("w_bp_Nm"),
        "experimental_data": experiment.get("experimental_data", []),
        "models": model_payload,
        "fit_method": "bounded_linear_lsq_plus_global_refinement_v22",
        "bounds": {"mu1": [0.0, 1.0], "mu2": [0.0, 1.0]},
    })


def existing_analysis_by_signature(analyses, signature):
    for aid, a in analyses.items():
        if a.get("analysis_signature") == signature:
            return aid, a
    return None, None


def calc_cuttings_area_A(ID_in, OD_in, BH_in):
    """Calculate cuttings area ratio A = Ac/(Aw-Ac).

    Uses the corrected first case: the bed segment is computed with R2, not R1.
    All geometric inputs are in inches; areas are in in² and A is dimensionless.
    """
    ID_in = float(ID_in); OD_in = float(OD_in); BH_in = float(BH_in)
    R1 = OD_in / 2.0
    R2 = ID_in / 2.0
    Aw = np.pi / 4.0 * (ID_in**2 - OD_in**2)
    if ID_in <= 0 or OD_in <= 0 or OD_in >= ID_in:
        raise ValueError("Require 0 < OD < ID.")
    if BH_in < 0 or BH_in > 2 * R2:
        raise ValueError("BH must satisfy 0 <= BH <= ID.")

    def acos_clip(z):
        return np.arccos(np.clip(z, -1.0, 1.0))

    def sqrt_pos(z):
        return np.sqrt(max(float(z), 0.0))

    if BH_in <= R2 - R1 + 1e-12:
        theta2 = 2 * acos_clip((R2 - BH_in) / R2)
        Ac = 0.5 * R2**2 * theta2 - (R2 - BH_in) * sqrt_pos(R2**2 - (R2 - BH_in)**2)
        case = "0 <= BH <= R2 - R1"
    elif BH_in <= R2 + 1e-12:
        theta2 = 2 * acos_clip((R2 - BH_in) / R2)
        theta1 = 2 * acos_clip((R2 - BH_in) / R1)
        Ac = 0.5 * (R2**2 * theta2 - R1**2 * theta1) - (R2 - BH_in) * (
            sqrt_pos(R2**2 - (R2 - BH_in)**2) - sqrt_pos(R1**2 - (R2 - BH_in)**2)
        )
        case = "R2 - R1 < BH <= R2"
    elif BH_in <= R2 + R1 + 1e-12:
        d = BH_in - R2
        theta2 = 2 * acos_clip(d / R2)
        theta1 = 2 * acos_clip(d / R1)
        outer_empty = 0.5 * R2**2 * theta2 - d * sqrt_pos(R2**2 - d**2)
        inner_empty = 0.5 * R1**2 * theta1 - d * sqrt_pos(R1**2 - d**2)
        Ac = Aw - outer_empty + inner_empty
        case = "R2 < BH <= R2 + R1"
    else:
        d = BH_in - R2
        theta2 = 2 * acos_clip(d / R2)
        outer_empty = 0.5 * R2**2 * theta2 - d * sqrt_pos(R2**2 - d**2)
        Ac = Aw - outer_empty
        case = "R2 + R1 < BH <= 2R2"

    Ac = min(max(float(Ac), 0.0), Aw - 1e-12)
    A_val = Ac / (Aw - Ac)
    return {"A": float(A_val), "Ac_in2": float(Ac), "Aw_in2": float(Aw), "case": case, "R1_in": R1, "R2_in": R2}


def calc_wbp_Nm(rho_f_ppg, rho_p_kgm3, w_pp_Nm):
    rho_f_kgm3 = float(rho_f_ppg) * 119.8264273
    rho_p_kgm3 = float(rho_p_kgm3)
    w_pp_Nm = float(w_pp_Nm)
    if rho_p_kgm3 <= 0:
        raise ValueError("rho_p must be positive.")
    return float(w_pp_Nm * (1.0 - rho_f_kgm3 / rho_p_kgm3))


def experiment_dataframe_from_record(exp):
    rows = exp.get("experimental_data", []) or []
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["x_ft", "FD_N"])
    for col in ["x_ft", "FD_N"]:
        if col not in df.columns:
            df[col] = np.nan
    return df[["x_ft", "FD_N"]]


def prepared_exp_data(campaign, experiment):
    df = experiment_dataframe_from_record(experiment).copy()
    df["x_ft"] = pd.to_numeric(df["x_ft"], errors="coerce")
    df["FD_N"] = pd.to_numeric(df["FD_N"], errors="coerce")
    df = df.dropna(subset=["x_ft", "FD_N"])
    L_ft = float(campaign.get("L_ft", 0))
    wbp = float(experiment.get("w_bp_Nm", 0))
    if L_ft <= 0 or wbp == 0:
        raise ValueError("Campaign L and experiment w_bp must be valid before fitting.")
    L_m = L_ft * 0.3048
    df["x_i"] = df["x_ft"] / L_ft
    df["FD_over_wbpL"] = df["FD_N"] / (wbp * L_m)
    df = df[(df["x_i"] >= 0) & (df["x_i"] <= 1)]
    return df


def valid_x_range_for_model(model, A_val):
    xmin = evaluate_limit(model.get("x_min", "0"), A_val)
    xmax = evaluate_limit(model.get("x_max", "1"), A_val)
    xmin = 0.0 if xmin is None else float(xmin)
    xmax = 1.0 if xmax is None else float(xmax)
    xmin, xmax = sorted([xmin, xmax])
    return max(0.0, xmin), min(1.0, xmax)


def r2_bounded(y_values, y_fit):
    y_values = np.asarray(y_values, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    resid = y_values - y_fit
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_values - np.mean(y_values))**2))
    if ss_tot <= 1e-15:
        raw = 1.0 if ss_res <= 1e-12 else 0.0
    else:
        raw = 1.0 - ss_res / ss_tot
    return float(np.clip(raw, 0.0, 1.0)), float(raw)


def fit_one_model(model, A_val, x_i_values, y_values):
    """Fit mu1 and mu2 for one model.

    The first fitting attempt uses bounded linear least squares. This is more stable
    for the current T&D model library because the model output Fd/(Wbp L) is linear
    in mu1 and mu2 after integration. A nonlinear least_squares refinement is then
    tested and the candidate with the lowest SSE is retained.
    """
    if not SCIPY_OK:
        raise RuntimeError("scipy is required for fitting. Install with: pip install scipy")

    x_i_values = np.asarray(x_i_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    xmin, xmax = valid_x_range_for_model(model, A_val)
    valid = np.isfinite(x_i_values) & np.isfinite(y_values) & (x_i_values >= xmin - 1e-10) & (x_i_values <= xmax + 1e-10)
    x_fit_data = x_i_values[valid]
    y_fit_data = y_values[valid]

    if len(x_fit_data) < 2:
        raise ValueError(
            f"Not enough valid experimental points for this model range. "
            f"At least two points are required to estimate mu1 and mu2. "
            f"Model valid range is {xmin:.6g} <= x_i <= {xmax:.6g}; valid points found: {len(x_fit_data)}."
        )

    # Compile the integrated model only once for this fit. Recomputing the
    # symbolic integral inside every optimizer evaluation made the fitting module
    # unnecessarily slow and, in practice, could look like the app had stopped.
    try:
        fd_expr_cached = fd_symbolic(model)
        fd_func_cached = sp.lambdify((xi, A, mu1, mu2), fd_expr_cached, "numpy")

        def fd_fast(xdata, m1, m2):
            return as_curve(fd_func_cached(xdata, A_val, m1, m2), xdata)
    except Exception:
        def fd_fast(xdata, m1, m2):
            return fd_curve_numeric(model, xdata, A_val, m1, m2)

    eps = 1e-8
    lower = np.array([eps, eps], dtype=float)
    upper = np.array([1.0 - eps, 1.0 - eps], dtype=float)

    def predict(params):
        m1, m2 = params
        y_pred = fd_fast(x_fit_data, m1, m2)
        return np.asarray(y_pred, dtype=float)

    def residual(params):
        y_pred = predict(params)
        return np.nan_to_num(y_pred - y_fit_data, nan=1e9, posinf=1e9, neginf=-1e9)

    def sse(params):
        r = residual(params)
        return float(np.sum(r**2))

    candidates = []

    # Method 1: bounded linear least squares using model basis curves.
    # y = g0 + mu1*(g1-g0) + mu2*(g2-g0)
    try:
        g0 = np.asarray(fd_fast(x_fit_data, 0.0, 0.0), dtype=float)
        g1 = np.asarray(fd_fast(x_fit_data, 1.0, 0.0), dtype=float)
        g2 = np.asarray(fd_fast(x_fit_data, 0.0, 1.0), dtype=float)
        B = np.column_stack([g1 - g0, g2 - g0])
        target = y_fit_data - g0
        finite_rows = np.isfinite(target) & np.all(np.isfinite(B), axis=1)
        if np.count_nonzero(finite_rows) >= 2:
            lin = lsq_linear(B[finite_rows], target[finite_rows], bounds=(lower, upper), method="trf", lsmr_tol="auto")
            params = np.clip(np.asarray(lin.x, dtype=float), lower, upper)
            candidates.append({"params": params, "sse": sse(params), "method": "bounded linear least squares", "success": bool(lin.success), "message": str(lin.message)})
    except Exception as e:
        candidates.append({"params": np.array([0.25, 0.50]), "sse": np.inf, "method": "bounded linear least squares failed", "success": False, "message": str(e)})

    # Method 2: nonlinear least_squares from multiple starts, including the linear solution.
    starts = [np.array([0.25, 0.50], dtype=float), np.array([0.10, 0.10], dtype=float), np.array([0.50, 0.50], dtype=float), np.array([0.80, 0.20], dtype=float), np.array([0.20, 0.80], dtype=float)]
    if candidates and np.isfinite(candidates[0].get("sse", np.inf)):
        starts.insert(0, candidates[0]["params"])

    for x0 in starts:
        try:
            res = least_squares(
                residual,
                x0=np.clip(x0, lower, upper),
                bounds=(lower, upper),
                method="trf",
                x_scale="jac",
                loss="linear",
                max_nfev=12000,
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )
            params = np.clip(np.asarray(res.x, dtype=float), lower, upper)
            candidates.append({"params": params, "sse": sse(params), "method": "least_squares refinement", "success": bool(res.success), "message": str(res.message)})
        except Exception as e:
            candidates.append({"params": np.clip(x0, lower, upper), "sse": np.inf, "method": "least_squares failed", "success": False, "message": str(e)})

    # The bounded linear solution plus nonlinear refinements are normally enough
    # for the current model library. A slower global search is intentionally not
    # run here to keep the Streamlit module responsive.

    valid_candidates = [c for c in candidates if np.isfinite(c.get("sse", np.inf))]
    if not valid_candidates:
        messages = "; ".join([f"{c.get('method')}: {c.get('message')}" for c in candidates])
        raise ValueError(f"Model fitting failed for this model. Details: {messages}")

    best = min(valid_candidates, key=lambda c: c["sse"])
    mu1_fit, mu2_fit = [float(v) for v in best["params"]]
    y_fit = predict([mu1_fit, mu2_fit])
    resid = y_fit_data - y_fit
    rmse = float(np.sqrt(np.mean(resid**2))) if len(resid) else np.nan
    r2, r2_raw = r2_bounded(y_fit_data, y_fit)

    return {
        "mu1": mu1_fit,
        "mu2": mu2_fit,
        "R2": r2,
        "R2_raw": r2_raw,
        "RMSE": rmse,
        "success": bool(best.get("success", True)),
        "message": str(best.get("message", "")),
        "method": str(best.get("method", "bounded linear least squares")),
        "n_points_total": int(len(x_i_values)),
        "n_points_used": int(len(x_fit_data)),
        "x_valid_min": float(xmin),
        "x_valid_max": float(xmax),
        "cost": float(best["sse"] / 2.0),
    }


def export_df_xlsx_bytes(df, sheet_name="Results"):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    buf.seek(0)
    return buf.getvalue()


def export_df_txt_bytes(df):
    return df.to_csv(sep="\t", index=False).encode("utf-8")


def fitted_plot_png_bytes(title, x_exp, y_exp, curve_items, xlabel="x_i", ylabel="Fd/(WbpL)"):
    if not MATPLOTLIB_OK:
        return None
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.scatter(x_exp, y_exp, label="Experimental data")
    for label, x, y in curve_items:
        ax.plot(x, y, label=label)
    ax.set_title(title, fontsize=14, color="black")
    ax.set_xlabel(xlabel, fontsize=13, color="black")
    ax.set_ylabel(ylabel, fontsize=13, color="black")
    ax.tick_params(axis="both", labelsize=11, colors="black")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=10)
    for spine in ax.spines.values():
        spine.set_color("black")
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.90)
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def normalize_equation(text: str) -> str:
    s = "" if text is None else str(text).strip()
    repl = {
        "μ₁":"mu_1", "µ₁":"mu_1", "μ1":"mu_1", "µ1":"mu_1", "mu1":"mu_1", "μ_1":"mu_1", "µ_1":"mu_1",
        "μ₂":"mu_2", "µ₂":"mu_2", "μ2":"mu_2", "µ2":"mu_2", "mu2":"mu_2", "μ_2":"mu_2", "µ_2":"mu_2",
        "xᵢ":"x_i", "xi":"x_i", "X_i":"x_i", "Xᵢ":"x_i", "^":"**", "−":"-", "–":"-", "×":"*",
    }
    for a,b in repl.items():
        s = s.replace(a,b)
    if "=" in s:
        s = s.split("=",1)[1]
    return s


def parse_expr_safe(text: str):
    local = {"x_i":xi, "A":A, "mu_1":mu1, "mu_2":mu2, "ln":sp.log, "log":sp.log, "sqrt":sp.sqrt, "exp":sp.exp, "sin":sp.sin, "cos":sp.cos, "tan":sp.tan}
    return sp.sympify(normalize_equation(text), locals=local)


def pretty_expr(expr):
    try:
        return sp.latex(sp.nsimplify(sp.simplify(expr), rational=False))
    except Exception:
        return str(expr)


def pretty_expr_expanded(expr):
    """LaTeX for equations shown in reports: simplified but not factored."""
    try:
        clean = sp.nsimplify(sp.simplify(expr), rational=False)
        return sp.latex(sp.expand(clean))
    except Exception:
        return pretty_expr(expr)


def fd_display_symbolic(model):
    """Fd expression for display, expanded to avoid factored form."""
    return sp.expand(sp.simplify(fd_symbolic(model)))


def fd_sign(model):
    """Return the integration sign according to the force direction.

    Old model libraries do not contain this field; they keep the original
    convention: Fd is opposite to x_i, so the integral receives a negative sign.
    New models can set force_direction='same' to use a positive integral.
    """
    direction = str(model.get("force_direction", "opposite")).lower().strip()
    return 1 if direction in {"same", "positive", "+", "same_as_xi", "same as x_i direction"} else -1


def fd_direction_label(model):
    direction = str(model.get("force_direction", "opposite")).lower().strip()
    if direction in {"same", "positive", "+", "same_as_xi", "same as x_i direction"}:
        return "same as x_i direction"
    return "opposite to x_i direction"


def fd_integral_symbol_latex(model):
    return "+" if fd_sign(model) > 0 else "-"


def canonical_point_key(point_text):
    try:
        return sp.srepr(sp.simplify(parse_expr_safe(point_text)))
    except Exception:
        return normalize_equation(point_text).replace(" ", "")


def plain_expr(expr):
    try:
        return str(sp.simplify(expr)).replace("mu_1", "mu1").replace("mu_2", "mu2")
    except Exception:
        return str(expr)


def evaluate_limit(limit_text, a_value):
    try:
        val = float(parse_expr_safe(limit_text).subs({A:a_value, mu1:1, mu2:1}))
        return val if np.isfinite(val) else None
    except Exception:
        return None


def model_expr(model):
    return parse_expr_safe(model.get("equation", "0"))


def integration_limit_expr(limit_text):
    """Parse integration limits while keeping x_i as the external variable, not the dummy variable."""
    if normalize_equation(limit_text).replace(" ", "") == "x_i":
        return xi
    return parse_expr_safe(limit_text)


def fd_symbolic(model):
    expr_s = model_expr(model).subs(xi, sym_s)
    limits = model.get("integration_limits", []) or []
    typ = model.get("integration_type", "option1")
    sign = fd_sign(model)
    if typ == "option1" and len(limits) >= 2:
        a_lim = integration_limit_expr(limits[0])
        b_lim = integration_limit_expr(limits[1])
        return sp.simplify(sign * sp.integrate(expr_s, (sym_s, a_lim, b_lim)))
    if typ == "option2":
        ref = integration_limit_expr(limits[0] if limits else "1")
        return sp.simplify(sign * sp.integrate(expr_s, (sym_s, ref, xi)))
    return sp.simplify(sign * sp.integrate(expr_s, (sym_s, 1, xi)))


def reference_point(model, a_value):
    limits = model.get("integration_limits", []) or []
    typ = model.get("integration_type", "option1")
    if typ == "option1" and len(limits) >= 2:
        for lim in limits:
            if normalize_equation(lim).replace(" ","") not in ["x_i"]:
                v = evaluate_limit(lim, a_value)
                return 1.0 if v is None else v
    if typ == "option2" and limits:
        v = evaluate_limit(limits[0], a_value)
        return 1.0 if v is None else v
    return 1.0


def x_grid(model, a_value, n=260, force_0_1=False):
    if force_0_1:
        xmin, xmax = 0.0005, 1.0
    else:
        xmin = evaluate_limit(model.get("x_min","0"), a_value)
        xmax = evaluate_limit(model.get("x_max","1"), a_value)
        xmin = 0.0 if xmin is None else xmin
        xmax = 1.0 if xmax is None else xmax
        xmin, xmax = sorted([xmin, xmax])
        xmin, xmax = max(0.0005, xmin), min(1.0, xmax)
    if np.isclose(xmin, xmax):
        xmax = min(1.0, xmin+0.01)
    return np.linspace(xmin, xmax, n)


def as_curve(vals, x):
    arr = np.asarray(vals, dtype=float)
    if arr.ndim == 0:
        arr = np.full_like(np.asarray(x, dtype=float), float(arr))
    arr[~np.isfinite(arr)] = np.nan
    return arr


def mu_curve(model, x, av, m1v, m2v):
    f = sp.lambdify((xi,A,mu1,mu2), model_expr(model), "numpy")
    return as_curve(f(x, av, m1v, m2v), x)


def fd_curve_symbolic(model, x, av, m1v, m2v):
    f = sp.lambdify((xi,A,mu1,mu2), fd_symbolic(model), "numpy")
    return as_curve(f(x, av, m1v, m2v), x)


def fd_curve_numeric(model, x, av, m1v, m2v):
    x = np.asarray(x, dtype=float)
    ref = reference_point(model, av)
    lo, hi = float(np.nanmin(np.r_[x, ref])), float(np.nanmax(np.r_[x, ref]))
    dense = np.unique(np.r_[np.linspace(lo, hi, max(900, 4*len(x))), x, ref])
    y = mu_curve(model, dense, av, m1v, m2v)
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0)
    cum = np.r_[0, np.cumsum(0.5*(y[:-1]+y[1:])*np.diff(dense))]
    return fd_sign(model) * (np.interp(x, dense, cum) - float(np.interp(ref, dense, cum)))


def fd_curve(model, x, av, m1v, m2v):
    try:
        y = fd_curve_symbolic(model, x, av, m1v, m2v)
        if np.all(np.isnan(y)):
            raise ValueError("all nan")
        return y
    except Exception:
        return fd_curve_numeric(model, x, av, m1v, m2v)


def selected_curve(model, output, x, av, m1v, m2v):
    return fd_curve(model, x, av, m1v, m2v) if output.startswith("Fd") else mu_curve(model, x, av, m1v, m2v)


def safe_symbolic_sub(expr, point_expr):
    try:
        p = parse_expr_safe(point_expr) if isinstance(point_expr, str) else point_expr
        val = sp.simplify(expr.subs(xi, p))
        if val.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
            return None
        return val
    except Exception:
        return None


def symbolic_reference_items(model):
    """Symbolic values only at this model's own endpoints.

    The Summary/PDF must not include additional generic points such as
    A/(A+1) or 1/(A+1) unless those are actually the model x_min/x_max.
    """
    mu_expr = model_expr(model)
    fd_expr = fd_symbolic(model)
    raw_points = [(r"x_{i,\min}", model.get("x_min","0")), (r"x_{i,\max}", model.get("x_max","1"))]
    seen = set(); items=[]
    for endpoint_name, point_text in raw_points:
        key = canonical_point_key(point_text)
        if key in seen:
            continue
        seen.add(key)
        mu_val = safe_symbolic_sub(mu_expr, point_text)
        fd_val = safe_symbolic_sub(fd_expr, point_text)
        p_ltx = pretty_expr(parse_expr_safe(point_text))
        items.append({"endpoint_name": endpoint_name, "point_latex": p_ltx, "mu": mu_val, "fd": fd_val})
    return items


def endpoint_rows(model, av, m1v, m2v):
    # Kept for sensitivity module numerical reference table.
    xs = []
    for label, lim in [("x_min", model.get("x_min","0")), ("x_max", model.get("x_max","1")), ("A/(A+1)", "A/(A+1)"), ("1/(A+1)", "1/(A+1)")]:
        v = evaluate_limit(lim, av)
        if v is not None and 0 <= v <= 1:
            xs.append((label, v))
    rows=[]
    for label,v in xs:
        rows.append({"Point":label, "x_i":v, "mu_eff":float(mu_curve(model, np.array([v]), av, m1v, m2v)[0]), "Fd/(Wbp.L)":float(fd_curve(model, np.array([v]), av, m1v, m2v)[0])})
    return pd.DataFrame(rows)


def axis_label(label):
    """Return Plotly-safe HTML labels instead of raw LaTeX strings.
    Raw Plotly math strings can appear literally/vertically in Plotly axis titles on some systems.
    """
    s = str(label)
    if "mu" in s or "\\mu" in s or "µ" in s or "μ" in s:
        return "μ<sub>eff</sub>"
    if "Fd" in s or "F_d" in s or "Wbp" in s:
        return "F<sub>d</sub>/(W<sub>bp</sub>L)"
    if s.startswith("d("):
        return s.replace("mu", "μ")
    return s.replace("$", "")


def plot_layout(fig, title, ytitle, height=520, show_title=True):
    fig.update_layout(
        title_text=(str(title) if show_title and title else ""),
        xaxis_title="x<sub>i</sub>",
        yaxis_title=axis_label(ytitle),
        template="plotly_white",
        height=height,
        width=760,
        margin=dict(l=82,r=30,t=34 if not show_title else 62,b=76),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_xaxes(showgrid=True, zeroline=False, title_standoff=14)
    fig.update_yaxes(showgrid=True, zeroline=False, title_standoff=16)
    return apply_plotly_text_style(fig)


def st_plot(fig, key=None):
    if "_plotly_unique_counter" not in st.session_state:
        st.session_state["_plotly_unique_counter"] = 0
    st.session_state["_plotly_unique_counter"] += 1
    unique_key = key or f"plotly_chart_{st.session_state['_plotly_unique_counter']}"
    fig = apply_plotly_text_style(fig)
    st.plotly_chart(
        fig,
        width="content",
        config={"displaylogo": False, "responsive": False},
        key=unique_key,
    )


def render_equation_block(models, selected, equation_type):
    st.markdown('<div class="eq-box">', unsafe_allow_html=True)
    for name in selected:
        m = models[name]
        try:
            st.markdown(f"<div class='eq-title'>{name}</div>", unsafe_allow_html=True)
            if equation_type == "mu":
                st.latex(r"\mu_{eff} = " + pretty_expr(model_expr(m)))
            else:
                st.latex(r"\frac{F_d}{W_{bp}L} = " + pretty_expr_expanded(fd_display_symbolic(m)))
        except Exception as e:
            st.warning(f"Could not render equation for {name}: {e}")
    st.markdown('</div>', unsafe_allow_html=True)


def render_symbolic_model_section(name, model):
    st.markdown(f"### {name}")
    try:
        st.latex(r"\mu_{eff} = " + pretty_expr(model_expr(model)))
        st.latex(r"\frac{F_d}{W_{bp}L} = " + pretty_expr_expanded(fd_display_symbolic(model)))
        st.caption("Force direction: " + fd_direction_label(model))
        st.latex(r"\frac{F_d}{W_{bp}L} = " + fd_integral_symbol_latex(model) + r"\int \mu_{eff}\,dx_i")
        try:
            xmin_ltx = pretty_expr(parse_expr_safe(model.get('x_min','0')))
            xmax_ltx = pretty_expr(parse_expr_safe(model.get('x_max','1')))
            st.latex(r"x_i \in \left[" + xmin_ltx + r",\ " + xmax_ltx + r"\right]")
        except Exception:
            st.latex(r"x_i \in \left[" + str(model.get('x_min','')) + r",\ " + str(model.get('x_max','')) + r"\right]")
        st.markdown("**Symbolic reference values**")
        for item in symbolic_reference_items(model):
            point = item["point_latex"]
            endpoint = item.get("endpoint_name", "")
            left = (endpoint + r"=" + point) if endpoint else point
            if item["mu"] is None:
                st.latex(r"\mu_{eff}\left(" + left + r"\right) = \text{undefined}")
            else:
                st.latex(r"\mu_{eff}\left(" + left + r"\right) = " + pretty_expr(item["mu"]))
            if item["fd"] is None:
                st.latex(r"\frac{F_d}{W_{bp}L}\left(" + left + r"\right) = \text{undefined}")
            else:
                st.latex(r"\frac{F_d}{W_{bp}L}\left(" + left + r"\right) = " + pretty_expr_expanded(item["fd"]))
    except Exception as e:
        st.error(f"Error in summary for {name}: {e}")

# ----------------------------- pdf helpers -----------------------------
def latex_png(latex_text, fontsize=13, width=7.2):
    """Render one math expression to a transparent PNG.

    This function never raises into the PDF builder. When matplotlib mathtext
    cannot parse a formula, the caller falls back to safe plain text instead
    of placing an error message in the report.
    """
    if not MATPLOTLIB_OK:
        return None
    try:
        buf = io.BytesIO()
        # Start wide enough for long symbolic expressions; bbox_inches will crop.
        fig = plt.figure(figsize=(width, 0.55), dpi=220)
        fig.patch.set_alpha(0)
        plt.axis('off')
        plt.text(0.0, 0.5, f"${latex_text}$", fontsize=fontsize, va='center', ha='left')
        plt.savefig(buf, format="png", dpi=220, transparent=True, bbox_inches="tight", pad_inches=0.035)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception:
        try:
            plt.close('all')
        except Exception:
            pass
        return None


def safe_plain_math_text(latex_text):
    """Conservative fallback for PDF equations if math rendering fails."""
    txt = str(latex_text)
    replacements = {
        r"\mu_{eff}": "mu_eff",
        r"\mu_1": "mu1", r"\mu_{1}": "mu1",
        r"\mu_2": "mu2", r"\mu_{2}": "mu2",
        r"\frac{F_d}{W_{bp}L}": "Fd/(Wbp L)",
        r"\left": "", r"\right": "", r"\quad": " ",
        r"\log": "log", r"\in": " in ",
        r"\min": "min", r"\max": "max",
    }
    for a, b in replacements.items():
        txt = txt.replace(a, b)
    txt = txt.replace('{','').replace('}','').replace('\\','')
    return txt


def add_latex(story, latex_text, width=7.2*inch, max_height=None, fontsize=12):
    """Add a LaTeX equation image to a ReportLab story with preserved aspect ratio."""
    img_buf = latex_png(latex_text, fontsize=fontsize)
    styles = getSampleStyleSheet()
    if not img_buf:
        story.append(Paragraph(safe_plain_math_text(latex_text), styles['Normal']))
        return
    try:
        from PIL import Image as PILImage
        img_buf.seek(0)
        pil = PILImage.open(img_buf)
        px_w, px_h = pil.size
        img_buf.seek(0)
        target_w = width
        target_h = target_w * (px_h / max(px_w, 1))
        if max_height and target_h > max_height:
            target_h = max_height
            target_w = target_h * (px_w / max(px_h, 1))
        story.append(Image(img_buf, width=target_w, height=target_h))
    except Exception:
        story.append(Image(img_buf, width=width, height=(max_height or 0.35*inch)))

def make_matplotlib_chart(models, selected, output, av, m1v, m2v):
    """Build a PNG chart for the PDF report.

    Use plain Unicode/ASCII labels instead of Matplotlib mathtext here.
    This avoids ParseException errors when the local Matplotlib/Python
    installation has trouble parsing LaTeX-like labels during tight_layout().
    """
    if not MATPLOTLIB_OK:
        return None
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(7.1, 4.8))
    for name in selected:
        model = models[name]
        x = x_grid(model, av)
        y = selected_curve(model, output, x, av, m1v, m2v)
        ax.plot(x, y, label=name)
    ax.set_xlabel("xᵢ", fontsize=15, color="black")
    ax.set_ylabel("μeff" if output == "mu_eff" else "Fd/(WbpL)", fontsize=15, color="black")
    ax.tick_params(axis="both", labelsize=12, colors="black")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=11, loc="best")
    for spine in ax.spines.values():
        spine.set_color("black")
    # Avoid tight_layout because it can trigger mathtext parsing errors in some environments.
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.14, top=0.96)
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    buf.seek(0)
    return buf


def comparison_pdf(models, selected, av, m1v, m2v):
    if not REPORTLAB_OK:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=34, leftMargin=34, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    small = ParagraphStyle('small', parent=styles['BodyText'], fontSize=8.4, leading=10.5)
    story = [Paragraph("T&D Model Comparison Report", styles['Title']), Spacer(1,8)]

    # Copy the same content shown in the Summary tab, model by model.
    story.append(Paragraph("Summary", styles['Heading2']))
    for idx, name in enumerate(selected):
        m = models[name]
        story.append(Spacer(1,7))
        story.append(Paragraph(name, styles['Heading3']))
        try:
            add_latex(story, r"\mu_{eff}=" + pretty_expr(model_expr(m)), width=7.2*inch, max_height=0.38*inch, fontsize=11)
            add_latex(story, r"\frac{F_d}{W_{bp}L}=" + pretty_expr_expanded(fd_display_symbolic(m)), width=7.2*inch, max_height=0.38*inch, fontsize=10)
            story.append(Paragraph("Force direction: " + fd_direction_label(m), small))
            add_latex(story, r"\frac{F_d}{W_{bp}L}=" + fd_integral_symbol_latex(m) + r"\int \mu_{eff}\,dx_i", width=4.8*inch, max_height=0.30*inch, fontsize=10)
            try:
                xmin_ltx = pretty_expr(parse_expr_safe(m.get('x_min','0')))
                xmax_ltx = pretty_expr(parse_expr_safe(m.get('x_max','1')))
                add_latex(story, r"x_i\in\left[" + xmin_ltx + r",\ " + xmax_ltx + r"\right]", width=4.8*inch, max_height=0.28*inch, fontsize=11)
            except Exception:
                story.append(Paragraph(f"x interval: {m.get('x_min','')} to {m.get('x_max','')}", small))
            story.append(Paragraph("Symbolic reference values", small))
            for item in symbolic_reference_items(m):
                point = item["point_latex"]
                endpoint = item.get("endpoint_name", "")
                left = (endpoint + r"=" + point) if endpoint else point
                mu_rhs = r"\text{undefined}" if item["mu"] is None else pretty_expr(item["mu"])
                fd_rhs = r"\text{undefined}" if item["fd"] is None else pretty_expr_expanded(item["fd"])
                add_latex(story, r"\mu_{eff}\left(" + left + r"\right)=" + mu_rhs, width=7.2*inch, max_height=0.32*inch, fontsize=9)
                add_latex(story, r"\frac{F_d}{W_{bp}L}\left(" + left + r"\right)=" + fd_rhs, width=7.2*inch, max_height=0.32*inch, fontsize=8)
        except Exception as e:
            story.append(Paragraph(f"Error rendering symbolic equations: {e}", small))
        if idx < len(selected)-1:
            story.append(Spacer(1,5))

    plot_cases = [
        ("Base case", r"\mu_1 < \mu_2", m1v, m2v),
        ("Inverse friction case", r"\mu_1 > \mu_2", 0.60, 0.30),
    ]
    for case_label, case_latex, cm1, cm2 in plot_cases:
        for out in ["mu_eff", "Fd/(Wbp.L)"]:
            try:
                img = make_matplotlib_chart(models, selected, out, av, cm1, cm2)
            except Exception as chart_error:
                img = None
                story.append(Paragraph(f"Chart rendering skipped: {chart_error}", styles['Normal']))
            if img:
                story.append(PageBreak())
                story.append(Paragraph(case_label, styles['Heading2']))
                add_latex(story, case_latex, width=2.0*inch, max_height=0.25*inch, fontsize=13)
                add_latex(story, rf"A={av:g},\quad \mu_1={cm1:g},\quad \mu_2={cm2:g}", width=4.9*inch, max_height=0.30*inch, fontsize=12)
                story.append(Image(img, width=6.9*inch, height=4.65*inch))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def sensitivity_pdf(models, model_name, av, m1v, m2v):
    if not REPORTLAB_OK:
        return None
    model = models[model_name]
    buffer = io.BytesIO(); doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=34, leftMargin=34, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet(); story=[Paragraph("T&D Sensitivity Analysis Report", styles['Title']), Paragraph(f"Model: {model_name}", styles['Heading2'])]
    if MATPLOTLIB_OK:
        add_latex(story, r"\mu_{eff}=" + pretty_expr(model_expr(model)), width=7.3*inch, max_height=0.45*inch, fontsize=11)
        add_latex(story, r"\frac{F_d}{W_{bp}L}=" + pretty_expr_expanded(fd_display_symbolic(model)), width=7.3*inch, max_height=0.45*inch, fontsize=11)
        add_latex(story, rf"A={av:g},\quad \mu_1={m1v:g},\quad \mu_2={m2v:g}", width=4.9*inch, max_height=0.35*inch, fontsize=12)
    else:
        story.append(Paragraph(f"mu_eff(x_i) = {plain_expr(model_expr(model))}", styles['Normal']))
        story.append(Paragraph(f"Fd/(Wbp.L) = {plain_expr(fd_display_symbolic(model))}", styles['Normal']))
        story.append(Paragraph(f"Base parameters: A={av}, mu1={m1v}, mu2={m2v}", styles['Normal']))
    story.append(Spacer(1,10))
    df = endpoint_rows(model, av, m1v, m2v).round(6)
    data = [list(df.columns)] + df.astype(str).values.tolist()
    tbl = Table(data, repeatRows=1); tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eef7')),('GRID',(0,0),(-1,-1),0.35,colors.grey),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold')]))
    story.append(Paragraph("Reference point table", styles['Heading2'])); story.append(tbl)
    doc.build(story); buffer.seek(0); return buffer.getvalue()

# ----------------------------- modules -----------------------------
def library_module(models):
    st.title("📚 Shared Model Library")
    st.caption("Todos os modelos salvos aqui ficam disponíveis em comparação e sensibilidade.")
    if models:
        rows=[]
        for name,m in models.items():
            rows.append({"Model":name,"Equation":m.get('equation',''),"x_min":m.get('x_min',''),"x_max":m.get('x_max',''),"Force direction":fd_direction_label(m),"Integration":m.get('integration_type',''),"Limits/reference":', '.join(m.get('integration_limits',[]) or []),"Date":m.get('date','')})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        with st.expander("Equation preview", expanded=False):
            for name,m in models.items():
                try:
                    st.markdown(f"**{name}:**")
                    st.latex(r"\mu_{eff}=" + pretty_expr(model_expr(m)))
                    st.caption("Force direction: " + fd_direction_label(m))
                    st.latex(r"\frac{F_d}{W_{bp}L} = " + fd_integral_symbol_latex(m) + r"\int \mu_{eff}\,dx_i")
                except Exception:
                    st.markdown(f"**{name}:** `{m.get('equation','')}`")
    else:
        st.warning("No models in the library.")

    st.subheader("Add / edit model")
    st.info("Para mudar o sinal da integral, edite o modelo aqui e escolha a direção da força em relação a xᵢ.")
    edit_existing = st.checkbox("Load an existing model to edit")
    selected = st.selectbox("Existing model", list(models.keys())) if edit_existing and models else None
    base = models[selected] if selected else {}
    with st.form("model_form"):
        c1,c2=st.columns([1,2])
        name=c1.text_input("Model name", value=selected or "New model")
        equation=c2.text_input("mu_eff equation", value=base.get('equation',"mu_1+mu_2*(1-x_i)"))
        c3,c4,c5=st.columns(3)
        x_min=c3.text_input("x_min", value=base.get('x_min',"A/(1+A)"))
        x_max=c4.text_input("x_max", value=base.get('x_max',"1"))
        int_type=c5.selectbox("Integration type", ["option1","option2"], index=0 if base.get('integration_type','option1')=='option1' else 1)
        default_limits = ', '.join(base.get('integration_limits', ['1','x_i'] if int_type=='option1' else ['1']))
        limits_text=st.text_input("Integration limits/reference", value=default_limits, help="option1: use two limits, e.g. 1, x_i. option2: use one reference point, e.g. 1.")
        direction_options = ["Opposite to x_i direction (- integral)", "Same as x_i direction (+ integral)"]
        current_direction = str(base.get('force_direction', 'opposite')).lower().strip()
        force_direction_choice = st.radio(
            "Force direction relative to x_i / Direção de Fd em relação a x_i",
            direction_options,
            index=1 if current_direction in {"same", "positive", "+", "same_as_xi"} else 0,
            horizontal=True,
            help="Use negative integral when Fd is opposite to dx_i; use positive integral when Fd has the same direction as dx_i."
        )
        if st.form_submit_button("Save model", type="primary"):
            try:
                parse_expr_safe(equation); parse_expr_safe(x_min); parse_expr_safe(x_max)
                limits=[p.strip() for p in limits_text.split(',') if p.strip()]
                for p in limits: parse_expr_safe(p)
                models[name.strip()]={"equation":equation.strip(),"x_min":x_min.strip(),"x_max":x_max.strip(),"integration_type":int_type,"integration_limits":limits,"force_direction":"same" if force_direction_choice.startswith("Same") else "opposite","date":datetime.now().strftime("%d/%m/%Y %H:%M")}
                save_models(models); st.success("Model saved in the shared library."); st.rerun()
            except Exception as e: st.error(f"Could not save model: {e}")
    st.divider()
    c1,c2=st.columns(2)
    c1.download_button("Download model library (.json)", json.dumps(models, ensure_ascii=False, indent=2), "td_models_library.json", "application/json")
    if models:
        del_name=c2.selectbox("Delete model", list(models.keys()), key="delete_model")
        if c2.button("Delete selected model"):
            models.pop(del_name,None); save_models(models); st.rerun()


def comparison_module(models):
    st.title("📈 Model Comparison")
    if not models:
        st.warning("Add models in the shared library first."); return
    selected=st.multiselect("Models", list(models.keys()), default=list(models.keys())[:min(5,len(models))])

    # User-controlled comparison settings.
    # These values are used only for plotting and for the PDF charts;
    # the Summary tab remains fully symbolic.
    st.markdown("#### Numerical values used in comparison plots")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.15])
    with c1:
        av = st.number_input("A", value=0.50, step=0.05, format="%.3f", key="cmp_A_v18")
    with c2:
        m1v = st.number_input("μ₁", value=0.30, step=0.05, format="%.3f", key="cmp_mu1_v18")
    with c3:
        m2v = st.number_input("μ₂", value=0.60, step=0.05, format="%.3f", key="cmp_mu2_v18")
    with c4:
        force01 = st.checkbox("Force xᵢ from 0 to 1", value=False, key="cmp_force01_v18")

    inv_m1v, inv_m2v = m2v, m1v

    if not selected:
        st.info("Select at least one model."); return

    tabs=st.tabs(["mu_eff plot", "Fd plot", "mu_eff inverse", "Fd inverse", "Summary", "PDF report"])
    fig_mu=go.Figure(); fig_fd=go.Figure(); fig_mu_inv=go.Figure(); fig_fd_inv=go.Figure(); ok_selected=[]
    for name in selected:
        m=models[name]
        try:
            x=x_grid(m,av, force_0_1=force01)
            fig_mu.add_trace(go.Scatter(x=x, y=mu_curve(m,x,av,m1v,m2v), mode='lines', name=name))
            fig_fd.add_trace(go.Scatter(x=x, y=fd_curve(m,x,av,m1v,m2v), mode='lines', name=name))
            fig_mu_inv.add_trace(go.Scatter(x=x, y=mu_curve(m,x,av,inv_m1v,inv_m2v), mode='lines', name=name))
            fig_fd_inv.add_trace(go.Scatter(x=x, y=fd_curve(m,x,av,inv_m1v,inv_m2v), mode='lines', name=name))
            ok_selected.append(name)
        except Exception as e:
            st.error(f"Error in model {name}: {e}")
    with tabs[0]:
        st.latex(r"A=" + f"{av:g}" + r",\quad \mu_1=" + f"{m1v:g}" + r",\quad \mu_2=" + f"{m2v:g}")
        st_plot(plot_layout(fig_mu,"",r"$\mu_{eff}$", show_title=False), key="comparison_mu_default")
        render_equation_block(models, ok_selected, "mu")
    with tabs[1]:
        st.latex(r"A=" + f"{av:g}" + r",\quad \mu_1=" + f"{m1v:g}" + r",\quad \mu_2=" + f"{m2v:g}")
        st_plot(plot_layout(fig_fd,"",r"$F_d/(W_{bp}L)$", show_title=False), key="comparison_fd_default")
        render_equation_block(models, ok_selected, "fd")
    with tabs[2]:
        st.caption("Inverse friction case: μ₁ and μ₂ are swapped from the selected comparison values.")
        st.latex(r"A=" + f"{av:g}" + r",\quad \mu_1=" + f"{inv_m1v:g}" + r",\quad \mu_2=" + f"{inv_m2v:g}")
        st_plot(plot_layout(fig_mu_inv,"",r"$\mu_{eff}$", show_title=False), key="comparison_mu_inverse")
        render_equation_block(models, ok_selected, "mu")
    with tabs[3]:
        st.caption("Inverse friction case: μ₁ and μ₂ are swapped from the selected comparison values.")
        st.latex(r"A=" + f"{av:g}" + r",\quad \mu_1=" + f"{inv_m1v:g}" + r",\quad \mu_2=" + f"{inv_m2v:g}")
        st_plot(plot_layout(fig_fd_inv,"",r"$F_d/(W_{bp}L)$", show_title=False), key="comparison_fd_inverse")
        render_equation_block(models, ok_selected, "fd")
    with tabs[4]:
        for name in ok_selected:
            render_symbolic_model_section(name, models[name])
            st.divider()
    with tabs[5]:
        st.write("The PDF report uses the same content shown in the **Summary** tab and includes the base case and inverse friction plots.")
        pdf=comparison_pdf(models, ok_selected, av, m1v, m2v)
        if pdf:
            st.download_button("Generate/download comparison PDF", pdf, "TD_model_comparison_report.pdf", "application/pdf", type="primary")
        else:
            st.error("ReportLab/Matplotlib is not installed. Run pip install reportlab matplotlib.")




def parse_number_list(text, default=None):
    """Parse comma/semicolon/space separated numbers for sensitivity inputs."""
    if default is None:
        default = []
    try:
        raw = str(text).replace(';', ',').replace('\n', ',').split(',')
        vals = []
        for part in raw:
            part = part.strip()
            if not part:
                continue
            vals.append(float(part))
        return vals or default
    except Exception:
        return default


def parse_mu_pairs(text, default=None):
    """Parse mu1,mu2 pairs from lines such as '0.3,0.6'."""
    if default is None:
        default = [(0.2, 0.4)]
    pairs = []
    try:
        for line in str(text).splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.replace(';', ',').split(',') if p.strip()]
            if len(parts) >= 2:
                pairs.append((float(parts[0]), float(parts[1])))
        return pairs or default
    except Exception:
        return default


def sensitivity_model_summary(model_name, model):
    st.markdown(f"### {model_name}")
    try:
        st.latex(r"\mu_{eff} = " + pretty_expr(model_expr(model)))
        st.latex(r"\frac{F_d}{W_{bp}L} = " + pretty_expr_expanded(fd_display_symbolic(model)))
        st.caption("Force direction: " + fd_direction_label(model))
        st.latex(r"\frac{F_d}{W_{bp}L} = " + fd_integral_symbol_latex(model) + r"\int \mu_{eff}\,dx_i")
        xmin_ltx = pretty_expr(parse_expr_safe(model.get('x_min','0')))
        xmax_ltx = pretty_expr(parse_expr_safe(model.get('x_max','1')))
        st.latex(r"x_i \in \left[" + xmin_ltx + r",\ " + xmax_ltx + r"\right]")
        st.markdown("**Symbolic values at model endpoints**")
        for item in symbolic_reference_items(model):
            endpoint = item.get("endpoint_name", "")
            point = item["point_latex"]
            left = endpoint + r"=" + point
            mu_rhs = r"\text{undefined}" if item["mu"] is None else pretty_expr(item["mu"])
            fd_rhs = r"\text{undefined}" if item["fd"] is None else pretty_expr_expanded(item["fd"])
            st.latex(r"\mu_{eff}\left(" + left + r"\right) = " + mu_rhs)
            st.latex(r"\frac{F_d}{W_{bp}L}\left(" + left + r"\right) = " + fd_rhs)
    except Exception as e:
        st.error(f"Could not render symbolic summary: {e}")


def combo_label(av, m1v, m2v):
    return f"A={av:g}, μ1={m1v:g}, μ2={m2v:g}"


def add_sensitivity_traces(fig, model, output, a_values, mu_pairs):
    for av in a_values:
        for m1v, m2v in mu_pairs:
            x = x_grid(model, av)
            y = selected_curve(model, output, x, av, m1v, m2v)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=combo_label(av, m1v, m2v)))
    return fig


def symbolic_derivative(model, var):
    try:
        return sp.simplify(sp.diff(fd_symbolic(model), var))
    except Exception:
        return None


def add_symbolic_derivative_equation(model, var, label):
    d = symbolic_derivative(model, var)
    if d is not None:
        st.markdown("**Resulting derivative equation**")
        st.latex(label + " = " + pretty_expr_expanded(d))


def fd_derivative_curve(model, x, av, m1v, m2v, var='mu1', h=1e-4):
    y0 = fd_curve(model, x, av, m1v, m2v)
    if var == 'mu1':
        y1 = fd_curve(model, x, av, m1v+h, m2v)
    elif var == 'mu2':
        y1 = fd_curve(model, x, av, m1v, m2v+h)
    else:
        y1 = fd_curve(model, x, av+h, m1v, m2v)
    return (y1-y0)/h




def eval_expr_adaptive(expr, xi_value, a_value, m1_value, m2_value):
    """Evaluate a SymPy expression without assuming it depends on A.

    Some models use A directly in mu_eff/Fd, while others use A only to define
    x_i,min/x_i,max or do not use A at all. This evaluator passes only the
    variables that are actually present in the expression, which avoids the
    operational-variable failure observed for models with no explicit A term.
    """
    expr = sp.simplify(expr)
    available = {xi: xi_value, A: a_value, mu1: m1_value, mu2: m2_value}
    symbols = [sym for sym in (xi, A, mu1, mu2) if sym in expr.free_symbols]
    if not symbols:
        return float(expr)
    args = [available[sym] for sym in symbols]
    f = sp.lambdify(symbols, expr, "numpy")
    val = f(*args)
    arr = np.asarray(val, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    arr[~np.isfinite(arr)] = np.nan
    return arr

# ----------------------------- operational variable helpers -----------------------------
OP_VARS = {
    "H*": {"label": "H* = Hbed/Dh", "base": 0.40, "min": 0.0, "max": 1.0},
    "V*": {"label": "V* = Vpull/Vref", "base": 1.00, "min": 0.0, "max": 2.0},
    "d*": {"label": "d* = dp/Dh", "base": 0.10, "min": 0.0, "max": 0.50},
}
OP_COEFF_KEYS = ["c0", "cH", "cV", "cd", "cHH", "cVV", "cdd", "cHV", "cHd", "cVd"]
# Defaults calibrated so that the base operational scenario gives
# mu1 = 0.2000 and mu2 = 0.4000 at H*=0.40, V*=1.00, d*=0.10.
# The linear coefficients are kept non-zero so that the operational-variable
# sensitivity plots are informative by default.
OP_DEFAULTS = {
    "mu1": {"c0": 0.157, "cH": 0.05, "cV": 0.02, "cd": 0.03, "cHH": 0.0, "cVV": 0.0, "cdd": 0.0, "cHV": 0.0, "cHd": 0.0, "cVd": 0.0},
    "mu2": {"c0": 0.314, "cH": 0.10, "cV": 0.04, "cd": 0.06, "cHH": 0.0, "cVV": 0.0, "cdd": 0.0, "cHV": 0.0, "cHd": 0.0, "cVd": 0.0},
}


def op_mu(coeffs, H, V, d):
    return (
        coeffs.get("c0",0.0)
        + coeffs.get("cH",0.0)*H + coeffs.get("cV",0.0)*V + coeffs.get("cd",0.0)*d
        + coeffs.get("cHH",0.0)*H**2 + coeffs.get("cVV",0.0)*V**2 + coeffs.get("cdd",0.0)*d**2
        + coeffs.get("cHV",0.0)*H*V + coeffs.get("cHd",0.0)*H*d + coeffs.get("cVd",0.0)*V*d
    )


def op_coeff_inputs(prefix, defaults):
    st.markdown(f"### {prefix}")
    coeffs = {}
    cols = st.columns(5)
    for i, key in enumerate(OP_COEFF_KEYS):
        with cols[i % 5]:
            coeffs[key] = st.number_input(f"{prefix}_{key}", value=float(defaults.get(key, 0.0)), step=0.01, format="%.4f", key=f"op_{prefix}_{key}")
    return coeffs


def op_response_value(model, response, av, xi_eval, coeff1, coeff2, H, V, d):
    """Operational response at one point.

    This version does not require the selected model to contain A explicitly.
    A is still used to evaluate x_i limits and is passed to the expression only
    when the expression actually contains A.
    """
    m1v = op_mu(coeff1, H, V, d)
    m2v = op_mu(coeff2, H, V, d)
    try:
        expr = model_expr(model) if response == "mueff" else fd_symbolic(model)
        val = eval_expr_adaptive(expr, float(xi_eval), float(av), m1v, m2v)
        if isinstance(val, np.ndarray):
            val = float(np.asarray(val).ravel()[0])
        if not np.isfinite(val):
            raise ValueError("non-finite operational response")
        return float(val), m1v, m2v
    except Exception:
        # Last-resort fallback using the existing curve machinery.
        xarr = np.array([xi_eval], dtype=float)
        if response == "mueff":
            return float(mu_curve(model, xarr, av, m1v, m2v)[0]), m1v, m2v
        return float(fd_curve(model, xarr, av, m1v, m2v)[0]), m1v, m2v


def op_axis_range(var):
    spec = OP_VARS[var]
    return np.linspace(spec["min"], spec["max"], 50)


def op_variable_plot(model, response, av, xi_eval, coeff1, coeff2, baseH, baseV, based, var):
    xs = op_axis_range(var)
    ys = []
    for val in xs:
        H, V, d = baseH, baseV, based
        if var == "H*": H = val
        elif var == "V*": V = val
        else: d = val
        y, _, _ = op_response_value(model, response, av, xi_eval, coeff1, coeff2, H, V, d)
        ys.append(y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=var, line=dict(width=3)))
    ylab = "μ<sub>eff</sub>" if response == "mueff" else "F<sub>d</sub>/(W<sub>bp</sub>L)"
    response_label = "μ<sub>eff</sub>" if response == "mueff" else "F<sub>d</sub>/(W<sub>bp</sub>L)"
    fig.update_layout(
        title_text=f"{response_label} sensitivity to {OP_VARS[var]['label']}",
        xaxis_title=OP_VARS[var]['label'],
        yaxis_title=ylab,
        template="plotly_white", height=440, width=760,
        margin=dict(l=75,r=25,t=60,b=70), showlegend=False
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)
    return apply_plotly_text_style(fig)


def op_raw_sensitivity(model, response, av, xi_eval, coeff1, coeff2, baseH, baseV, based, var, h=1e-5):
    base, _, _ = op_response_value(model, response, av, xi_eval, coeff1, coeff2, baseH, baseV, based)
    H, V, d = baseH, baseV, based
    if var == "H*": H += h
    elif var == "V*": V += h
    else: d += h
    yp, _, _ = op_response_value(model, response, av, xi_eval, coeff1, coeff2, H, V, d)
    return (yp-base)/h, base


def op_tornado_chart(model, response, av, xi_eval, coeff1, coeff2, baseH, baseV, based, perturb):
    base, _, _ = op_response_value(model, response, av, xi_eval, coeff1, coeff2, baseH, baseV, based)
    rows = []
    for var in ["H*", "V*", "d*"]:
        lo = max(OP_VARS[var]["min"], {"H*": baseH, "V*": baseV, "d*": based}[var]*(1-perturb))
        hi = min(OP_VARS[var]["max"], {"H*": baseH, "V*": baseV, "d*": based}[var]*(1+perturb))
        vals = []
        for val in [lo, hi]:
            H,V,d = baseH, baseV, based
            if var == "H*": H = val
            elif var == "V*": V = val
            else: d = val
            y, _, _ = op_response_value(model, response, av, xi_eval, coeff1, coeff2, H, V, d)
            vals.append(y-base)
        rows.append((var, vals[0], vals[1], max(abs(vals[0]), abs(vals[1]))))
    rows.sort(key=lambda r: r[3], reverse=True)
    fig = go.Figure()
    for var, low, high, _ in rows:
        fig.add_trace(go.Bar(y=[var], x=[low], orientation='h', name=f"{var} low", showlegend=False))
        fig.add_trace(go.Bar(y=[var], x=[high], orientation='h', name=f"{var} high", showlegend=False))
    fig.update_layout(
        title_text=f"Tornado chart for {response}, perturbation ±{perturb:.0%}",
        xaxis_title=f"Change in {response} relative to base case", yaxis_title="",
        barmode="overlay", template="plotly_white", height=420, width=760,
        margin=dict(l=70,r=20,t=55,b=55)
    )
    fig.update_xaxes(zeroline=True, showgrid=True)
    fig.update_yaxes(tickfont=PLOT_TICK_FONT, color="#000000")
    return apply_plotly_text_style(fig), rows, base


def op_response_grid(model, response, av, xi_eval, coeff1, coeff2, H, V, d):
    """Fast vectorized response evaluation for operational response surfaces.

    Works for models with or without explicit A dependence. If A is not present
    in mu_eff/Fd, it is ignored in the expression and only used for the interval
    and selected x_i point.
    """
    m1v = op_mu(coeff1, H, V, d)
    m2v = op_mu(coeff2, H, V, d)
    try:
        expr = model_expr(model) if response == "mueff" else fd_symbolic(model)
        Z = eval_expr_adaptive(expr, xi_eval, av, m1v, m2v)
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 0:
            Z = np.full_like(np.asarray(m1v, dtype=float), float(Z))
        Z[~np.isfinite(Z)] = np.nan
        return Z
    except Exception:
        # Fallback path: still works, but should rarely be used.
        H_arr, V_arr, d_arr = np.asarray(H), np.asarray(V), np.asarray(d)
        Z = np.empty_like(H_arr, dtype=float)
        it = np.nditer(H_arr, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            Z[idx], _, _ = op_response_value(model, response, av, xi_eval, coeff1, coeff2, float(H_arr[idx]), float(V_arr[idx]), float(d_arr[idx]))
            it.iternext()
        return Z


def op_surface_fig(model, response, av, xi_eval, coeff1, coeff2, baseH, baseV, based, pair, view):
    a, b = pair.split("-")
    xs = op_axis_range(a)
    ys = op_axis_range(b)
    X, Y = np.meshgrid(xs, ys)
    H = np.full_like(X, baseH, dtype=float)
    V = np.full_like(X, baseV, dtype=float)
    d = np.full_like(X, based, dtype=float)
    if a == "H*": H = X
    elif a == "V*": V = X
    else: d = X
    if b == "H*": H = Y
    elif b == "V*": V = Y
    else: d = Y
    Z = op_response_grid(model, response, av, xi_eval, coeff1, coeff2, H, V, d)
    response_title = "μeff" if response == "mueff" else "Fd/(WbpL)"
    if view == "3D surface":
        fig = go.Figure(data=[go.Surface(x=xs, y=ys, z=Z, colorbar=dict(title=response_title))])
        fig.update_layout(title_text=f"Response surface of {response_title}: {pair}", height=520, width=700, margin=dict(l=0,r=0,t=50,b=20), scene=dict(xaxis_title=OP_VARS[a]['label'], yaxis_title=OP_VARS[b]['label'], zaxis_title=response_title))
    else:
        fig = go.Figure(data=[go.Contour(x=xs, y=ys, z=Z, colorbar=dict(title=response_title), contours=dict(showlabels=True))])
        fig.update_layout(title_text=f"Response surface of {response_title}: {pair}", xaxis_title=OP_VARS[a]['label'], yaxis_title=OP_VARS[b]['label'], template="plotly_white", height=520, width=700, margin=dict(l=70,r=25,t=55,b=60))
    return apply_plotly_text_style(fig)


def operational_variables_panel(model, a_values):
    st.markdown("## ⚙️ Operational variables")
    st.markdown(
        "This analysis links the operational variables to the friction coefficients "
        "μ1 and μ2, and then evaluates the selected T&D model."
    )

    response_label = st.radio(
        "Response variable",
        ["μeff", "Fd/(WbpL)"],
        horizontal=True,
        index=0,
        key="op_response_v14",
    )
    response = "mueff" if response_label == "μeff" else "FD"

    st.latex(r"\mu_j = c_{0,j}+c_{H,j}H^*+c_{V,j}V^*+c_{d,j}d^*+c_{HH,j}(H^*)^2+c_{VV,j}(V^*)^2+c_{dd,j}(d^*)^2+c_{HV,j}H^*V^*+c_{Hd,j}H^*d^*+c_{Vd,j}V^*d^*")
    st.caption("Set any coefficient to zero to remove that term from the model. Models without explicit A dependence are supported; A is then used only for xᵢ limits/selection when applicable.")

    c1, c2, c3 = st.columns(3)
    with c1:
        baseH = st.slider(OP_VARS["H*"]["label"], 0.0, 1.0, OP_VARS["H*"]["base"], 0.01, key="op_H_v14")
    with c2:
        baseV = st.slider(OP_VARS["V*"]["label"], 0.0, 2.0, OP_VARS["V*"]["base"], 0.01, key="op_V_v14")
    with c3:
        based = st.slider(OP_VARS["d*"]["label"], 0.0, 0.5, OP_VARS["d*"]["base"], 0.01, key="op_d_v14")

    # Versioned keys avoid Streamlit keeping old values from previous packages.
    def op_coeff_inputs_v14(prefix, defaults):
        st.markdown(f"### {prefix}")
        coeffs = {}
        cols = st.columns(5)
        for i, key in enumerate(OP_COEFF_KEYS):
            with cols[i % 5]:
                coeffs[key] = st.number_input(
                    f"{prefix}_{key}",
                    value=float(defaults.get(key, 0.0)),
                    step=0.01,
                    format="%.4f",
                    key=f"op_v14_{prefix}_{key}",
                )
        return coeffs

    coeff1 = op_coeff_inputs_v14("mu1", OP_DEFAULTS["mu1"])
    coeff2 = op_coeff_inputs_v14("mu2", OP_DEFAULTS["mu2"])

    m1_base = op_mu(coeff1, baseH, baseV, based)
    m2_base = op_mu(coeff2, baseH, baseV, based)
    metric_cols = st.columns(2)
    metric_cols[0].metric("Base scenario μ1", f"{m1_base:.4f}")
    metric_cols[1].metric("Base scenario μ2", f"{m2_base:.4f}")

    av = st.selectbox("A for operational sensitivity", a_values, index=0, key="op_A_v14")
    xmin = evaluate_limit(model.get('x_min','0'), av)
    xmax = evaluate_limit(model.get('x_max','1'), av)
    xmin = 0.0005 if xmin is None else max(0.0005, float(xmin))
    xmax = 1.0 if xmax is None else float(xmax)
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    xi_eval = st.slider("xi evaluation point", float(xmin), float(xmax), float((xmin+xmax)/2), 0.001, key="op_xi_v14")

    out_label = "μeff" if response == "mueff" else "Fd/(WbpL)"
    base_response, _, _ = op_response_value(model, response, av, xi_eval, coeff1, coeff2, baseH, baseV, based)

    st.markdown("### Operational scenario used in the analysis")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("A", f"{float(av):.3g}")
    sc2.metric("xi", f"{float(xi_eval):.3f}")
    sc3.metric("Base μ1", f"{m1_base:.4f}")
    sc4.metric("Base μ2", f"{m2_base:.4f}")
    st.metric(f"Base {out_label} at selected xi", f"{base_response:.5f}")

    st.markdown("### One-variable operational sensitivity")
    st.caption("Each plot varies one operational variable while keeping the other two fixed at the base scenario.")
    try:
        colH, colV = st.columns(2)
        with colH:
            st_plot(op_variable_plot(model, response, av, xi_eval, coeff1, coeff2, baseH, baseV, based, "H*"), key="op_v14_plot_H")
        with colV:
            st_plot(op_variable_plot(model, response, av, xi_eval, coeff1, coeff2, baseH, baseV, based, "V*"), key="op_v14_plot_V")
        st_plot(op_variable_plot(model, response, av, xi_eval, coeff1, coeff2, baseH, baseV, based, "d*"), key="op_v14_plot_d")
    except Exception as e:
        st.error(f"Could not generate one-variable operational plots: {e}")

    st.markdown("### Tornado analysis")
    perturb = st.slider("Tornado perturbation", 0.01, 0.50, 0.20, 0.01, key="op_perturb_v14")
    try:
        tornado_fig, tornado_rows, base_val = op_tornado_chart(model, response, av, xi_eval, coeff1, coeff2, baseH, baseV, based, perturb)
        st_plot(tornado_fig, key="op_v14_tornado")
        tornado_df = pd.DataFrame(
            [
                {
                    "Variable": var,
                    "Low change": low,
                    "High change": high,
                    "Max |change|": mag,
                }
                for var, low, high, mag in tornado_rows
            ]
        )
        st.dataframe(tornado_df, width="stretch")
    except Exception as e:
        st.error(f"Could not generate tornado analysis: {e}")

    st.markdown(f"### {out_label} response surfaces")
    st.caption("Surfaces are generated only when requested, to keep the page responsive.")
    generate_surface = st.checkbox("Generate response surface", value=False, key="op_surface_generate_v14")
    if generate_surface:
        pair = st.selectbox("Surface variables", ["H*-V*", "H*-d*", "V*-d*"], index=0, key="op_surface_pair_v14")
        view = st.radio("Surface view", ["2D contour", "3D surface"], horizontal=True, key="op_surface_view_v14")
        try:
            st_plot(op_surface_fig(model, response, av, xi_eval, coeff1, coeff2, baseH, baseV, based, pair, view), key=f"op_v14_surface_{pair}_{view}")
        except Exception as e:
            st.error(f"Could not generate response surface: {e}")
    else:
        st.info("Select 'Generate response surface' when you need the 2D/3D surface. This avoids slow automatic rendering.")

def sensitivity_module(models):
    st.title("🔎 Sensitivity Analysis")
    if not models:
        st.warning("Add models in the shared library first."); return

    model_name = st.selectbox("Model under analysis", list(models.keys()))
    m = models[model_name]

    with st.expander("Model equations, validity interval and symbolic endpoint values", expanded=True):
        sensitivity_model_summary(model_name, m)

    st.markdown("#### Sensitivity settings")
    c1, c2 = st.columns([1, 1])
    a_values_text = c1.text_input(
        "A values",
        value="0.5, 1.0, 1.5",
        help="Use comma-separated values. Example: 0.5, 1.0, 1.5",
        key="sens_A_values_v14",
    )
    mu_pairs_text = c2.text_area(
        "μ1, μ2 pairs",
        value="0.2, 0.4",
        height=80,
        help="One pair per line. Example: 0.2, 0.4",
        key="sens_mu_pairs_v14",
    )
    a_values = parse_number_list(a_values_text, [0.5, 1.0, 1.5])
    a_values = [v for v in a_values if np.isfinite(v) and v > 0]
    if not a_values:
        a_values = [0.5]
    mu_pairs = parse_mu_pairs(mu_pairs_text, [(0.2, 0.4)])

    st.caption("Each analysis defines its own output. Several values of A and several μ1/μ2 pairs can be plotted at the same time.")

    left, right = st.columns([0.28, 0.72], gap="large")
    with left:
        st.markdown("### Navigation")
        nav = st.radio(
            "",
            [
                "📈 mueff curves",
                "📉 FD vs xi",
                "🔍 dFD/dmu1",
                "🔎 dFD/dmu2",
                "📊 Combined sensitivity",
                "⚖️ Sensitivity ratio",
                "⚙️ Operational variables",
            ],
            label_visibility="collapsed",
        )

    with right:
        if nav == "📈 mueff curves":
            st.subheader("mueff curves")
            st.latex(r"\mu_{eff} = " + pretty_expr(model_expr(m)))
            fig = add_sensitivity_traces(go.Figure(), m, "mu_eff", a_values, mu_pairs)
            st_plot(plot_layout(fig, "", r"$\mu_{eff}$", height=520, show_title=False), key=f"sens_mueff_{model_name}")
            st.caption("Curves of μeff versus xᵢ for the selected A values and μ1/μ2 pairs.")

        elif nav == "📉 FD vs xi":
            st.subheader("Fd/(WbpL) vs xᵢ")
            st.latex(r"\frac{F_d}{W_{bp}L} = " + pretty_expr_expanded(fd_display_symbolic(m)))
            fig = add_sensitivity_traces(go.Figure(), m, "Fd/(Wbp.L)", a_values, mu_pairs)
            st_plot(plot_layout(fig, "", r"$F_d/(W_{bp}L)$", height=520, show_title=False), key=f"sens_fd_{model_name}")
            st.caption("Curves of Fd/(WbpL) versus xᵢ for the selected A values and μ1/μ2 pairs.")

        elif nav == "🔍 dFD/dmu1":
            st.subheader("Sensitivity to μ1")
            add_symbolic_derivative_equation(m, mu1, r"\frac{\partial}{\partial \mu_1}\left(\frac{F_d}{W_{bp}L}\right)")
            fig = go.Figure()
            for av in a_values:
                for m1v, m2v in mu_pairs:
                    x = x_grid(m, av)
                    fig.add_trace(go.Scatter(x=x, y=fd_derivative_curve(m, x, av, m1v, m2v, 'mu1'), mode='lines', name=combo_label(av,m1v,m2v)))
            st_plot(plot_layout(fig, "", "d(Fd/(WbpL))/dμ1", height=520, show_title=False), key=f"sens_dfdm1_{model_name}")

        elif nav == "🔎 dFD/dmu2":
            st.subheader("Sensitivity to μ2")
            add_symbolic_derivative_equation(m, mu2, r"\frac{\partial}{\partial \mu_2}\left(\frac{F_d}{W_{bp}L}\right)")
            fig = go.Figure()
            for av in a_values:
                for m1v, m2v in mu_pairs:
                    x = x_grid(m, av)
                    fig.add_trace(go.Scatter(x=x, y=fd_derivative_curve(m, x, av, m1v, m2v, 'mu2'), mode='lines', name=combo_label(av,m1v,m2v)))
            st_plot(plot_layout(fig, "", "d(Fd/(WbpL))/dμ2", height=520, show_title=False), key=f"sens_dfdm2_{model_name}")

        elif nav == "📊 Combined sensitivity":
            st.subheader("Combined sensitivity")
            st.latex(r"\frac{\partial}{\partial \mu_1}\left(\frac{F_d}{W_{bp}L}\right)\quad\text{and}\quad\frac{\partial}{\partial \mu_2}\left(\frac{F_d}{W_{bp}L}\right)")
            fig = go.Figure()
            h = 1e-4
            for av in a_values:
                for m1v, m2v in mu_pairs:
                    x = x_grid(m, av)
                    y0 = fd_curve(m, x, av, m1v, m2v)
                    d1 = (fd_curve(m, x, av, m1v+h, m2v)-y0)/h
                    d2 = (fd_curve(m, x, av, m1v, m2v+h)-y0)/h
                    fig.add_trace(go.Scatter(x=x, y=d1, mode='lines', name=f"dF/dμ1, {combo_label(av,m1v,m2v)}"))
                    fig.add_trace(go.Scatter(x=x, y=d2, mode='lines', line=dict(dash='dash'), name=f"dF/dμ2, {combo_label(av,m1v,m2v)}"))
            st_plot(plot_layout(fig, "", "Sensitivity", height=540, show_title=False), key=f"sens_combined_{model_name}")

        elif nav == "⚖️ Sensitivity ratio":
            st.subheader("Sensitivity ratio")
            st.latex(r"R_{\mu}=\frac{\left|\partial F_D/\partial \mu_2\right|}{\left|\partial F_D/\partial \mu_1\right|}")
            h = 1e-4
            fig = go.Figure()
            for av in a_values:
                for m1v, m2v in mu_pairs:
                    x = x_grid(m, av)
                    y0 = fd_curve(m, x, av, m1v, m2v)
                    d1 = (fd_curve(m, x, av, m1v+h, m2v)-y0)/h
                    d2 = (fd_curve(m, x, av, m1v, m2v+h)-y0)/h
                    ratio = np.abs(d2)/(np.abs(d1)+1e-12)
                    ratio[~np.isfinite(ratio)] = np.nan
                    fig.add_trace(go.Scatter(x=x, y=ratio, mode='lines', name=combo_label(av,m1v,m2v)))
            st_plot(plot_layout(fig, "", "|dFd/dμ2| / |dFd/dμ1|", height=520, show_title=False), key=f"sens_ratio_{model_name}")

        elif nav == "⚙️ Operational variables":
            operational_variables_panel(m, a_values)



def campaign_library_module():
    st.title("🗂️ Campaign Library")
    st.caption("Create and manage experimental campaigns. Experiments must always be linked to exactly one campaign.")
    campaigns = load_json_dict(CAMPAIGN_PATH)

    if campaigns:
        rows = []
        for cid, c in campaigns.items():
            rows.append({
                "Campaign": c.get("name", cid),
                "rho_f (ppg)": c.get("rho_f_ppg"),
                "rho_p (kg/m³)": c.get("rho_p_kgm3"),
                "w_pp (N/m)": c.get("w_pp_Nm"),
                "OD (in)": c.get("OD_in"),
                "ID (in)": c.get("ID_in"),
                "L (ft)": c.get("L_ft"),
                "Date": c.get("date", ""),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No campaign saved yet.")

    st.subheader("Add / edit campaign")
    edit = st.checkbox("Load an existing campaign to edit")
    labels = {cid: campaigns[cid].get("name", cid) for cid in campaigns}
    selected_id = None
    if edit and campaigns:
        selected_label = st.selectbox("Existing campaign", list(labels.values()))
        selected_id = next(cid for cid, lab in labels.items() if lab == selected_label)
    base = campaigns.get(selected_id, {}) if selected_id else {}

    with st.form("campaign_form"):
        name = st.text_input("Name of the campaign", value=base.get("name", "New campaign"))
        c1, c2, c3 = st.columns(3)
        rho_f = c1.number_input("ρf - fluid density (ppg)", value=float(base.get("rho_f_ppg", 8.34)), step=0.1, format="%.4f")
        rho_p = c2.number_input("ρp - pipe density (kg/m³)", value=float(base.get("rho_p_kgm3", 7850.0)), step=10.0, format="%.4f")
        w_pp = c3.number_input("wpp - specific weight of pipe (N/m)", value=float(base.get("w_pp_Nm", 100.0)), step=1.0, format="%.4f")
        c4, c5, c6 = st.columns(3)
        od = c4.number_input("OD - outer diameter of inner pipe (in)", value=float(base.get("OD_in", 1.0)), step=0.01, format="%.4f")
        id_ = c5.number_input("ID - inner diameter of outer pipe (in)", value=float(base.get("ID_in", 2.0)), step=0.01, format="%.4f")
        L = c6.number_input("L - total length (ft)", value=float(base.get("L_ft", 10.0)), step=0.1, format="%.4f")
        submitted = st.form_submit_button("Save campaign", type="primary")
        if submitted:
            try:
                if not name.strip():
                    raise ValueError("Campaign name is required.")
                if duplicate_campaign_name(campaigns, name, selected_id):
                    raise ValueError("A campaign with this name already exists. Use a unique campaign name.")
                if not (0 < od < id_):
                    raise ValueError("Require 0 < OD < ID.")
                if L <= 0:
                    raise ValueError("L must be positive.")
                cid = selected_id or new_id("campaign")
                campaigns[cid] = {
                    "campaign_id": cid,
                    "name": name.strip(),
                    "rho_f_ppg": float(rho_f),
                    "rho_p_kgm3": float(rho_p),
                    "w_pp_Nm": float(w_pp),
                    "OD_in": float(od),
                    "ID_in": float(id_),
                    "L_ft": float(L),
                    "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                }
                save_json_dict(CAMPAIGN_PATH, campaigns)
                st.success("Campaign saved.")
                st.rerun()
            except Exception as e:
                
                msg = str(e)
                if "campaign with this name" in msg:
                    st.error("It is not possible to save campaigns with the same name.")
                    try:
                        st.toast("It is not possible to save campaigns with the same name.", icon="⚠️")
                    except Exception:
                        pass
                else:
                    st.error(f"Could not save campaign: {e}")

    if campaigns:
        st.divider()
        c1, c2 = st.columns(2)
        c1.download_button("Download campaign library (.json)", json.dumps(campaigns, ensure_ascii=False, indent=2), "td_campaign_library.json", "application/json")
        delete_label = c2.selectbox("Delete campaign", list(labels.values()), key="delete_campaign_select")
        delete_id = next(cid for cid, lab in labels.items() if lab == delete_label)
        if c2.button("Delete selected campaign"):
            experiments = load_json_dict(EXPERIMENT_PATH)
            linked = [e for e in experiments.values() if e.get("campaign_id") == delete_id]
            if linked:
                st.error("This campaign has linked experiments. Delete or reassign those experiments first.")
            else:
                campaigns.pop(delete_id, None)
                save_json_dict(CAMPAIGN_PATH, campaigns)
                st.rerun()


def experiment_library_module():
    st.title("🧪 Experiment Library")
    campaigns = load_json_dict(CAMPAIGN_PATH)
    experiments = load_json_dict(EXPERIMENT_PATH)
    if not campaigns:
        st.warning("Create at least one campaign before creating experiments.")
        return

    campaign_labels = {cid: campaigns[cid].get("name", cid) for cid in campaigns}
    if experiments:
        rows = []
        for eid, e in experiments.items():
            c = campaigns.get(e.get("campaign_id"), {})
            rows.append({
                "Experiment": e.get("name", eid),
                "Campaign": c.get("name", e.get("campaign_id")),
                "BH (in)": e.get("BH_in"),
                "bh": e.get("bh"),
                "cs": e.get("cs"),
                "v (ft/s)": e.get("v_ft_s"),
                "A": e.get("A"),
                "w_bp (N/m)": e.get("w_bp_Nm"),
                "Points": len(e.get("experimental_data", []) or []),
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No experiment saved yet.")

    st.subheader("Add / edit experiment")
    edit = st.checkbox("Load an existing experiment to edit")
    exp_labels = {eid: experiments[eid].get("name", eid) for eid in experiments}
    selected_exp_id = None
    if edit and experiments:
        selected_exp_label = st.selectbox("Existing experiment", list(exp_labels.values()))
        selected_exp_id = next(eid for eid, lab in exp_labels.items() if lab == selected_exp_label)
    base = experiments.get(selected_exp_id, {}) if selected_exp_id else {}

    current_cid = base.get("campaign_id") if base else list(campaigns.keys())[0]
    default_campaign_index = list(campaigns.keys()).index(current_cid) if current_cid in campaigns else 0
    selected_campaign_label = st.selectbox("Associated campaign", list(campaign_labels.values()), index=default_campaign_index)
    campaign_id = next(cid for cid, lab in campaign_labels.items() if lab == selected_campaign_label)
    campaign = campaigns[campaign_id]

    with st.form("experiment_form"):
        name = st.text_input("Name of the experiment", value=base.get("name", "New experiment"))
        c1, c2, c3, c4 = st.columns(4)
        BH = c1.number_input("BH - cuttings bed height (in)", value=float(base.get("BH_in", 0.25)), step=0.01, format="%.5f")
        bh = c2.number_input("bh - dimensionless bed height", value=float(base.get("bh", 0.0)), step=0.01, format="%.5f")
        cs = c3.number_input("cs - dimensionless cuttings size", value=float(base.get("cs", 0.0)), step=0.01, format="%.5f")
        vel = c4.number_input("v - displacement velocity (ft/s)", value=float(base.get("v_ft_s", 0.1)), step=0.01, format="%.5f")
        st.markdown("#### Experimental data")
        st.caption("Use x in ft and FD in N. You can type, paste from Excel directly into the table, paste a full table below, or upload CSV/XLSX.")
        base_df = experiment_dataframe_from_record(base)
        if base_df.empty:
            base_df = pd.DataFrame({"x_ft": [0.0, float(campaign.get("L_ft", 10.0))], "FD_N": [0.0, 0.0]})

        uploaded_data = st.file_uploader("Upload experimental data (.xlsx, .xls, or .csv)", type=["xlsx", "xls", "csv"], key=f"exp_upload_{selected_exp_id or 'new'}")
        pasted_data = st.text_area("Or paste data copied from Excel", value="", height=90, placeholder="x_ft\tFD_N\n0\t12.3\n0.5\t18.7", key=f"exp_paste_{selected_exp_id or 'new'}")

        table_df = base_df
        try:
            uploaded_df = dataframe_from_uploaded_table(uploaded_data)
            pasted_df = dataframe_from_pasted_text(pasted_data)
            if uploaded_df is not None and not uploaded_df.empty:
                table_df = uploaded_df
                st.info("Using uploaded experimental data in the table below.")
            elif pasted_df is not None and not pasted_df.empty:
                table_df = pasted_df
                st.info("Using pasted experimental data in the table below.")
        except Exception as e:
            st.warning(f"Could not read uploaded/pasted data: {e}")

        edited_df = st.data_editor(table_df, num_rows="dynamic", width="stretch", key=f"exp_data_editor_{selected_exp_id or 'new'}")
        submitted = st.form_submit_button("Save experiment", type="primary")
        if submitted:
            try:
                if not name.strip():
                    raise ValueError("Experiment name is required.")
                if duplicate_experiment_name(experiments, campaign_id, name, selected_exp_id):
                    raise ValueError("An experiment with this name already exists in this campaign. Use a unique experiment name within the campaign.")
                geo = calc_cuttings_area_A(campaign["ID_in"], campaign["OD_in"], BH)
                wbp = calc_wbp_Nm(campaign["rho_f_ppg"], campaign["rho_p_kgm3"], campaign["w_pp_Nm"])
                clean_df = edited_df.copy()
                clean_df["x_ft"] = pd.to_numeric(clean_df["x_ft"], errors="coerce")
                clean_df["FD_N"] = pd.to_numeric(clean_df["FD_N"], errors="coerce")
                clean_df = clean_df.dropna(subset=["x_ft", "FD_N"])
                eid = selected_exp_id or new_id("experiment")
                experiments[eid] = {
                    "experiment_id": eid,
                    "campaign_id": campaign_id,
                    "name": name.strip(),
                    "BH_in": float(BH),
                    "bh": float(bh),
                    "cs": float(cs),
                    "v_ft_s": float(vel),
                    "A": geo["A"],
                    "Ac_in2": geo["Ac_in2"],
                    "Aw_in2": geo["Aw_in2"],
                    "A_case": geo["case"],
                    "w_bp_Nm": wbp,
                    "experimental_data": clean_df[["x_ft", "FD_N"]].to_dict(orient="records"),
                    "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                }
                save_json_dict(EXPERIMENT_PATH, experiments)
                st.success("Experiment saved with calculated A and w_bp.")
                st.rerun()
            except Exception as e:
                
                msg = str(e)
                if "experiment with this name" in msg:
                    st.error("It is not possible to save experiments with the same name inside the same campaign.")
                    try:
                        st.toast("It is not possible to save experiments with the same name inside the same campaign.", icon="⚠️")
                    except Exception:
                        pass
                else:
                    st.error(f"Could not save experiment: {e}")

    try:
        geo_preview = calc_cuttings_area_A(campaign["ID_in"], campaign["OD_in"], BH)
        wbp_preview = calc_wbp_Nm(campaign["rho_f_ppg"], campaign["rho_p_kgm3"], campaign["w_pp_Nm"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Calculated A", f"{geo_preview['A']:.6g}")
        m2.metric("w_bp (N/m)", f"{wbp_preview:.6g}")
        m3.metric("A case", geo_preview["case"])
    except Exception as e:
        st.warning(f"Preview unavailable: {e}")

    if experiments:
        st.divider()
        c1, c2 = st.columns(2)
        c1.download_button("Download experiment library (.json)", json.dumps(experiments, ensure_ascii=False, indent=2), "td_experiment_library.json", "application/json")
        delete_label = c2.selectbox("Delete experiment", list(exp_labels.values()), key="delete_experiment_select")
        delete_id = next(eid for eid, lab in exp_labels.items() if lab == delete_label)
        if c2.button("Delete selected experiment"):
            experiments.pop(delete_id, None)
            save_json_dict(EXPERIMENT_PATH, experiments)
            st.rerun()


def fitting_analysis_module(models):
    st.title("🧩 Model Fitting Analysis")
    if not models:
        st.warning("Add models in the Shared Model Library first.")
        return
    campaigns = load_json_dict(CAMPAIGN_PATH)
    experiments = load_json_dict(EXPERIMENT_PATH)
    analyses = load_json_dict(ANALYSIS_PATH)
    if not campaigns or not experiments:
        st.warning("Create at least one campaign and one experiment before fitting models.")
        return
    if not SCIPY_OK:
        st.error("scipy is required for fitting. Install dependencies from requirements.txt.")
        return

    exp_labels = {eid: f"{experiments[eid].get('name', eid)} — {campaigns.get(experiments[eid].get('campaign_id'), {}).get('name', 'No campaign')}" for eid in experiments}
    selected_exp_label = st.selectbox("Experiment to analyze", list(exp_labels.values()))
    exp_id = next(eid for eid, lab in exp_labels.items() if lab == selected_exp_label)
    exp = experiments[exp_id]
    camp = campaigns.get(exp.get("campaign_id"), {})
    st.caption(f"Campaign: {camp.get('name', exp.get('campaign_id'))}")

    selected_models = st.multiselect("Models to fit", list(models.keys()), default=list(models.keys())[:min(3, len(models))])
    with st.expander("Prepared experimental data", expanded=False):
        try:
            pdata = prepared_exp_data(camp, exp)
            st.dataframe(pdata, width="stretch", hide_index=True)
        except Exception as e:
            st.error(str(e))
            return

    if st.button("Run fitting and save analysis", type="primary"):
        if len(pdata) < 2:
            st.error("At least two valid experimental points are required to estimate mu1 and mu2.")
            return
        if not selected_models:
            st.error("Select at least one model.")
            return

        sig = analysis_signature(exp, selected_models, models)
        existing_id, existing_analysis = existing_analysis_by_signature(analyses, sig)
        if existing_analysis is not None:
            st.warning("This exact analysis already exists and was not saved again.")
            st.dataframe(pd.DataFrame(existing_analysis.get("results", [])).drop(columns=["model_snapshot"], errors="ignore"), width="stretch")
            return

        analysis_id = new_id("analysis")
        results = []
        for model_name in selected_models:
            try:
                fit = fit_one_model(models[model_name], float(exp["A"]), pdata["x_i"].values, pdata["FD_over_wbpL"].values)
                results.append({"model_name": model_name, "model_snapshot": models[model_name], **fit})
            except Exception as e:
                results.append({"model_name": model_name, "model_snapshot": models.get(model_name, {}), "error": str(e), "success": False})
        analyses[analysis_id] = {
            "analysis_id": analysis_id,
            "analysis_signature": sig,
            "campaign_id": exp.get("campaign_id"),
            "campaign_snapshot": camp,
            "experiment_id": exp_id,
            "experiment_snapshot": exp,
            "fit_method": "bounded linear least squares + least_squares/global refinement",
            "results": results,
            "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
        }
        save_json_dict(ANALYSIS_PATH, analyses)
        st.success("Analysis saved.")
        st.dataframe(pd.DataFrame(results).drop(columns=["model_snapshot"], errors="ignore"), width="stretch")

    st.divider()
    st.subheader("Saved analyses")
    if analyses:
        rows=[]
        analysis_labels = {}
        for aid, a in analyses.items():
            exp_name = a.get("experiment_snapshot", {}).get("name", a.get("experiment_id"))
            camp_name = a.get("campaign_snapshot", {}).get("name", a.get("campaign_id"))
            models_list = ", ".join([r.get("model_name", "") for r in a.get("results", [])])
            label = f"{a.get('date', '')} — {camp_name} — {exp_name} — {models_list}"
            analysis_labels[aid] = label
            rows.append({"Date": a.get("date"), "Campaign": camp_name, "Experiment": exp_name, "Models": models_list})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.markdown("#### Delete saved analysis")
        delete_analysis_label = st.selectbox("Select an analysis to delete", list(analysis_labels.values()), key="delete_analysis_select")
        delete_analysis_id = next(aid for aid, lab in analysis_labels.items() if lab == delete_analysis_label)
        cdel1, cdel2 = st.columns([1, 2])
        confirm_delete = cdel1.checkbox("Confirm deletion", key="confirm_delete_analysis")
        if cdel2.button("Delete selected analysis", type="secondary"):
            if not confirm_delete:
                st.warning("Please check 'Confirm deletion' before deleting the selected analysis.")
            else:
                analyses.pop(delete_analysis_id, None)
                save_json_dict(ANALYSIS_PATH, analyses)
                st.success("Selected analysis deleted.")
                st.rerun()


        st.markdown("#### Export compiled fit results")
        campaign_labels_export = {cid: campaigns[cid].get("name", cid) for cid in campaigns}
        exp_c1, exp_c2 = st.columns(2)
        selected_campaign_export = exp_c1.selectbox(
            "Campaign for compiled export",
            list(campaign_labels_export.values()),
            key="compiled_export_campaign",
        )
        export_campaign_id = next(cid for cid, lab in campaign_labels_export.items() if lab == selected_campaign_export)
        selected_model_export = exp_c2.selectbox(
            "Model for compiled export",
            list(models.keys()),
            key="compiled_export_model",
        )
        compiled_df = compiled_fit_results_for_campaign_model(
            analyses, campaigns, experiments, export_campaign_id, selected_model_export
        )
        if compiled_df.empty:
            st.info("No saved fit results found for this campaign/model combination.")
        else:
            st.dataframe(compiled_df, width="stretch", hide_index=True)
            safe_campaign_export = selected_campaign_export.replace(" ", "_")
            safe_model_export = selected_model_export.replace(" ", "_")
            ex1, ex2 = st.columns(2)
            ex1.download_button(
                "Export compiled table as Excel (.xlsx)",
                export_df_xlsx_bytes(compiled_df, "CompiledResults"),
                f"{safe_campaign_export}_{safe_model_export}_compiled_fit_results.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            ex2.download_button(
                "Export compiled table as TXT",
                export_df_txt_bytes(compiled_df),
                f"{safe_campaign_export}_{safe_model_export}_compiled_fit_results.txt",
                "text/plain",
            )

        st.download_button("Download analysis library (.json)", json.dumps(analyses, ensure_ascii=False, indent=2), "td_analysis_library.json", "application/json")
    else:
        st.info("No saved fitting analysis yet.")


def compiled_fit_results_for_campaign_model(analyses, campaigns, experiments, campaign_id, model_name):
    """Compile all saved fit results for one campaign and one model."""
    rows = []
    campaign_name = campaigns.get(campaign_id, {}).get("name", campaign_id)
    for aid, a in sorted(analyses.items(), key=lambda kv: str(kv[1].get("date", ""))):
        if a.get("campaign_id") != campaign_id:
            continue
        exp_id = a.get("experiment_id")
        exp = experiments.get(exp_id, a.get("experiment_snapshot", {})) or {}
        for r in a.get("results", []) or []:
            if r.get("model_name") != model_name:
                continue
            if r.get("error"):
                rows.append({
                    "Campaign": campaign_name,
                    "Experiment": exp.get("name", exp_id),
                    "bh": exp.get("bh"),
                    "cs": exp.get("cs"),
                    "v": exp.get("v_ft_s"),
                    "mu1": None,
                    "mu2": None,
                    "R2": None,
                    "RMSE": None,
                    "Success": False,
                    "Error": r.get("error"),
                    "Analysis date": a.get("date", ""),
                })
            else:
                rows.append({
                    "Campaign": campaign_name,
                    "Experiment": exp.get("name", exp_id),
                    "bh": exp.get("bh"),
                    "cs": exp.get("cs"),
                    "v": exp.get("v_ft_s"),
                    "mu1": r.get("mu1"),
                    "mu2": r.get("mu2"),
                    "R2": r.get("R2"),
                    "RMSE": r.get("RMSE"),
                    "Success": r.get("success", True),
                    "Error": "",
                    "Analysis date": a.get("date", ""),
                })
    return pd.DataFrame(rows)


def render_fd_equation_and_validity_section(model_items, A_val=None):
    """Render Fd/(WbpL) equations and model validity intervals.

    model_items must be a list of (model_name, model_dict). The displayed
    validity interval is the symbolic interval stored in the model library.
    When A_val is provided, the corresponding numerical interval is also shown.
    """
    if not model_items:
        return
    st.subheader("Model equations and validity intervals")
    for model_name, model in model_items:
        st.markdown(f"**{model_name}**")
        try:
            st.latex(r"\frac{F_d}{W_{bp}L} = " + pretty_expr_expanded(fd_display_symbolic(model)))
        except Exception as e:
            st.warning(f"Could not render Fd/(WbpL) equation for {model_name}: {e}")
        try:
            xmin_ltx = pretty_expr(parse_expr_safe(model.get('x_min','0')))
            xmax_ltx = pretty_expr(parse_expr_safe(model.get('x_max','1')))
            st.latex(r"x_i \in \left[" + xmin_ltx + r",\ " + xmax_ltx + r"\right]")
        except Exception:
            st.write(f"Validity interval: {model.get('x_min','')} <= x_i <= {model.get('x_max','')}")
        if A_val is not None:
            try:
                xmin_num, xmax_num = valid_x_range_for_model(model, float(A_val))
                st.caption(f"Numerical interval for A = {float(A_val):.6g}: {xmin_num:.6g} <= x_i <= {xmax_num:.6g}")
            except Exception:
                pass


def find_latest_fit_for_exp_model(analyses, experiment_id, model_name):
    candidates=[]
    for aid,a in analyses.items():
        if a.get("experiment_id") != experiment_id:
            continue
        for r in a.get("results", []) or []:
            if r.get("model_name") == model_name and "error" not in r:
                candidates.append((a.get("date", ""), aid, a, r))
    if not candidates:
        return None
    return sorted(candidates, key=lambda t: t[0])[-1]


def analyze_model_fit_by_campaign_module(models):
    st.title("📌 Model Fit by Campaign")
    campaigns = load_json_dict(CAMPAIGN_PATH)
    experiments = load_json_dict(EXPERIMENT_PATH)
    analyses = load_json_dict(ANALYSIS_PATH)
    if not campaigns or not experiments or not analyses:
        st.warning("You need saved campaigns, experiments, and fitting analyses first.")
        return
    campaign_labels = {cid: campaigns[cid].get("name", cid) for cid in campaigns}
    selected_campaign_label = st.selectbox("Campaign", list(campaign_labels.values()))
    campaign_id = next(cid for cid, lab in campaign_labels.items() if lab == selected_campaign_label)
    model_name = st.selectbox("Model", list(models.keys()))

    rows=[]
    plot_options=[]
    for eid,e in experiments.items():
        if e.get("campaign_id") != campaign_id:
            continue
        fit_item = find_latest_fit_for_exp_model(analyses, eid, model_name)
        if fit_item:
            _, aid, analysis, r = fit_item
            rows.append({
                "Experiment": e.get("name", eid),
                "bh": e.get("bh"),
                "cs": e.get("cs"),
                "v": e.get("v_ft_s"),
                "mu1": r.get("mu1"),
                "mu2": r.get("mu2"),
                "R2": r.get("R2"),
                "RMSE": r.get("RMSE"),
                "Analysis date": analysis.get("date", ""),
            })
            plot_options.append((eid, e.get("name", eid), analysis, r))

    if not rows:
        st.info("No fitted results found for this campaign/model combination.")
        return

    st.subheader("Experimental data and fitted model")
    exp_name = st.selectbox("Experiment to plot", [p[1] for p in plot_options])
    eid, _, analysis, r = next(p for p in plot_options if p[1] == exp_name)
    exp = experiments[eid]
    campaign = campaigns[campaign_id]
    pdata = prepared_exp_data(campaign, exp)
    model_for_plot = r.get("model_snapshot", models[model_name])
    xmin, xmax = valid_x_range_for_model(model_for_plot, float(exp["A"]))
    x_line = np.linspace(xmin, xmax, 250)
    y_line = fd_curve(model_for_plot, x_line, float(exp["A"]), float(r["mu1"]), float(r["mu2"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pdata["x_i"], y=pdata["FD_over_wbpL"], mode="markers", name="Experimental data"))
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name=f"{model_name} fitted"))
    st_plot(plot_layout(fig, "", "Fd/(WbpL)", show_title=False), key=f"option1_plot_{eid}_{model_name}")
    png = fitted_plot_png_bytes(f"{campaign_labels[campaign_id]} - {model_name} - {exp_name}", pdata["x_i"], pdata["FD_over_wbpL"], [(model_name, x_line, y_line)])
    if png:
        st.download_button("Export graph as PNG", png, f"{campaign_labels[campaign_id].replace(' ', '_')}_{model_name.replace(' ', '_')}_{exp_name.replace(' ', '_')}.png", "image/png")

    st.subheader("Fitted parameters and statistics")
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    safe_campaign = campaign_labels[campaign_id].replace(" ", "_")
    safe_model = model_name.replace(" ", "_")
    c1, c2 = st.columns(2)
    c1.download_button("Export table as Excel (.xlsx)", export_df_xlsx_bytes(df, "Option1"), f"{safe_campaign}_{safe_model}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c2.download_button("Export table as TXT", export_df_txt_bytes(df), f"{safe_campaign}_{safe_model}.txt", "text/plain")

    render_fd_equation_and_validity_section([(model_name, model_for_plot)], A_val=float(exp["A"]))


def compare_fitted_models_module(models):
    st.title("📊 Compare Fitted Models")
    campaigns = load_json_dict(CAMPAIGN_PATH)
    experiments = load_json_dict(EXPERIMENT_PATH)
    analyses = load_json_dict(ANALYSIS_PATH)
    if not campaigns or not experiments or not analyses:
        st.warning("You need saved campaigns, experiments, and fitting analyses first.")
        return
    campaign_labels = {cid: campaigns[cid].get("name", cid) for cid in campaigns}
    selected_campaign_label = st.selectbox("Campaign", list(campaign_labels.values()))
    campaign_id = next(cid for cid, lab in campaign_labels.items() if lab == selected_campaign_label)
    exp_ids = [eid for eid,e in experiments.items() if e.get("campaign_id") == campaign_id]
    if not exp_ids:
        st.info("No experiments linked to this campaign.")
        return
    exp_labels = {eid: experiments[eid].get("name", eid) for eid in exp_ids}
    selected_exp_label = st.selectbox("Experiment", list(exp_labels.values()))
    exp_id = next(eid for eid, lab in exp_labels.items() if lab == selected_exp_label)
    available = []
    for model_name in models.keys():
        if find_latest_fit_for_exp_model(analyses, exp_id, model_name):
            available.append(model_name)
    selected_models = st.multiselect("Models to compare", available, default=available[:min(3, len(available))])
    if not selected_models:
        st.info("Select at least one fitted model.")
        return

    exp = experiments[exp_id]
    campaign = campaigns[campaign_id]
    pdata = prepared_exp_data(campaign, exp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pdata["x_i"], y=pdata["FD_over_wbpL"], mode="markers", name="Experimental data"))
    curve_items=[]
    rows=[]
    model_equation_items=[]
    for model_name in selected_models:
        _, aid, analysis, r = find_latest_fit_for_exp_model(analyses, exp_id, model_name)
        model_for_plot = r.get("model_snapshot", models[model_name])
        model_equation_items.append((model_name, model_for_plot))
        xmin, xmax = valid_x_range_for_model(model_for_plot, float(exp["A"]))
        x_line = np.linspace(xmin, xmax, 300)
        y_line = fd_curve(model_for_plot, x_line, float(exp["A"]), float(r["mu1"]), float(r["mu2"]))
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name=model_name))
        curve_items.append((model_name, x_line, y_line))
        rows.append({"Model": model_name, "mu1": r.get("mu1"), "mu2": r.get("mu2"), "R2": r.get("R2"), "RMSE": r.get("RMSE"), "Analysis date": analysis.get("date", "")})
    st_plot(plot_layout(fig, "", "Fd/(WbpL)", show_title=False), key=f"option2_plot_{exp_id}")
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    safe_campaign = campaign_labels[campaign_id].replace(" ", "_")
    safe_exp = exp_labels[exp_id].replace(" ", "_")
    png = fitted_plot_png_bytes(f"{campaign_labels[campaign_id]} - {exp_labels[exp_id]}", pdata["x_i"], pdata["FD_over_wbpL"], curve_items)
    c1, c2, c3 = st.columns(3)
    if png:
        c1.download_button("Export graph as PNG", png, f"{safe_campaign}_{safe_exp}_model_comparison.png", "image/png")
    c2.download_button("Export table as Excel (.xlsx)", export_df_xlsx_bytes(df, "Option2"), f"{safe_campaign}_{safe_exp}_model_comparison.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c3.download_button("Export table as TXT", export_df_txt_bytes(df), f"{safe_campaign}_{safe_exp}_model_comparison.txt", "text/plain")

    render_fd_equation_and_validity_section(model_equation_items, A_val=float(exp["A"]))



# ----------------------------- Box-Behnken statistical analysis helpers -----------------------------
def get_latest_fit_rows_for_campaign_model(analyses, experiments, campaign_id, model_name):
    """Return one latest successful fitted result per experiment for one campaign/model."""
    latest = {}
    for aid, a in analyses.items():
        if a.get("campaign_id") != campaign_id:
            continue
        exp_id = a.get("experiment_id")
        exp = experiments.get(exp_id, a.get("experiment_snapshot", {})) or {}
        if not exp or exp.get("campaign_id", campaign_id) != campaign_id:
            continue
        for r in a.get("results", []) or []:
            if r.get("model_name") != model_name or r.get("error"):
                continue
            try:
                mu1_val = float(r.get("mu1"))
                mu2_val = float(r.get("mu2"))
            except Exception:
                continue
            date_key = str(a.get("date", ""))
            item = {
                "analysis_id": aid,
                "experiment_id": exp_id,
                "Experiment": exp.get("name", exp_id),
                "bh": safe_float(exp.get("bh"), np.nan),
                "cs": safe_float(exp.get("cs"), np.nan),
                "v": safe_float(exp.get("v_ft_s"), np.nan),
                "mu1": mu1_val,
                "mu2": mu2_val,
                "Fit_R2": r.get("R2"),
                "Fit_RMSE": r.get("RMSE"),
                "Analysis date": a.get("date", ""),
            }
            if exp_id not in latest or date_key >= str(latest[exp_id].get("Analysis date", "")):
                latest[exp_id] = item
    df = pd.DataFrame(list(latest.values()))
    if not df.empty:
        df = df.dropna(subset=["bh", "cs", "v", "mu1", "mu2"]).reset_index(drop=True)
    return df


def infer_three_levels(values, label, tol=1e-9):
    vals = sorted(pd.Series(values).dropna().astype(float).unique().tolist())
    collapsed = []
    for v in vals:
        if not collapsed or abs(v - collapsed[-1]) > tol:
            collapsed.append(v)
    if len(collapsed) != 3:
        raise ValueError(f"Variable '{label}' must have exactly three levels for a Box-Behnken design. Found: {collapsed}")
    low, center, high = collapsed
    if not (low < center < high):
        raise ValueError(f"Variable '{label}' levels must be ordered as low < center < high.")
    return {"low": low, "center": center, "high": high, "delta_low": center-low, "delta_high": high-center}


def code_value(value, levels, label, tol=1e-7):
    value = float(value)
    candidates = [(levels["low"], -1), (levels["center"], 0), (levels["high"], 1)]
    for actual, coded in candidates:
        if abs(value - actual) <= tol * max(1.0, abs(actual)):
            return coded
    raise ValueError(f"Value {value} for '{label}' does not match its inferred low, center, or high level.")


def actual_to_coded(value, levels):
    value = np.asarray(value, dtype=float)
    center = float(levels["center"])
    high = float(levels["high"])
    low = float(levels["low"])
    # Box-Behnken levels are normally equally spaced. This handles small asymmetry using the average half-range.
    delta = (high - low) / 2.0
    if delta == 0:
        return value * np.nan
    return (value - center) / delta


def validate_box_behnken_design(df):
    """Validate a 3-factor Box-Behnken matrix using bh, cs, and v."""
    if df.empty:
        raise ValueError("No fitted data are available for this campaign/model combination.")
    levels = {
        "bh": infer_three_levels(df["bh"], "bh"),
        "cs": infer_three_levels(df["cs"], "cs"),
        "v": infer_three_levels(df["v"], "v"),
    }
    coded_rows = []
    invalid = []
    for _, row in df.iterrows():
        try:
            xb = code_value(row["bh"], levels["bh"], "bh")
            xc = code_value(row["cs"], levels["cs"], "cs")
            xv = code_value(row["v"], levels["v"], "v")
            triple = (xb, xc, xv)
            if not (sum(abs(x) for x in triple) == 2 or triple == (0, 0, 0)):
                invalid.append({"Experiment": row.get("Experiment"), "coded point": triple})
            coded_rows.append(triple)
        except Exception as e:
            invalid.append({"Experiment": row.get("Experiment"), "coded point": str(e)})
    expected = set()
    for zero_pos in range(3):
        for a in [-1, 1]:
            for b in [-1, 1]:
                point = [None, None, None]
                point[zero_pos] = 0
                remaining = [i for i in range(3) if i != zero_pos]
                point[remaining[0]] = a
                point[remaining[1]] = b
                expected.add(tuple(point))
    present = set(coded_rows)
    missing = sorted(expected - present)
    has_center = (0, 0, 0) in present
    if invalid or missing or not has_center:
        parts = []
        if missing:
            parts.append(f"Missing Box-Behnken points: {missing}")
        if not has_center:
            parts.append("Missing center point: (0, 0, 0)")
        if invalid:
            parts.append(f"Invalid points: {invalid}")
        raise ValueError("The available fitted results do not match a valid 3-factor Box-Behnken matrix. " + " | ".join(parts))
    coded_df = df.copy()
    coded_df["X_bh"] = [r[0] for r in coded_rows]
    coded_df["X_cs"] = [r[1] for r in coded_rows]
    coded_df["X_v"] = [r[2] for r in coded_rows]
    coded_df["coded_point"] = [str(r) for r in coded_rows]
    return levels, coded_df


BBD_TERMS = [
    ("X_bh", lambda d: d["X_bh"].values),
    ("X_cs", lambda d: d["X_cs"].values),
    ("X_v", lambda d: d["X_v"].values),
    ("X_bh:X_cs", lambda d: d["X_bh"].values * d["X_cs"].values),
    ("X_bh:X_v", lambda d: d["X_bh"].values * d["X_v"].values),
    ("X_cs:X_v", lambda d: d["X_cs"].values * d["X_v"].values),
    ("X_bh^2", lambda d: d["X_bh"].values ** 2),
    ("X_cs^2", lambda d: d["X_cs"].values ** 2),
    ("X_v^2", lambda d: d["X_v"].values ** 2),
]


def build_design_matrix(coded_df, term_names):
    cols = [np.ones(len(coded_df), dtype=float)]
    names = ["Intercept"]
    term_map = {name: fn for name, fn in BBD_TERMS}
    for name in term_names:
        cols.append(np.asarray(term_map[name](coded_df), dtype=float))
        names.append(name)
    return np.column_stack(cols), names


def ols_fit_with_stats(coded_df, response_col, term_names):
    from scipy import stats
    y = coded_df[response_col].astype(float).values
    X, names = build_design_matrix(coded_df, term_names)
    n, p = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    resid = y - yhat
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    ssr = max(0.0, sst - sse)
    df_res = int(n - np.linalg.matrix_rank(X))
    df_model = int(np.linalg.matrix_rank(X) - 1)
    mse = sse / df_res if df_res > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    r2 = float(1.0 - sse / sst) if sst > 0 else np.nan
    r2 = min(1.0, max(0.0, r2)) if np.isfinite(r2) else np.nan
    try:
        cov = mse * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        tvals = beta / se
        pvals = 2 * stats.t.sf(np.abs(tvals), df_res) if df_res > 0 else np.full_like(beta, np.nan, dtype=float)
    except Exception:
        se = np.full_like(beta, np.nan, dtype=float)
        tvals = np.full_like(beta, np.nan, dtype=float)
        pvals = np.full_like(beta, np.nan, dtype=float)
    coeff_df = pd.DataFrame({
        "Term": names,
        "Coefficient": beta,
        "Std. Error": se,
        "t-value": tvals,
        "p-level": pvals,
        "Significant (p<0.1)": [True if nm == "Intercept" else (pv < 0.1 if np.isfinite(pv) else False) for nm, pv in zip(names, pvals)],
    })
    ms_model = ssr / df_model if df_model > 0 else np.nan
    f_value = ms_model / mse if df_model > 0 and df_res > 0 and mse > 0 else np.nan
    p_model = stats.f.sf(f_value, df_model, df_res) if np.isfinite(f_value) else np.nan
    anova_df = pd.DataFrame([
        {"Source": "Regression", "SS": ssr, "df": df_model, "MS": ms_model, "F-value": f_value, "p-level": p_model},
        {"Source": "Residual", "SS": sse, "df": df_res, "MS": mse, "F-value": np.nan, "p-level": np.nan},
        {"Source": "Total", "SS": sst, "df": n - 1, "MS": np.nan, "F-value": np.nan, "p-level": np.nan},
    ])
    return {
        "terms": names,
        "term_names_no_intercept": term_names,
        "beta": beta,
        "yhat": yhat,
        "resid": resid,
        "coefficients": coeff_df,
        "anova": anova_df,
        "R2": r2,
        "RMSE": rmse,
        "SSE": sse,
        "df_res": df_res,
    }


def significant_reduced_fit(coded_df, response_col, alpha=0.1):
    full_term_names = [name for name, _ in BBD_TERMS]
    full = ols_fit_with_stats(coded_df, response_col, full_term_names)
    coeff = full["coefficients"]
    selected = []
    for _, row in coeff.iterrows():
        term = row["Term"]
        if term == "Intercept":
            continue
        try:
            if float(row["p-level"]) < alpha:
                selected.append(term)
        except Exception:
            pass
    reduced = ols_fit_with_stats(coded_df, response_col, selected)
    return full, reduced, selected


def term_value_from_arrays(term, X_bh, X_cs, X_v):
    if term == "X_bh": return X_bh
    if term == "X_cs": return X_cs
    if term == "X_v": return X_v
    if term == "X_bh:X_cs": return X_bh * X_cs
    if term == "X_bh:X_v": return X_bh * X_v
    if term == "X_cs:X_v": return X_cs * X_v
    if term == "X_bh^2": return X_bh ** 2
    if term == "X_cs^2": return X_cs ** 2
    if term == "X_v^2": return X_v ** 2
    return 0


def predict_empirical_from_coded(fit, X_bh, X_cs, X_v):
    beta = fit["beta"]
    names = fit["terms"]
    y = np.zeros_like(np.asarray(X_bh, dtype=float), dtype=float) + float(beta[0])
    for coef, term in zip(beta[1:], names[1:]):
        y = y + float(coef) * term_value_from_arrays(term, X_bh, X_cs, X_v)
    return y


def empirical_formula_text(response_name, fit):
    pieces = [f"{fit['beta'][0]:.6g}"]
    for coef, term in zip(fit["beta"][1:], fit["terms"][1:]):
        sign = "+" if coef >= 0 else "-"
        pieces.append(f" {sign} {abs(coef):.6g}*{term}")
    return f"{response_name} = " + "".join(pieces)


def make_surface_figure(response_name, fit, levels, pair):
    pair_labels = {"bh": "bh", "cs": "cs", "v": "v"}
    a, b = pair
    avals = np.linspace(levels[a]["low"], levels[a]["high"], 40)
    bvals = np.linspace(levels[b]["low"], levels[b]["high"], 40)
    AA, BB = np.meshgrid(avals, bvals)
    actual = {
        "bh": np.zeros_like(AA) + levels["bh"]["center"],
        "cs": np.zeros_like(AA) + levels["cs"]["center"],
        "v": np.zeros_like(AA) + levels["v"]["center"],
    }
    actual[a] = AA
    actual[b] = BB
    X_bh = actual_to_coded(actual["bh"], levels["bh"])
    X_cs = actual_to_coded(actual["cs"], levels["cs"])
    X_v = actual_to_coded(actual["v"], levels["v"])
    Z = predict_empirical_from_coded(fit, X_bh, X_cs, X_v)
    fig = go.Figure(data=[go.Surface(x=AA, y=BB, z=Z, colorscale="Viridis", showscale=True)])
    fig.update_layout(
        title=f"{response_name}: {pair_labels[a]} vs {pair_labels[b]} with the third variable at center",
        scene=dict(
            xaxis_title=pair_labels[a],
            yaxis_title=pair_labels[b],
            zaxis_title=response_name,
        ),
        height=620,
        margin=dict(l=0, r=0, b=0, t=50),
    )
    return apply_plotly_text_style(fig)


def box_behnken_statistical_analysis_module(models):
    st.title("📐 Box-Behnken Statistical Analysis")
    campaigns = load_json_dict(CAMPAIGN_PATH)
    experiments = load_json_dict(EXPERIMENT_PATH)
    analyses = load_json_dict(ANALYSIS_PATH)
    if not campaigns or not experiments or not analyses:
        st.warning("You need saved campaigns, experiments, and fitting analyses before running the Box-Behnken statistical analysis.")
        return
    campaign_labels = {cid: campaigns[cid].get("name", cid) for cid in campaigns}
    c1, c2 = st.columns(2)
    selected_campaign_label = c1.selectbox("Campaign", list(campaign_labels.values()), key="bbd_campaign")
    campaign_id = next(cid for cid, lab in campaign_labels.items() if lab == selected_campaign_label)
    model_name = c2.selectbox("Model", list(models.keys()), key="bbd_model")

    fit_df = get_latest_fit_rows_for_campaign_model(analyses, experiments, campaign_id, model_name)
    st.caption("The module uses the latest saved successful fit for each experiment in the selected campaign/model combination.")
    if fit_df.empty:
        st.info("No saved fitted results were found for this campaign/model combination.")
        return

    with st.expander("Input fitted data used for the statistical analysis", expanded=False):
        st.dataframe(fit_df, width="stretch", hide_index=True)

    try:
        levels, coded_df = validate_box_behnken_design(fit_df)
    except Exception as e:
        st.error(str(e))
        st.markdown("The available fitted results must contain the 12 Box-Behnken edge points plus at least one center point for the variables `bh`, `cs`, and `v`.")
        return

    st.success("The fitted data match a valid 3-factor Box-Behnken matrix for bh, cs, and v.")
    level_df = pd.DataFrame([
        {"Variable": k, "-1 level": v["low"], "0 level": v["center"], "+1 level": v["high"]}
        for k, v in levels.items()
    ])
    st.subheader("Inferred factor levels")
    st.dataframe(level_df, width="stretch", hide_index=True)

    coded_preview = coded_df[["Experiment", "bh", "cs", "v", "X_bh", "X_cs", "X_v", "mu1", "mu2"]].copy()
    st.subheader("Box-Behnken coded matrix")
    st.dataframe(coded_preview, width="stretch", hide_index=True)

    for response_col, response_label in [("mu1", "mu1"), ("mu2", "mu2")]:
        st.divider()
        st.header(f"Empirical model for {response_label}")
        full, reduced, selected = significant_reduced_fit(coded_df, response_col, alpha=0.1)
        st.markdown("**Coding used in the empirical model:**")
        st.code(
            "X_bh = (bh - bh_center) / ((bh_high - bh_low)/2)\n"
            "X_cs = (cs - cs_center) / ((cs_high - cs_low)/2)\n"
            "X_v  = (v  - v_center)  / ((v_high  - v_low)/2)",
            language="text",
        )
        st.markdown("**Final empirical model using only significant terms with p-level < 0.1:**")
        st.code(empirical_formula_text(response_label, reduced), language="text")
        m1, m2 = st.columns(2)
        m1.metric("R²", f"{reduced['R2']:.4f}" if np.isfinite(reduced["R2"]) else "n/a")
        m2.metric("RMSE", f"{reduced['RMSE']:.6g}" if np.isfinite(reduced["RMSE"]) else "n/a")

        st.subheader(f"Coefficient significance table for {response_label} - full quadratic model")
        st.dataframe(full["coefficients"], width="stretch", hide_index=True)
        st.subheader(f"Reduced model coefficients for {response_label}")
        st.dataframe(reduced["coefficients"], width="stretch", hide_index=True)
        st.subheader(f"ANOVA table for {response_label} - reduced model")
        st.dataframe(reduced["anova"], width="stretch", hide_index=True)

        export_pack = {
            "Levels": level_df,
            f"{response_label}_Full_coefficients": full["coefficients"],
            f"{response_label}_Reduced_coefficients": reduced["coefficients"],
            f"{response_label}_ANOVA": reduced["anova"],
        }
        # Make a compact combined export table for txt/xlsx by stacking the main tables.
        combined = []
        combined.append(pd.DataFrame({"Section": [f"{response_label} empirical formula"], "Value": [empirical_formula_text(response_label, reduced)]}))
        combined.append(pd.DataFrame({"Section": [f"{response_label} R2"], "Value": [reduced["R2"]]}))
        combined.append(pd.DataFrame({"Section": [f"{response_label} RMSE"], "Value": [reduced["RMSE"]]}))
        combined_df = pd.concat(combined, ignore_index=True)
        safe_campaign = selected_campaign_label.replace(" ", "_")
        safe_model = model_name.replace(" ", "_")
        d1, d2 = st.columns(2)
        d1.download_button(
            f"Export {response_label} summary as Excel (.xlsx)",
            export_df_xlsx_bytes(combined_df, f"{response_label}_Summary"),
            f"{safe_campaign}_{safe_model}_{response_label}_bbd_summary.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        d2.download_button(
            f"Export {response_label} summary as TXT",
            export_df_txt_bytes(combined_df),
            f"{safe_campaign}_{safe_model}_{response_label}_bbd_summary.txt",
            "text/plain",
        )

        st.subheader(f"Response surfaces for {response_label}")
        for pair in [("bh", "cs"), ("bh", "v"), ("cs", "v")]:
            fig = make_surface_figure(response_label, reduced, levels, pair)
            st.plotly_chart(fig, width="stretch")

def main():
    models=load_models()
    campaigns = load_json_dict(CAMPAIGN_PATH)
    experiments = load_json_dict(EXPERIMENT_PATH)
    analyses = load_json_dict(ANALYSIS_PATH)
    st.sidebar.title("T&D Complete App")
    module=st.sidebar.radio(
        "Module",
        [
            "Model Comparison",
            "Sensitivity Analysis",
            "Shared Model Library",
            "Campaign Library",
            "Experiment Library",
            "Model Fitting Analysis",
            "Model Fit by Campaign",
            "Compare Fitted Models",
            "Box-Behnken Statistical Analysis",
        ],
    )
    st.sidebar.metric("Models", len(models))
    st.sidebar.metric("Campaigns", len(campaigns))
    st.sidebar.metric("Experiments", len(experiments))
    st.sidebar.metric("Analyses", len(analyses))
    st.sidebar.caption("Modelos, campanhas, experimentos e análises são salvos em bibliotecas JSON compartilhadas.")
    if module=="Model Comparison": comparison_module(models)
    elif module=="Sensitivity Analysis": sensitivity_module(models)
    elif module=="Shared Model Library": library_module(models)
    elif module=="Campaign Library": campaign_library_module()
    elif module=="Experiment Library": experiment_library_module()
    elif module=="Model Fitting Analysis": fitting_analysis_module(models)
    elif module=="Model Fit by Campaign": analyze_model_fit_by_campaign_module(models)
    elif module=="Compare Fitted Models": compare_fitted_models_module(models)
    elif module=="Box-Behnken Statistical Analysis": box_behnken_statistical_analysis_module(models)

if __name__ == "__main__":
    main()
