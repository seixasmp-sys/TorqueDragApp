import streamlit as st
import pandas as pd
from models import (
    xi_min,
    candidate_mu,
    MODEL_NAMES,
    model_latex,
    derivative_latex,
    symbolic_endpoint_expressions,
)
from plots import (
    plot_muT,
    plot_FD,
    plot_dfd_mu1,
    plot_dfd_mu2,
    plot_sensitivity,
    plot_sensitivity_ratio,
    plot_physical_variable,
    plot_tornado,
    plot_muT_surface,
    plot_muT_surface_3d,
    plot_normalized_muT_sensitivity,
    normalized_muT_sensitivities,
)

st.set_page_config(page_title="Torque and Drag Sensitivity App", layout="wide")

st.markdown("""
<style>
/* Sidebar navigation buttons */
div[data-testid="stRadio"] label {
    font-size: 18px !important;
    font-weight: 700 !important;
    padding: 8px 10px !important;
    border-radius: 10px !important;
}

div[data-testid="stRadio"] label:hover {
    background-color: #eef5ff !important;
}

div[data-testid="stSidebar"] h1,
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3 {
    font-weight: 800 !important;
}

/* General section titles */
h1, h2, h3 {
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Torque and Drag Sensitivity App")
st.caption("Pre-experimental sensitivity analysis for horizontal pull-out drag models.")

with st.sidebar:
    st.header("Model setup")
    model_name = st.selectbox("Select model", MODEL_NAMES)
    st.subheader("A values")
    A_text = st.text_input("A values", "0.5, 1.0, 1.5")
    st.subheader("Friction pairs")
    st.caption("Enter one pair per line as: mu1, mu2")
    mu_pairs_text = st.text_area("mu1, mu2 pairs", "0.47, 0.53", height=120)
    n_points = st.slider("Number of points per curve", 100, 2000, 500, 100)


def parse_list(text):
    return [float(p.strip()) for p in text.replace(";", ",").split(",") if p.strip()]


def parse_pairs(text):
    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid pair: {line}")
        pairs.append((float(parts[0]), float(parts[1])))
    return pairs


try:
    A_values = parse_list(A_text)
    mu_pairs = parse_pairs(mu_pairs_text)
except Exception as e:
    st.error(f"Input error: {e}")
    st.stop()

st.subheader("Selected model")
mu_latex, fd_latex = model_latex(model_name)
st.latex(mu_latex)
st.latex(fd_latex)

st.markdown("## Validity interval and endpoint values")
st.latex(r"\frac{A}{1+A}\le x_i\le 1")

st.markdown("### Numerical interval limits for selected A values")
interval_only = pd.DataFrame({
    "A": A_values,
    "xi_min = A/(1+A)": [xi_min(A) for A in A_values],
    "xi_max": [1.0 for _ in A_values],
})
st.dataframe(interval_only, use_container_width=True)

st.markdown("### Symbolic endpoint equations")
expr = symbolic_endpoint_expressions(model_name)

st.markdown("**Lower limit:**")
st.markdown("$$x_{i,min}=" + expr["xi_min"] + "$$")
st.markdown("**Effective friction at lower limit:**")
st.markdown("$$\\mu_T(x_{i,min})=" + expr["muT_min"] + "$$")
st.markdown("**Normalized drag at lower limit:**")
st.markdown("$$\\frac{F_D}{W_{bp}L}(x_{i,min})=" + expr["fd_min"] + "$$")

st.markdown("**Upper limit:**")
st.markdown("$$x_{i,max}=1$$")
st.markdown("$$\\mu_T(x_{i,max})=" + expr["muT_max"] + "$$")
st.markdown("$$\\frac{F_D}{W_{bp}L}(x_{i,max})=" + expr["fd_max"] + "$$")


st.sidebar.divider()
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "📈 mu_T curves",
        "📉 FD vs xi",
        "🔍 dFD/dmu1",
        "🔍 dFD/dmu2",
        "📊 Combined sensitivity",
        "⚖️ Sensitivity ratio",
        "⚙️ Operational variables",
    ],
    label_visibility="collapsed",
)

if page == "📈 mu_T curves":
    st.header("📈 mu_T curves")
    st.pyplot(plot_muT(model_name, mu_pairs, A_values, n_points))

elif page == "📉 FD vs xi":
    st.header("📉 FD vs xi")
    st.pyplot(plot_FD(model_name, mu_pairs, A_values, n_points))

elif page == "🔍 dFD/dmu1":
    st.header("🔍 dFD/dmu1")
    st.markdown("### Resulting derivative equation")
    der1, der2 = derivative_latex(model_name)
    st.latex(der1)
    st.pyplot(plot_dfd_mu1(model_name, A_values, n_points))

elif page == "🔍 dFD/dmu2":
    st.header("🔍 dFD/dmu2")
    st.markdown("### Resulting derivative equation")
    der1, der2 = derivative_latex(model_name)
    st.latex(der2)
    st.pyplot(plot_dfd_mu2(model_name, A_values, n_points))

elif page == "📊 Combined sensitivity":
    st.header("📊 Combined sensitivity")
    st.pyplot(plot_sensitivity(model_name, A_values, n_points))

elif page == "⚖️ Sensitivity ratio":
    st.header("⚖️ Sensitivity ratio")
    st.latex(r"R_\mu=\frac{|\partial F_D/\partial\mu_2|}{|\partial F_D/\partial\mu_1|}")
    st.pyplot(plot_sensitivity_ratio(model_name, A_values, n_points))

else:
    st.header("⚙️ Operational variables")
    st.markdown("This section evaluates the influence of operational variables on either normalized drag force or directly on the effective friction coefficient.")
    response = st.radio("Response variable", ["muT", "FD"], horizontal=True, index=0)

    st.latex(r"\mu_j=c_{0,j}+c_{H,j}H^*+c_{V,j}V^*+c_{d,j}d^*+c_{HH,j}(H^*)^2+c_{VV,j}(V^*)^2+c_{dd,j}(d^*)^2+c_{HV,j}H^*V^*+c_{Hd,j}H^*d^*+c_{Vd,j}V^*d^*")
    st.caption("Set any coefficient to zero to remove that term from the model.")

    col1, col2, col3 = st.columns(3)
    with col1:
        H = st.slider("Base H* = Hbed/Dh", 0.0, 1.0, 0.4, 0.05)
    with col2:
        V = st.slider("Base V* = Vpull/Vref", 0.0, 2.0, 1.0, 0.05)
    with col3:
        d = st.slider("Base d* = dp/Dh", 0.0, 0.5, 0.1, 0.01)

    def coeff_inputs(prefix, defaults):
        st.subheader(prefix)
        labels = ["c0", "cH", "cV", "cd", "cHH", "cVV", "cdd", "cHV", "cHd", "cVd"]
        cols = st.columns(5)
        vals = {}
        for i, lab in enumerate(labels):
            with cols[i % 5]:
                vals[lab] = st.number_input(f"{prefix}_{lab}", value=float(defaults.get(lab, 0.0)), key=f"{prefix}_{lab}")
        return vals

    defaults1 = {"c0": 0.35, "cH": 0.05, "cV": 0.02, "cd": 0.03, "cHH": 0.0, "cVV": 0.0, "cdd": 0.0, "cHV": 0.0, "cHd": 0.0, "cVd": 0.0}
    defaults2 = {"c0": 0.45, "cH": 0.10, "cV": 0.04, "cd": 0.06, "cHH": 0.0, "cVV": 0.0, "cdd": 0.0, "cHV": 0.0, "cHd": 0.0, "cVd": 0.0}
    coeffs1 = coeff_inputs("mu1", defaults1)
    coeffs2 = coeff_inputs("mu2", defaults2)

    mu1_scenario = candidate_mu(H, V, d, coeffs1)
    mu2_scenario = candidate_mu(H, V, d, coeffs2)
    m1, m2 = st.columns(2)
    m1.metric("Base scenario mu1", f"{mu1_scenario:.4f}")
    m2.metric("Base scenario mu2", f"{mu2_scenario:.4f}")

    A_op = st.selectbox("A for operational sensitivity", A_values, index=0)
    xi_eval = st.slider("xi evaluation point", float(xi_min(A_op)), 1.0, 0.85, 0.01)

    cA, cB = st.columns(2)
    with cA:
        st.pyplot(plot_physical_variable(model_name, "H*", response, A_op, xi_eval, H, V, d, coeffs1, coeffs2))
        st.pyplot(plot_physical_variable(model_name, "d*", response, A_op, xi_eval, H, V, d, coeffs1, coeffs2))
    with cB:
        st.pyplot(plot_physical_variable(model_name, "V*", response, A_op, xi_eval, H, V, d, coeffs1, coeffs2))
        perturb = st.slider("Tornado perturbation", 0.05, 0.50, 0.20, 0.05)
        st.pyplot(plot_tornado(model_name, response, A_op, xi_eval, H, V, d, coeffs1, coeffs2, perturbation=perturb))

    st.subheader("muT-specific sensitivity")
    S, raw, mut_value = normalized_muT_sensitivities(model_name, A_op, xi_eval, H, V, d, coeffs1, coeffs2)
    st.metric("Base muT at selected xi", f"{mut_value:.5f}")
    st.dataframe(pd.DataFrame({
        "Variable": list(S.keys()),
        "Raw sensitivity dmuT/dvariable": [raw[k] for k in S.keys()],
        "Normalized sensitivity": [S[k] for k in S.keys()],
    }), use_container_width=True)
    st.pyplot(plot_normalized_muT_sensitivity(model_name, A_op, xi_eval, H, V, d, coeffs1, coeffs2))

    st.subheader("muT response surfaces")
    pair = st.selectbox("Surface variables", ["H-V", "H-d", "V-d"])
    surface_view = st.radio("Surface view", ["2D contour", "3D surface"], horizontal=True)
    if surface_view == "2D contour":
        st.pyplot(plot_muT_surface(model_name, pair, A_op, xi_eval, H, V, d, coeffs1, coeffs2))
    else:
        st.pyplot(plot_muT_surface_3d(model_name, pair, A_op, xi_eval, H, V, d, coeffs1, coeffs2))
