"""
StatLab — Convergence des lois & test d'adéquation
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import binom

from stats import (
    compute_binomial_approximation,
    compute_chi2_test,
    plot_binomial_distribution,
    plot_chi2_density,
    prepare_imported_sim_table,
)

st.set_page_config(page_title="StatLab", page_icon="📈", layout="centered")

st.title("StatLab")
st.caption("Convergence des lois & test d'adéquation")

tab_conv, tab_fit = st.tabs(["Binomiale → Normale", "Test du χ²"])

# ========================================================================
# MODULE 1 — De Moivre-Laplace
# ========================================================================
with tab_conv:
    st.header("De Moivre–Laplace")
    st.markdown(
        "Une somme de *n* épreuves de Bernoulli indépendantes, "
        "*X* ~ ℬ(*n*, *p*), s'approche par une loi normale de même "
        "moyenne et variance quand *n* grandit."
    )

    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("n (essais)", min_value=1, max_value=300, value=40)
    with col2:
        p = st.slider("p (succès)", min_value=0.01, max_value=0.99, value=0.50, step=0.01)

    mode = st.selectbox("Mode", ["Distribution exacte", "Simulation Monte-Carlo"])

    nsim = None
    if mode == "Simulation Monte-Carlo":
        nsim = st.number_input(
            "Tirages", min_value=100, max_value=200_000, value=5000, step=100
        )

    st.subheader("Distribution")

    stats = compute_binomial_approximation(n, p)

    if mode == "Distribution exacte":
        bar_values = binom.pmf(stats["k_values"], n, p)
    else:
        tirages = np.random.binomial(n, p, size=int(nsim))
        counts = np.bincount(tirages, minlength=n + 1)
        bar_values = counts / nsim
        st.session_state["last_sim"] = {"n": n, "p": p, "counts": counts, "total": int(nsim)}

    fig = plot_binomial_distribution(
        stats["k_values"], bar_values, stats["x_curve"], stats["y_curve"], stats["bounds"]
    )
    st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Validité", "Satisfaite" if stats["is_valid"] else "Fragile")
    c2.metric("E(X) = np", f"{stats['mean']:.2f}")
    c3.metric("V(X) = npq", f"{stats['variance']:.2f}")

    with st.expander("Comprendre le calcul"):
        st.markdown("La probabilité ponctuelle exacte suit la loi binomiale :")
        st.latex(r"P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}")
        st.markdown(
            "Elle est approchée par la densité normale centrée sur la moyenne "
            "et l'écart-type de la binomiale :"
        )
        st.latex(
            r"P(X = k) \approx \frac{1}{\sigma}\,\varphi\!\left(\frac{k-np}{\sigma}\right),"
            r"\qquad \varphi(x) = \frac{1}{\sqrt{2\pi}}\,e^{-x^2/2}"
        )
        st.markdown(
            "Validité usuelle : *np* > 5, *n*(1-*p*) > 5. En dessous, la courbe "
            "se décale visiblement des barres — teste-le avec *p* proche de 0 "
            "ou 1 et *n* petit."
        )

# ========================================================================
# MODULE 2 — Test d'adéquation du χ²
# ========================================================================
with tab_fit:
    st.header("Test d'adéquation du χ²")
    st.markdown(
        "On teste H₀ : l'échantillon suit la loi théorique spécifiée, en "
        "comparant les effectifs observés *n*ᵢ aux effectifs attendus "
        "*n p*ᵢ sous H₀."
    )
    st.latex(r"\chi^2_{obs} = \sum_{i=1}^{k} \frac{(n_i - np_i)^2}{np_i}")

    col1, col2, col3 = st.columns(3)
    with col1:
        k_classes = st.number_input("Classes k", min_value=2, max_value=20, value=4, step=1)
    with col2:
        r_params = st.number_input("Paramètres estimés r", min_value=0, max_value=5, value=0, step=1)
    with col3:
        alpha = st.selectbox("Seuil α", [0.10, 0.05, 0.01], index=1, format_func=lambda a: f"{int(a*100)}%")

    bcol1, bcol2 = st.columns(2)
    build_clicked = bcol1.button("Construire le tableau", use_container_width=True)
    import_clicked = bcol2.button(
        "Importer la simulation du module 1", use_container_width=True
    )

    def default_table(k: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Classe": [f"Classe {i + 1}" for i in range(k)],
                "Observé ni": [0] * k,
                "p théorique": [round(1 / k, 4)] * k,
            }
        )

    if "fit_table" not in st.session_state or build_clicked:
        st.session_state["fit_table"] = default_table(int(k_classes))

    if import_clicked:
        last_sim = st.session_state.get("last_sim")
        if last_sim is None:
            st.error(
                "Passe d'abord le module 1 en mode simulation Monte-Carlo "
                "pour générer un échantillon."
            )
        else:
            st.session_state["fit_table"] = prepare_imported_sim_table(last_sim)

    st.subheader("Classes")
    edited = st.data_editor(
        st.session_state["fit_table"],
        num_rows="fixed",
        use_container_width=True,
        key="fit_table_editor",
    )
    st.session_state["fit_table"] = edited

    if st.button("Calculer χ²obs"):
        table = st.session_state["fit_table"]
        obs = table["Observé ni"].to_numpy()
        p_th = table["p théorique"].to_numpy(dtype=float)

        error = None
        if len(table) < 2:
            error = "Il faut au moins 2 classes."
        elif np.any(obs < 0) or np.any(obs != obs.astype(int)):
            error = "Chaque effectif observé doit être un entier ≥ 0."
        elif np.any((p_th <= 0) | (p_th >= 1)):
            error = "Chaque probabilité théorique doit être strictement comprise entre 0 et 1."
        else:
            N = obs.sum()
            sum_p = p_th.sum()
            if N <= 0:
                error = "La somme des effectifs observés doit être strictement positive."
            elif abs(sum_p - 1) > 0.02:
                error = f"La somme des probabilités théoriques ({sum_p:.4f}) doit valoir 1 (tolérance 2%)."
            elif (len(table) - 1 - int(r_params)) < 1:
                error = "ddl = k − 1 − r doit être ≥ 1. Réduis r ou ajoute des classes."

        if error:
            st.error(error)
        else:
            res = compute_chi2_test(obs, p_th, int(r_params), alpha)

            result_table = table.copy()
            result_table["npᵢ"] = np.round(res["np_i"], 2)
            st.dataframe(result_table, use_container_width=True, hide_index=True)

            if res["cochran_warning"]:
                st.warning(
                    "Plus de 20% des classes ont npᵢ < 5 (règle de Cochran) — "
                    "regroupe des classes pour fiabiliser le test."
                )

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("χ²obs", f"{res['chi2_obs']:.4f}")
            r2.metric("ddl", res["ddl"])
            r3.metric("χ²(1-α;ddl)", f"{res['crit']:.4f}")
            r4.metric("p-value", f"{res['pval']:.4f}")

            fig = plot_chi2_density(res["chi2_obs"], res["crit"], res["ddl"])
            st.pyplot(fig)

            if res["rejected"]:
                st.error(f"Rejet de H₀ — χ²obs dépasse la valeur critique au seuil α = {int(alpha * 100)}%.")
            else:
                st.success(f"Non-rejet de H₀ — rien n'indique un écart à la loi testée au seuil α = {int(alpha * 100)}%.")

    with st.expander("Comprendre le calcul"):
        st.markdown(
            "Sous H₀, chaque écart normalisé (*n*ᵢ - *n p*ᵢ)/√(*n p*ᵢ) est "
            "approximativement gaussien centré réduit — encore De "
            "Moivre-Laplace, appliqué classe par classe. La somme des carrés "
            "de *k*-1-*r* de ces écarts suit la loi du χ² :"
        )
        st.latex(r"\chi^2 = \sum_{i=1}^{n} X_i^2, \qquad X_i \sim \mathcal{N}(0,1)")
        st.markdown("Sa fonction de répartition s'exprime avec la fonction Gamma incomplète régularisée :")
        st.latex(r"F(x) = P\!\left(\frac{n}{2}, \frac{x}{2}\right), \qquad P(a,x) = \frac{\int_0^x t^{a-1}e^{-t}\,dt}{\Gamma(a)}")
