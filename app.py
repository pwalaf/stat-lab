"""
StatLab — Convergence des lois & test d'adéquation

Module 1 : convergence Binomiale -> Normale
Module 2 : test d'adéquation du χ²

Les calculs probabilistes utilisent scipy.stats (binom, norm, chi2)
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import binom, chi2, norm
import matplotlib.pyplot as plt

st.set_page_config(page_title="StatLab", page_icon="📈", layout="centered")

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": "#888888",
    "axes.grid": False,
})

# ----------------------------------------------------------------------
# En-tête
# ----------------------------------------------------------------------
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

    mean = n * p
    variance = n * p * (1 - p)
    sigma = np.sqrt(variance)
    ok_np = mean > 5
    ok_nq = n * (1 - p) > 5
    valide = ok_np and ok_nq

    k_values = np.arange(0, n + 1)

    if mode == "Distribution exacte":
        bar_values = binom.pmf(k_values, n, p)
    else:
        tirages = np.random.binomial(n, p, size=int(nsim))
        counts = np.bincount(tirages, minlength=n + 1)
        bar_values = counts / nsim
        # Sauvegarde pour le module 2 ("Importer la simulation du module 1")
        st.session_state["last_sim"] = {"n": n, "p": p, "counts": counts, "total": int(nsim)}

    lo = max(0, mean - 4 * sigma)
    hi = min(n, mean + 4 * sigma)
    x_curve = np.linspace(lo, hi, 400)
    y_curve = norm.pdf(x_curve, mean, sigma)

    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.bar(k_values, bar_values, width=0.8, color="#3f6b4a", alpha=0.35, label="Distribution")
    ax.plot(x_curve, y_curve, color="#3f6b4a", linewidth=1.6, label="Approximation normale")
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    st.pyplot(fig)
    plt.close(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Validité", "Satisfaite" if valide else "Fragile")
    c2.metric("E(X) = np", f"{mean:.2f}")
    c3.metric("V(X) = npq", f"{variance:.2f}")

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

    def default_table(k):
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
            n_s, p_s = last_sim["n"], last_sim["p"]
            counts, total = last_sim["counts"], last_sim["total"]
            pmf = binom.pmf(np.arange(0, n_s + 1), n_s, p_s)

            classes = []
            i = 0
            while i <= n_s:
                acc, acc_p, start = counts[i], pmf[i], i
                while acc_p * total < 5 and i < n_s:
                    i += 1
                    acc += counts[i]
                    acc_p += pmf[i]
                label = f"X = {start}" if i == start else f"X ∈ [{start};{i}]"
                classes.append({"label": label, "obs": int(acc), "p": float(acc_p)})
                i += 1

            if len(classes) > 1 and classes[-1]["p"] * total < 5:
                last = classes.pop()
                classes[-1]["obs"] += last["obs"]
                classes[-1]["p"] += last["p"]
                classes[-1]["label"] += " ∪ " + last["label"]

            st.session_state["fit_table"] = pd.DataFrame(
                {
                    "Classe": [c["label"] for c in classes],
                    "Observé ni": [c["obs"] for c in classes],
                    "p théorique": [round(c["p"], 6) for c in classes],
                }
            )

    st.subheader("Classes")
    edited = st.data_editor(
        st.session_state["fit_table"],
        num_rows="fixed",
        use_container_width=True,
        key="fit_table_editor",
    )
    st.session_state["fit_table"] = edited

    compute_clicked = st.button("Calculer χ²obs")

    if compute_clicked:
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
                error = (
                    f"La somme des probabilités théoriques ({sum_p:.4f}) "
                    "doit valoir 1 (tolérance 2%)."
                )
            else:
                ddl = len(table) - 1 - int(r_params)
                if ddl < 1:
                    error = "ddl = k − 1 − r doit être ≥ 1. Réduis r ou ajoute des classes."

        if error:
            st.error(error)
        else:
            norm_factor = 1 / sum_p
            np_i = N * p_th * norm_factor
            chi2_obs = float(np.sum((obs - np_i) ** 2 / np_i))
            np_low = int(np.sum(np_i < 5))

            result_table = table.copy()
            result_table["npᵢ"] = np.round(np_i, 2)
            st.dataframe(result_table, use_container_width=True, hide_index=True)

            if np_low / len(table) > 0.2:
                st.warning(
                    "Plus de 20% des classes ont npᵢ < 5 (règle de Cochran) — "
                    "regroupe des classes pour fiabiliser le test."
                )

            crit = float(chi2.ppf(1 - alpha, ddl))
            pval = float(chi2.sf(chi2_obs, ddl))

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("χ²obs", f"{chi2_obs:.4f}")
            r2.metric("ddl", ddl)
            r3.metric("χ²(1-α;ddl)", f"{crit:.4f}")
            r4.metric("p-value", f"{pval:.4f}")

            x_max = max(chi2_obs, crit) * 1.4 + 1
            x_pts = np.linspace(0.001, x_max, 400)
            y_pts = chi2.pdf(x_pts, ddl)

            fig2, ax2 = plt.subplots(figsize=(7, 2.2))
            ax2.plot(x_pts, y_pts, color="#3f6b4a", linewidth=1.6)
            mask = x_pts >= crit
            ax2.fill_between(x_pts[mask], y_pts[mask], color="#9c3b30", alpha=0.2)
            ax2.axvline(chi2_obs, color="#9c3b30", linestyle="--", linewidth=1.2)
            ax2.set_xlim(0, x_max)
            ax2.set_ylim(bottom=0)
            ax2.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig2)
            plt.close(fig2)

            if chi2_obs > crit:
                st.error(
                    f"Rejet de H₀ — χ²obs dépasse la valeur critique au seuil "
                    f"α = {int(alpha * 100)}%."
                )
            else:
                st.success(
                    f"Non-rejet de H₀ — rien n'indique un écart à la loi "
                    f"testée au seuil α = {int(alpha * 100)}%."
                )

    with st.expander("Comprendre le calcul"):
        st.markdown(
            "Sous H₀, chaque écart normalisé (*n*ᵢ - *n p*ᵢ)/√(*n p*ᵢ) est "
            "approximativement gaussien centré réduit — encore De "
            "Moivre-Laplace, appliqué classe par classe. La somme des carrés "
            "de *k*-1-*r* de ces écarts (indépendants une fois la contrainte "
            "Σ*n*ᵢ=*n* posée, et *r* paramètres estimés retirés) suit la loi "
            "du χ² :"
        )
        st.latex(r"\chi^2 = \sum_{i=1}^{n} X_i^2, \qquad X_i \sim \mathcal{N}(0,1)")
        st.markdown(
            "Sa fonction de répartition s'exprime avec la fonction Gamma "
            "incomplète régularisée :"
        )
        st.latex(
            r"F(x) = P\!\left(\frac{n}{2}, \frac{x}{2}\right), "
            r"\qquad P(a,x) = \frac{\int_0^x t^{a-1}e^{-t}\,dt}{\Gamma(a)}"
        )