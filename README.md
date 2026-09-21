# 📈 StatLab — Convergence des lois & Test d'adéquation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stat-lab-m1-gid.streamlit.app/)

**StatLab** est une application web interactive conçue pour illustrer et pratiquer deux concepts clés des statistiques inférentielles : la convergence d'une loi binomiale vers la loi normale (Théorème de De Moivre-Laplace) et le test d'adéquation du $\chi^2$.

🔗 **Application en ligne :** [stat-lab-m1-gid.streamlit.app](https://stat-lab-m1-gid.streamlit.app/)

---

## Fonctionnalités

L'application est divisée en deux modules complémentaires :

### 1. Binomiale → Normale (Théorème de De Moivre-Laplace)
* **Visualisation en temps réel :** Superposition de la loi binomiale $\mathcal{B}(n, p)$ et de son approximation normale $\mathcal{N}(np, \sqrt{np(1-p)})$.
* **Deux modes au choix :**
  * *Distribution exacte* : Calcul des probabilités théoriques.
  * *Simulation Monte-Carlo* : Génération de tirages aléatoires ($100$ à $200\,000$ simulations).
* **Indicateur de validité :** Vérification automatique des conditions usuelles d'approximation ($np > 5$ et $n(1-p) > 5$).
* **Exportation :** Possibilité de transférer les données simulées vers le module du $\chi^2$.

### 2. Test d'adéquation du $\chi^2$
* **Saisie dynamique :** Éditeur de tableau interactif pour renseigner les effectifs observés $n_i$ et les probabilités théoriques $p_i$.
* **Import automatique :** Récupération directe des résultats issus de la simulation Monte-Carlo du Module 1 (avec regroupement automatique des classes selon la règle de Cochran).
* **Analyse complète :**
  * Calcul de la statistique $\chi^2_{\text{obs}}$, des degrés de liberté ($ddl = k - 1 - r$), de la valeur critique et de la $p$-value.
  * Graphique de la densité de la loi du $\chi^2$ avec zone de rejet ($H_0$).
  * Avertissement automatique en cas de non-respect de la règle de Cochran ($n p_i < 5$).

---

## Structure du Projet

Composé de (4 fichiers) :

```text
.
├── app.py           # Interface utilisateur Streamlit
├── stats.py         # Moteur de calcul statistique & figures Matplotlib
├── requirements.txt # Dépendances Python
└── README.md        # Documentation
```

## Lancer en local

```
pip install -r requirements.txt
streamlit run app.py
```