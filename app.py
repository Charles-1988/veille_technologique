import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


API_URL = "https://credit-scoring-api-tqja.onrender.com"

st.set_page_config(
    page_title="Dashboard Crédit Scoring",
    layout="wide"
)


@st.cache_data
def load_data():
    # Dataset utilisé pour comparer le client aux dossiers historiques
    return pd.read_csv(
        "https://raw.githubusercontent.com/Charles-1988/credit_scoring_API/refs/heads/main/data/df_100.csv"
    )


df = load_data()


# Variables utilisées par le modèle uniquement
exclude = [
    "SK_ID_CURR",
    "TARGET",
    "decision",
    "classe",
    "class"
]

features = [
    c for c in df.columns
    if c not in exclude
]


# Variables genre

df["Genre"] = df["CODE_GENDER_F"].map(
    {
        0: "Homme",
        1: "Femme"
    }
)

# Conversion jours  années
df["Age"] = -df["DAYS_BIRTH"] / 365.25


display_features = [
    {
        "CODE_GENDER_F": "Genre",
        "DAYS_BIRTH": "Age"
    }.get(c, c)
    for c in features
]


def graph_value(data, var):
    """
    Retourne la valeur affichée dans les graphiques.
    """
    return {
        "Genre": data["Genre"],
        "Age": data["Age"]
    }.get(var, data[var])


def client_value(client, var):
    """
    Retourne la valeur du client sélectionné.
    """
    return {
        "Genre": client["CODE_GENDER_F"],
        "Age": client["Age"]
    }.get(var, client[var])


st.title("Dashboard Crédit Scoring")

st.caption(
"""
Analyse du risque crédit et explication de la décision.
"""
)


client_id = st.sidebar.selectbox(
    "Sélection client",
    df["SK_ID_CURR"].astype(int)
)


# Récupération du dossier client
client = (
    df[df["SK_ID_CURR"] == client_id]
    .iloc[0]
    .to_dict()
)


tab1, tab2 = st.tabs(
    [
        "Analyse client",
        "Explication modèle"
    ]
)


with tab1:

    if st.button("Analyser ce dossier"):

        # Données envoyées à l'API
        payload = {
            k: v
            for k, v in client.items()
            if k in features
        }

        payload["SK_ID_CURR"] = int(client_id)


        st.session_state.pred = requests.post(
            f"{API_URL}/predict",
            json=payload
        ).json()


        st.session_state.shap = requests.post(
            f"{API_URL}/explain",
            json=payload
        ).json()



    if "pred" in st.session_state:

        pred = st.session_state.pred

        risque = pred["proba"] * 100


        # profil clients

        st.subheader("Profil client")


        col1, col2, col3, col4 = st.columns(4)


        with col1:
            st.metric(
                "Age",
                f"{client['Age']:.0f} ans"
            )


        with col2:
            st.metric(
                "Revenu annuel",
                f"{client['AMT_INCOME_TOTAL']:,.0f} €"
            )


        with col3:
            st.metric(
                "Montant crédit",
                f"{client['AMT_CREDIT']:,.0f} €"
            )



        st.divider()


        st.subheader("Score de risque crédit")
        
        
        col1, col2 = st.columns(2)


        with col1:

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risque,
                    title={
                        "text": "Probabilité de défaut (%)"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "steps": [
                            {
                                "range": [0, 10],
                                "color": "lightgreen"
                            },
                            {
                                "range": [10, 100],
                                "color": "salmon"
                            }
                        ]
                    }
                )
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


        with col2:

            if pred["decision"] == "accordé":

                st.success(
                    f"Crédit accepté\n\nRisque : {risque:.1f}%"
                )

            else:

                st.error(
                    f"Crédit refusé\n\nRisque : {risque:.1f}%"
                )


        st.caption(
"""
La probabilité représente le risque estimé par le modèle.

Plus le score est élevé, plus le risque de défaut augmente.
"""
        )


        st.divider()


        # Comparaison avec les dossiers existants

        st.subheader(
            "Positionnement du client"
        )


        variable = st.selectbox(
            "Variable",
            display_features
        )


        plot = df.copy()


        plot["Statut"] = plot["TARGET"].map(
            {
                0: "Acceptés",
                1: "Refusés"
            }
        )


        fig = px.violin(
            plot,
            x=graph_value(plot, variable),
            y="Statut",
            color="Statut",
            box=True,
            points="all",
            color_discrete_map={
                "Acceptés": "green",
                "Refusés": "red"
            }
        )


        # Position du client dans la distribution

        if variable != "Genre":

            fig.add_vline(
                x=client_value(client, variable),
                line_dash="dash",
                line_color="black"
            )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.caption(
"""
Comparaison du client avec les dossiers historiques.

Vert : crédits acceptés.
Rouge : crédits refusés.

La ligne noire représente le client analysé.
"""
        )


        st.divider()


        # Analyse de deux variables simultanément

        st.subheader(
            "Analyse croisée des variables"
        )


        col1, col2 = st.columns(2)


        with col1:

            x_var = st.selectbox(
                "Variable X",
                display_features,
                key="x"
            )


        with col2:

            y_var = st.selectbox(
                "Variable Y",
                display_features,
                key="y"
            )


        fig = px.scatter(
            plot,
            x=graph_value(plot, x_var),
            y=graph_value(plot, y_var),
            color="Statut",
            color_discrete_map={
                "Acceptés": "green",
                "Refusés": "red"
            }
        )


        # Ajout du client étudié

        fig.add_trace(
            go.Scatter(
                x=[
                    client_value(client, x_var)
                ],
                y=[
                    client_value(client, y_var)
                ],
                mode="markers",
                marker={
                    "size": 16,
                    "color": "black",
                    "symbol": "x"
                },
                name="Client"
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        st.caption(
"""
La croix noire représente le client étudié.

Le graphique permet d'observer
sa position par rapport aux autres dossiers.
"""
        )
        
with tab2:

    st.subheader(
        "Explication de la décision du modèle"
    )


    if "shap" in st.session_state:

        shap_values = st.session_state.shap


        # Importance globale du modèle
        global_imp = requests.get(
            f"{API_URL}/global-importance"
        ).json()


        # Noms plus lisibles
        shap_labels = {
            "CODE_GENDER_F": "Genre",
            "CODE_GENDER_M": "Genre",
            "DAYS_BIRTH": "Age"
        }


        # Création du dataframe SHAP

        shap_df = pd.DataFrame(
            {
                "Variable": [
                    shap_labels.get(k, k)
                    for k in global_imp.keys()
                ],

                "Global": list(global_imp.values()),

                "Local": [
                    shap_values.get(k, 0)
                    for k in global_imp.keys()
                ]
            }
        )


        # Regroupement des variables affichées
        shap_df = (
            shap_df
            .groupby("Variable", as_index=False)
            .sum()
        )


        global_df = (
            shap_df
            .sort_values(
                "Global",
                ascending=True
            )
        )


        fig_global = px.bar(
            global_df,
            x="Global",
            y="Variable",
            orientation="h",
            title="Variables les plus utilisées par le modèle"
        )


        fig_global.update_layout(
            height=500,
            xaxis_title="Importance du modèle",
            yaxis_title=""
        )


        st.plotly_chart(
            fig_global,
            use_container_width=True
        )


        st.caption(
"""
L'importance globale indique quelles variables
ont le plus influencé le modèle sur l'ensemble
des clients.
"""
        )


        st.divider()


        st.subheader(
            "Impact des variables pour ce client"
        )


        local_df = (
            shap_df
            .sort_values(
                "Local",
                ascending=True
            )
        )


        # Couleur selon l'effet sur le risque

        local_df["Impact"] = local_df["Local"].apply(
            lambda x:
            "Augmente le risque"
            if x > 0
            else "Diminue le risque"
        )


        fig_local = px.bar(
            local_df,
            x="Local",
            y="Variable",
            orientation="h",
            color="Impact",
            title="Contribution SHAP locale",
            color_discrete_map={
                "Augmente le risque": "red",
                "Diminue le risque": "blue"
            }
        )


        # Même échelle des deux côtés

        max_value = max(
            abs(local_df["Local"].min()),
            abs(local_df["Local"].max())
        )


        fig_local.update_xaxes(
            range=[
                -max_value,
                max_value
            ]
        )


        # Ligne neutre

        fig_local.add_vline(
            x=0,
            line_dash="dash",
            line_color="black"
        )


        fig_local.update_layout(
            height=600,
            xaxis_title="Impact SHAP",
            yaxis_title=""
        )


        st.plotly_chart(
            fig_local,
            use_container_width=True
        )


        st.caption(
"""
Interprétation :

🔴 À droite (valeurs positives)
→ augmente la probabilité de défaut
→ impact défavorable pour le client.

🔵 À gauche (valeurs négatives)
→ diminue la probabilité de défaut
→ impact favorable pour le client.

Plus la barre est éloignée de zéro,
plus la variable influence la décision.
"""
        )


    else:

        st.info(
            "Analysez un client pour afficher l'explication SHAP."
        )
