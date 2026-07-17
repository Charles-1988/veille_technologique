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
    # Dataset utilisé pour comparer le client aux historiques
    return pd.read_csv(
        "https://raw.githubusercontent.com/Charles-1988/credit_scoring_API/refs/heads/main/data/df_100.csv"
    )


df = load_data()


# Variables non utilisées dans les analyses
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




# Conversion variables techniques pour affichage utilisateur

df["Genre"] = df["CODE_GENDER_F"].map(
    {
        0:"Homme",
        1:"Femme"
    }
)

df["Age"] = -df["DAYS_BIRTH"] / 365.25



# Variables affichées dans les menus

display_features = [
    {
        "CODE_GENDER_F":"Genre",
        "DAYS_BIRTH":"Age"
    }.get(c,c)

    for c in features
]



# Valeur utilisée dans les graphiques

def graph_value(data,var):

    return {
        "Genre": data["Genre"],
        "Age": data["Age"]
    }.get(var,data[var])



# Valeur du client sélectionné

def client_value(client,var):

    return {
        "Genre": client["CODE_GENDER_F"],
        "Age": client["Age"]
    }.get(var,client[var])




st.title(
    "Dashboard Crédit Scoring"
)


st.caption(
"""
Evaluation du risque crédit,
comparaison du profil client
et explication de la décision du modèle.
"""
)


client_id = st.sidebar.selectbox(
    "Sélection client",
    df["SK_ID_CURR"].astype(int)
)


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


        # Variables envoyées au modèle
        payload = {
            k:v for k,v in client.items()
            if k in features
        }

        payload["SK_ID_CURR"] = int(client_id)


        # Appels API
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



     

        st.subheader(
            "Score de risque crédit"
        )


        col1,col2 = st.columns(2)


        with col1:


            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risque,
                    title={
                        "text":"Probabilité de défaut (%)"
                    },
                    gauge={
                        "axis":{
                            "range":[0,100]
                        },
                        "steps":[
                            {
                                "range":[0,10],
                                "color":"lightgreen"
                            },
                            {
                                "range":[10,100],
                                "color":"salmon"
                            }
                        ]
                    }
                )
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


            st.caption(
"""
La jauge représente la probabilité
estimée de défaut.

Plus le score est élevé,
plus le risque crédit augmente.
"""
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



     

        st.divider()

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
                0:"Acceptés",
                1:"Refusés"
            }
        )


        fig = px.violin(
            plot,
            x=graph_value(plot,variable),
            y="Statut",
            color="Statut",
            box=True,
            points="all",
            color_discrete_map={
                "Acceptés":"green",
                "Refusés":"red"
            }
        )


        # Position du client analysé

        if variable != "Genre":

            fig.add_vline(
                x=client_value(client,variable),
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

Vert : clients acceptés.
Rouge : clients refusés.

La ligne noire indique le client analysé.
"""
        )





        st.divider()

        st.subheader(
            "Analyse croisée des variables"
        )


        c1,c2 = st.columns(2)


        with c1:

            x_var = st.selectbox(
                "Variable X",
                display_features,
                key="x"
            )


        with c2:

            y_var = st.selectbox(
                "Variable Y",
                display_features,
                key="y"
            )



        fig = px.scatter(
            plot,
            x=graph_value(plot,x_var),
            y=graph_value(plot,y_var),
            color="Statut",
            color_discrete_map={
                "Acceptés":"green",
                "Refusés":"red"
            }
        )


        # Client sélectionné

        fig.add_trace(
            go.Scatter(
                x=[
                    client_value(client,x_var)
                ],
                y=[
                    client_value(client,y_var)
                ],
                mode="markers",
                marker={
                    "size":16,
                    "color":"black",
                    "symbol":"x"
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
Cette analyse compare le client
selon deux variables simultanément.

La croix noire représente le client étudié.
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


        # Renommage des variables techniques
        # pour affichage métier

        shap_labels = {
            "CODE_GENDER_F": "Genre",
            "CODE_GENDER_M": "Genre",
            "DAYS_BIRTH": "Age"
        }



        # Création dataframe SHAP

        shap_df = pd.DataFrame(
            {
                "Variable": [
                    shap_labels.get(k, k)
                    for k in global_imp.keys()
                ],

                "Global": [
                    v for v in global_imp.values()
                ],

                "Local": [
                    shap_values.get(k, 0)
                    for k in global_imp.keys()
                ]
            }
        )



        # Fusion des variables représentant
        # la même information métier

        shap_df = (
            shap_df
            .groupby("Variable", as_index=False)
            .agg(
                {
                    "Global":"sum",
                    "Local":"sum"
                }
            )
        )



        # Normalisation importance globale

        shap_df["Global_N"] = (
            shap_df["Global"]
            /
            shap_df["Global"].max()
        )



        # Normalisation SHAP
        # négatif = réduit le risque
        # positif = augmente le risque

        max_shap = shap_df["Local"].abs().max()


        shap_df["Local_N"] = (
            shap_df["Local"] / max_shap
            if max_shap != 0
            else 0
        )



        # Tri par importance

        shap_df = shap_df.sort_values(
            "Global_N"
        )



        fig = go.Figure()



        # Importance globale modèle

        fig.add_trace(
            go.Bar(
                y=shap_df["Variable"],
                x=shap_df["Global_N"],
                orientation="h",
                name="Importance globale",
                marker_color="#34495e"
            )
        )



       

        fig.add_trace(
            go.Scatter(
                y=shap_df["Variable"],
                x=shap_df["Local_N"],
                mode="markers",
                name="Impact client",
                marker=dict(
                    size=13,
                    color="blue"
                )
            )
        )



   

        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="black"
        )



        fig.update_layout(
            height=650,

            xaxis_title=
"""
Impact SHAP :

Gauche : réduction du risque
Droite : augmentation du risque
""",

            legend=dict(
                orientation="h"
            )
        )



        st.plotly_chart(
            fig,
            use_container_width=True
        )



        st.caption(
"""
Les valeurs SHAP expliquent la contribution
des variables dans la décision du modèle.

À gauche :
la variable diminue le risque de défaut.

À droite :
la variable augmente le risque de défaut.

Plus le point est éloigné de zéro,
plus son influence est importante.
"""
        )


    else:

        st.info(
            "Analysez un client pour afficher l'explication SHAP."
        )