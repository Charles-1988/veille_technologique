import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIG ---
API_URL = "https://credit-scoring-api-tqja.onrender.com"
st.set_page_config(page_title="Dashboard d'Aide à la Décision Client", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/Charles-1988/credit_scoring_API/refs/heads/main/data/df_100.csv")

df_100 = load_data()
raw_features = [c for c in df_100.columns if c not in ["SK_ID_CURR", "TARGET", "decision", "classe", "class"]]
display_features = sorted(list(set(["GENRE" if "GENDER" in c else c for c in raw_features])))

def get_real_val(client_data, var_name):
    if var_name == "GENRE": return client_data["CODE_GENDER_F"]
    if var_name == "DAYS_BIRTH": return client_data[var_name] / -365.25
    return client_data[var_name]

def format_var(df, var_name):
    if var_name == "GENRE": return df["CODE_GENDER_F"]
    if var_name == "DAYS_BIRTH": return df[var_name] / -365.25
    return df[var_name]

# --- UI ---
st.title("Interface d'aide à la décision")
tab1, tab2 = st.tabs(["Analyse", "Analyse Globale vs Locale"])

# --- SIDEBAR ---
st.sidebar.header("Sélection Client")
selected_id = st.sidebar.selectbox("ID Client :", df_100["SK_ID_CURR"].astype(int).tolist())
client_data = df_100[df_100["SK_ID_CURR"] == selected_id].iloc[0].to_dict()

# --- TAB 1 ---
with tab1:
    if st.button("Analyser ce dossier"):
        payload = {k: v for k, v in client_data.items() if k in raw_features}
        payload["SK_ID_CURR"] = int(selected_id)
        st.session_state.res_predict = requests.post(f"{API_URL}/predict", json=payload).json()
        st.session_state.res_explain = requests.post(f"{API_URL}/explain", json=payload).json()

    if "res_predict" in st.session_state:
        st.markdown("### Score de Risque Crédit")
        proba = float(st.session_state.res_predict.get("proba", 0)) * 100
        col_jauge, col_txt = st.columns([1, 1])
        with col_jauge:
            fig_j = go.Figure(go.Indicator(mode="gauge+number", value=proba, title={'text': "Risque (%)"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "black"},
                       'steps': [{'range': [0, 9], 'color': "#27ae60"}, {'range': [9, 100], 'color': "#c0392b"}]}))
            st.plotly_chart(fig_j, use_container_width=True)
            st.caption("Jauge de risque : la zone verte représente les dossiers éligibles (risque < 9%) et la zone rouge les dossiers nécessitant une étude approfondie.")
        with col_txt:
            if proba < 9: st.success(f"**Dossier Favorable** : Risque de **{proba:.1f}%**.")
            else: st.error(f"**Dossier Alerte** : Risque de **{proba:.1f}%**.")

    st.divider()
    st.markdown("### Analyse de positionnement client")
    var_v = st.selectbox("Choisir variable :", display_features)
    
    df_plot = df_100.copy()
    df_plot["Groupe"] = df_100["TARGET"].map({0: "Acceptés", 1: "Refusés"})
    fig_v = px.violin(df_plot, x=format_var(df_plot, var_v), y="Groupe", color="Groupe", 
                      box=True, orientation='h', color_discrete_map={"Acceptés": "#27ae60", "Refusés": "#c0392b"})
    fig_v.add_vline(x=get_real_val(client_data, var_v), line_dash="dash", line_width=3, line_color="black")
    st.plotly_chart(fig_v, use_container_width=True)
    st.caption("Ce graphique en violon compare la valeur du client sélectionné (trait pointillé noir) à la distribution des clients acceptés et refusés pour la variable choisie.")

    st.divider()
    st.markdown("### Analyse croisée des variables")
    col1, col2 = st.columns(2)
    with col1: x_var = st.selectbox("Variable X", options=display_features, index=0)
    with col2: y_var = st.selectbox("Variable Y", options=display_features, index=1)
    
    df_bi = df_100.copy()
    fig_bi = px.scatter(df_bi, x=format_var(df_bi, x_var), y=format_var(df_bi, y_var), 
                        color=df_bi["TARGET"].astype(str), color_discrete_map={"0": "#27ae60", "1": "#c0392b"}, opacity=0.7)
    
    fig_bi.add_trace(go.Scatter(x=[get_real_val(client_data, x_var)], y=[get_real_val(client_data, y_var)], 
                                mode='markers', marker=dict(color='black', size=16, symbol='x'), name="Client"))
    st.plotly_chart(fig_bi, use_container_width=True)
    st.caption("Nuage de points situant le client (croix noire) par rapport aux autres clients (points verts pour acceptés, rouges pour refusés) selon deux critères croisés.")

# --- TAB 2 ---
with tab2:
    st.markdown("### Moteur de décision : Local vs Global")
    st.write("Comparez l'importance réelle des variables pour ce client (Bleu) à leur poids habituel dans le modèle (Gris). Plus le point bleu est à droite, plus ce critère a été décisif pour ce dossier.")
    
    if "res_explain" in st.session_state:
        local = {k: float(v) for k, v in st.session_state.res_explain.items()}
        global_data = {k: float(v) for k, v in requests.get(f"{API_URL}/global-importance").json().items()}
        
        df = pd.DataFrame({"Critère": list(global_data.keys()), "Global": list(global_data.values()), "Local": [local.get(k, 0) for k in global_data.keys()]})
        
        gender_cols = [c for c in df["Critère"] if "GENDER" in c]
        if gender_cols:
            g_global = df[df["Critère"].isin(gender_cols)]["Global"].sum()
            g_local = df[df["Critère"].isin(gender_cols)]["Local"].sum()
            df = df[~df["Critère"].isin(gender_cols)]
            df = pd.concat([df, pd.DataFrame({"Critère": ["GENRE"], "Global": [g_global], "Local": [g_local]})], ignore_index=True)

        df["Global_N"] = df["Global"] / df["Global"].max()
        df["Local_N"] = df["Local"].abs() / df["Local"].abs().max()
        df = df.sort_values(by="Global_N", ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(y=df["Critère"], x=df["Global_N"], orientation='h', name="Global (Moyenne)", marker_color="#34495e"))
        fig.add_trace(go.Scatter(y=df["Critère"], x=df["Local_N"], mode='markers', name="Local (Client)", marker=dict(color="#3498db", size=12)))
        fig.update_layout(height=700, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Cliquez sur 'Analyser ce dossier' dans l'onglet 1.")