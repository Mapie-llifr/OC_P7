import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Affichage utilisant toutes la surface de l'écran
st.set_page_config(layout="wide")

# Liens vers les sources de données externes pour utilisation locale, ou après déploiement
#DATA_URL = "https://eu.pythonanywhere.com/user/Mapiellifr/files/home/Mapiellifr/OC_P7/small_df_model_final.csv"
#DEF_URL = "https://eu.pythonanywhere.com/user/Mapiellifr/files/home/Mapiellifr/OC_P7/HomeCredit_columns_description.csv"
api_path = "https://pretadepenser-06a8123a2ba8.herokuapp.com"  #"http://mapiellifr.eu.pythonanywhere.com"  https://pretadepenser-06a8123a2ba8.herokuapp.com/
#api_path = "http://192.168.1.15:5000"
DATA_URL = "./Docs_projet7/small_df_model_final.csv"  #local
DEF_URL = "./Docs_projet7/HomeCredit_columns_description.csv"   # local
#api_path = "http://127.0.0.1:5000"    #local


# Téléchargement des données de sources externes, et mise en cache
@st.cache_data
def load_data(nrows):
    def_data = pd.read_csv(DEF_URL, encoding_errors='ignore')
    data = pd.read_csv(DATA_URL, nrows=nrows)
    data = data.drop('TARGET', axis=1)
    return data, def_data


def data_boxplot(data, colonne, value=None): 
    """Affiche un boxplot pour une variable quantitative d'un DF.
    
    data : DataFrame
    colonne : string, nom de la colonne correspondant à la variable
    value : float or int : une valeur possible de la variable
    """
    fig = plt.figure()
    plt.boxplot(data[colonne], vert=False, showfliers=False)
    if value :
        plt.scatter(value, 1)
    plt.title(colonne)
    return fig


def feature_bar (feat_dict, client) :
    """
    Retourne une figure montrant l\'importance locale des features dans la prediction. 

    Parameters
    ----------
    feat_dict : dict : dictionnaire des dix meilleurs features (clé) et leurs importances (value).
    client : int : valeur de 'SK_ID_CURR' pour le client.

    Returns
    -------
    fig : pyplot object : bar chart, features en ordonnées, importances en abscisse.
    """
    feat_df = pd.Series(feat_dict, index=feat_dict.keys())
    feat_df = feat_df.sort_values(ascending=False)
    fig = plt.figure()
    plt.barh(y=feat_df.index, width=-feat_df.values)
    plt.title("Importance des variables dans l'attribution du score")
    return fig


def predict(client):
    """
    Envoie une requete pour récupérer la prédiction du modèle.

    Parameters
    ----------
    client : int : valeur de 'SK_ID_CURR' pour le client.

    Returns
    -------
    pre_dict : dict : {'prediction': probabilité de non remboursement, 
                       'score': interprétation de la probabilité, 
                       'id' : valeur de 'SK_ID_CURR' pour le client}.
    """
    pred_path = api_path + "/predict"
    pred_params = {'id' : client}
    pre_dict = requests.get(pred_path, params=pred_params).json()
    return pre_dict


def scoring(client): 
    """
    paramètres d'affichage de l'interprétation de la probabilité de non remboursement.

    Parameters
    ----------
    client : int : valeur de 'SK_ID_CURR' pour le client.

    Returns
    -------
    pret : dict : {'body' : 'PRÊT ACCORDÉ', 'PRÊT RISQUÉ', 'PRÊT REFUSÉ', 'erreur de calcul', 
                   'divider' : 'green', 'blue', 'red', 'grey'}.
    """
    score = predict(client)['score']
    if score == 1 : 
        pret = {'body' : 'PRÊT ACCORDÉ', 
                'divider' : 'green'}
        st.balloons()
    elif score == 5 :
        pret = {'body' : 'PRÊT RISQUÉ', 
                'divider' : 'blue'}
    elif score == 0 : 
        pret = {'body' : 'PRÊT REFUSÉ', 
                'divider' : 'red'}
        st.snow()
    else : 
        pret = {'body' : 'erreur de calcul', 
                'divider' : 'grey'}
    return pret


def feat_import(client):
    """
    Envoie une requete pour récupérer les importances locales des features. 

    Parameters
    ----------
    client : int : valeur de 'SK_ID_CURR' pour le client.

    Returns
    -------
    feature_importance : dict : dictionnaire des dix meilleurs features (clé) et leurs importances (value).
    """
    feat_path = api_path + "/feats"
    feat_params = {'id' : client}
    feature_importance = requests.get(feat_path, params=feat_params).json()
    return feature_importance


def fig_var(df, var, client) : 
    """
    Retourne une figure montrant la distribution d'une variable et le positionnement d'un client.

    Parameters
    ----------
    df : DataFrame : Données de l'ensemble de la clientèle.
    var : str : nom de la variable.
    client : int : valeur de 'SK_ID_CURR' pour le client.

    Returns
    -------
    fig : pyplot object : boxplot, distribution de la variable, avec positionnement du client.
    """
    idx_client = df[df['SK_ID_CURR'] == client].index[0]
    value = df.at[idx_client, var]
    fig = data_boxplot(df, var, value)
    return fig


def make_def(col) : 
    """
    Retourne la définition de la variable, telle que dans le tableau de données.

    Parameters
    ----------
    col : str : nom de la variable.

    Returns
    -------
    str : Phrase permettant l'affichage de la description de la variable.

    """
    idx_col = def_df[def_df.Row == col].index[0]
    description = def_df.iloc[idx_col, 3]
    return f"Définition de la variable séléctionnée : {description}"

# Titres
st.title(body="Prêt à Dépenser", 
         anchor=False)
st.header(body="Demande de crédit", 
          anchor=False)
st.divider()

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
df, def_df = load_data(20000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Loading data... Done!")

#Affichage en deux colonnes
col1, col2 =  st.columns(spec=[0.5, 0.5], 
                         gap="small")

with col1 :
    # Champ de saisie de la valeur 'SK_ID_CURR' du client
    id_client = int(st.number_input(label="ID Client", 
                                    min_value=100002, 
                                    max_value=456255, 
                                    step=1, 
                                    format=None))
    
    if id_client not in df['SK_ID_CURR'].values :
        st.error('Cet id client n\'existe pas', icon="🚨")
    
    # Affichage de la probabilité de remboursement
    st.metric(label="Probabilité de remboursement" , 
              value = (1-predict(id_client)['prediction'])*100, 
              help = "Pourcentage de chance que l'emprunt soit remboursé si le crédit est accepté.")
    
    # Affichage de l'interprétation de la probabilité de non remboursement
    st.header(body = scoring(id_client)['body'],
                divider = scoring(id_client)['divider'],
                anchor=False, 
                help = "Interpretation de la probabilité de non remboursement du prêt, en fonction du risque métier. Si la probabilité de non remboursement atteint un certain seuil la demande de prêt se verra refusée.")
    
    # Affichage de la figure montrant la feature importance locale
    st.subheader(body="Importance des variables dans la décision d'accord ou de refus du prêt.", 
                 anchor=False, 
                 help = "Dix variables ayant le plus de poids dans la prise de décision d'accordé ou de refusé le prêt. Les variables à valeur positive sont en faveur de l'accord, les variables à valeur négative sont en faveur d'un refus du prêt.")
    st.pyplot(fig = feature_bar(feat_import(id_client), id_client),
              clear_figure=True, 
              use_container_width = True)
    
    # Affichage des données d'application du client
    st.subheader(body="Données indiquées par le client lors de la demande d'emprunt.", 
                 anchor=False)
    st.dataframe(data=df[df['SK_ID_CURR'] == id_client].transpose(), 
                 hide_index = False, 
                 use_container_width = True)

if "colonne" not in st.session_state : 
    st.session_state.colonne = 'SK_ID_CURR'

with col2 :
    
    # Champ de saisie du nom de la variable
    st.selectbox(label="Sur quelle variable souhaitez vous comparer votre client:", 
                 options = df.columns, 
                 placeholder = "Selectionnez une variable ...",
                 key="colonne"
                 )
    
    if st.session_state.colonne == 'SK_ID_CURR': 
        st.text("Indiquez une variable ...")
    
    # Affichage de la distribution de la variable et de sa définition
    else : 
        st.subheader(body="Distribution de la variable séléctionnée sur l'ensemble de la clientèle.", 
                     anchor=False, 
                     help = "Positionnement du client (point bleu), par rapport au reste des clients (la valeur médinane est la barre horizontale orange).")
        st.pyplot(fig = fig_var(df, st.session_state.colonne, id_client), 
                       clear_figure=False, 
                       use_container_width = True)
        st.text(body=make_def(st.session_state.colonne))
    
# streamlit run dashboard_streamlit.py
    
    


