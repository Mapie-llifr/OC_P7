import streamlit as st
import pandas as pd
#import random
import matplotlib.pyplot as plt
import requests

st.set_page_config(layout="wide")

api_path = "http://127.0.0.1:5000"
apa_path = "https://eu.pythonanywhere.com/user/Mapiellifr/webapps/#id_mapiellifr_eu_pythonanywhere_com"
#DATA_URL = "./Docs_projet7/df_model_final.csv"
DATA_URL = apa_path + "/home/Mapiellifr/OC_P7/small_df_model_final.csv"


@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    data = data.drop('TARGET', axis=1)
    return data


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
    feat_df = pd.Series(feat_dict, index=feat_dict.keys())
    feat_df = feat_df.sort_values(ascending=False)
    fig = plt.figure()
    plt.barh(y=feat_df.index, width=-feat_df.values)
    plt.title("Importance des variables dans l'attribution du score")
    return fig


def predict(client):
    pred_path = api_path + "/predict"
    pred_params = {'id' : client}
    pre_dict = requests.get(pred_path, params=pred_params).json()
    return pre_dict


def scoring(client): 
    score = predict(client)['score']
    if score == 1 : 
        pret = {'body' : 'pret accordé', 
                'divider' : 'green'}
        st.balloons()
    elif score == 5 :
        pret = {'body' : 'pret risqué', 
                'divider' : 'blue'}
    elif score == 0 : 
        pret = {'body' : 'pret refusé', 
                'divider' : 'red'}
        st.snow()
    else : 
        pret = {'body' : 'erreur de calcul', 
                'divider' : 'grey'}
    return pret


def feat_import(client):
    feat_path = api_path + "/feats"
    feat_params = {'id' : client}
    feature_importance = requests.get(feat_path, params=feat_params).json()
    return feature_importance


def fig_var(df, var, client) : 
    idx_client = df[df['SK_ID_CURR'] == client].index[0]
    value = df.at[idx_client, var]
    fig = data_boxplot(df, var, value)
    return fig


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
df = load_data(20000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Loading data... Done!")


st.title(body="Mon dashboard", 
         anchor=False)



col1, col2 =  st.columns(spec=[0.5, 0.5], 
                         gap="small")

with col1 :
    id_client = int(st.number_input(label="ID Client", 
                                    min_value=100002, 
                                    max_value=456255, 
                                    step=1, 
                                    format=None))
    
    if id_client not in df['SK_ID_CURR'].values :
        st.error('Cet id client n\'existe pas', icon="🚨")
    
    st.metric(label="Probabilité de remboursement" , 
              value = (1-predict(id_client)['prediction'])*100, 
              help = "Pourcentage de chance de remboursement de l'emprunt.")
    
    st.subheader(body = scoring(id_client)['body'],
                divider = scoring(id_client)['divider'],
                help = "Interpretation de la probabilité en fonction du risque métier.")

    st.pyplot(
              #fig = feature_bar(feature_importance, id_client), 
              fig = feature_bar(feat_import(id_client), id_client),
              clear_figure=True, 
              use_container_width = True)
    
    st.dataframe(data=df[df['SK_ID_CURR'] == id_client].transpose(), 
                 hide_index = False, 
                 use_container_width = True)

if "colonne" not in st.session_state : 
    st.session_state.colonne = 'SK_ID_CURR'

with col2 :
    
    st.selectbox(label="Sur quelle variable souhaitez vous comparer votre client:", 
                 options = df.columns, 
                 placeholder = "Selectionnez une variable ...",
                 key="colonne"
                 )
    
    if st.session_state.colonne == 'SK_ID_CURR': 
        st.text("Indiquez une variable ...")
    else : 
        st.pyplot(fig = fig_var(df, st.session_state.colonne, id_client), 
                       clear_figure=False, 
                       use_container_width = True)
    
# streamlit run test_streamlit.py
    
    


