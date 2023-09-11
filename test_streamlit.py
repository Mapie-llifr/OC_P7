import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

DATA_URL = "./Docs_projet7/df_explore.csv"

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    data = data.drop('TARGET', axis=1)
    return data

def distrib_score(data, categ, value=None):
    """Affiche un pie plot représentant la distribution d'une variable catégorielle.
    
    data : DataFrame
    categ : string : nom d'une variable catégorielle dans data
    """
    fig = plt.figure()
    nb_per_grade = data[categ].value_counts()
    plt.pie(x=nb_per_grade.values, labels=nb_per_grade.index)
    if value :
        fig.suptitle('Le client est '+value)
    plt.title(f"Distribution de variable quali {categ}")
    plt.legend(loc=(1.3,0.5))  # Place le coin en bas à gauche de la legende
    return fig


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
    feat_df = feat_df.sort_values()
    fig = plt.figure()
    plt.barh(y=feat_df.index, width=feat_df.values)
    plt.title("Importance des variables dans l'attribution du score du client n°" + str(client))
    return fig


id_client = 100002
colonne = 'AMT_INCOME_TOTAL'  #'CODE_GENDER' #

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
df = load_data(30000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache_data)")

prediction = random.random()
score_metier = random.uniform(0,10)

key_feat = [col for col in random.sample(df.columns.to_list(), 10)]
feature_importance = {key : random.random() for key in key_feat}
    
fig_bar = feature_bar(feature_importance, id_client)
fig_bar.show()

def fig_var(df, var, client) : 
    idx_client = df[df['SK_ID_CURR'] == client].index[0]
    value = df.at[idx_client, var]
    if df[var].dtype == 'object' : 
        fig = distrib_score(df, var, value)
    else :
        fig = data_boxplot(df, var, value)
    return fig


st.title(body="Mon dashboard", 
         anchor=False)

st.balloons()

col1, col2 =  st.columns(spec=[0.5, 0.5], 
                         gap="small")

with col1 :
    id_client = int(st.number_input(label="ID Client", 
                                    min_value=100002, 
                                    step=1, 
                                    format=None))
    st.metric(label="Probabilité de remboursement", 
              value = prediction, 
              help = "Pourcentage de chance de remboursement de l'emprunt.")

    st.metric(label="Risque métier", 
              value = score_metier, 
              help = "Note de 0 à 10, indiquant le cout d'une erreur, le moins le mieux.")

    st.pyplot(fig = feature_bar(feature_importance, id_client), 
              clear_figure=True, 
              use_container_width = True)

with col2 :
    st.dataframe(data=df[df['SK_ID_CURR'] == id_client].transpose(), 
                 hide_index = False, 
                 use_container_width = True)


    colonne = st.selectbox(label="Sur quelle variable souhaitez vous comparer votre client:", 
                           options = df.columns, 
                           placeholder = "Selectionnez une variable ...",
                           #on_change=
                           )

    st.pyplot(fig = fig_var(df, colonne, id_client), 
              clear_figure=True, 
              use_container_width = True)


