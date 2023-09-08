import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
# import seaborn as sns
# import matplotlib.pyplot as plt

from datetime import datetime
from modules.model_gensim import ModelGensim
# from modules.model_als import mode_als

# ------------ Load Mode ------------ # 
# --- Content Base --- #
with open('checkpoint/model_gensim.pkl', 'rb') as inp:
    model_gensim = pickle.load(inp)
path_gensim = 'data/ProductRaw.csv'
df_cont = pd.read_csv(path_gensim)
df_gensim = df_cont[['item_id', 'name', 'description', 'group']]
df_gensim = df_gensim.dropna()
df_gensim["name_view_group"] =  df_gensim['name'] + ' ' + df_gensim['description'] + ' ' + df_gensim['group']
# -------------------- #
# --- Collaboration --- #
# dataframe
path_als = 'data/ReviewRaw.csv'
df_als = pd.read_csv(path_als)
# model 
json_file_path = "checkpoint/model_als.json"
with open(json_file_path, 'r') as j:
    dict_als = json.loads(j.read())
# --------------------- #
# ----------------------------------- # 

# ------------ GUI ------------ #
st.title("Data Science Project 2 ")
# menu
menu = ["Business Objective", "Build Project", "Prediction"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""###### Recommender system is a project to help improve the ability to search as well as recommend products for customers. It is implemented based on two algorithms: content-base and collaboration """)  
    st.write("""###### Problem/ Requirement: Use Machine Learning algorithms in Python for project.""")
    st.image("project.png", caption='')

elif choice == 'Build Project':
    st.write("## Recommender system")
    # Upload file gensim
    uploaded_file_gensim = st.file_uploader("Choose a file for content-base", type=['csv'])
    if uploaded_file_gensim is not None:
        # dataframe
        df_gensim = pd.read_csv(uploaded_file_gensim, encoding='latin-1')
        date = int(datetime.now().timestamp())
        df_gensim.to_csv(f"data/product_{date}.csv", index = False)
        df_gensim = df_gensim[['item_id', 'name', 'description', 'group']]
        df_gensim = df_gensim.dropna()
        df_gensim["name_view_group"] =  df_gensim['name'] + ' ' + df_gensim['description'] + ' ' + df_gensim['group']
        # model
        model_gensim = ModelGensim('data/vietnamese-stopwords.txt')
        model_gensim.train(df_gensim["name_view_group"].tolist())
        # save_model
        with open(f'checkpoint/model_gensim_{date}.pkl', 'wb') as fs:
            pickle.dump(model_gensim, fs, pickle.HIGHEST_PROTOCOL)
    # Upload file als
    # uploaded_file_als = st.file_uploader("Choose a file for collaboration", type=['csv'])
    # if uploaded_file_als is not None:
    #     df_als = pd.read_csv(uploaded_file_als, encoding='latin-1')
    #     date = int(datetime.now().timestamp())
    #     df_als.to_csv(f"review{date}.csv", index = False)
    #     rmse = mode_als(df_als)
    #     print("RMSE =",rmse)
        
elif choice == 'Prediction':
    type = st.radio("## Choice Content-base or Collaboration", options=("Content-base", "Collaboration"))
    if type=="Content-base":
        # Upload file
        st.subheader("Product")
        input_text = st.text_input('Text')
        input_number = int( st.slider('Number', 0, 10, 5))
        clicked = st.button("Search")
        if input_text and clicked:
            output = model_gensim.predict(input_text,df_gensim)
            result = output[['item_id', 'name','group','score']].iloc[:input_number]
            st.dataframe(result)

    if type=="Collaboration":        
        input_text = st.text_input('Id')
        input_number = int( st.slider('Number', 0, 10, 5))
        clicked = st.button("Search")
        if input_text and clicked:
            output = model_gensim.predict(input_text,df_gensim)
            recom = dict_als.get(input_text) 
            if recom:
                recom = [x[0] for x in recom][:input_number+1]
                result = df_gensim[df_gensim['item_id'].isin(recom)]
                st.dataframe(result)
            else:
                st.write("Not ID Customer in Dataset")
        
