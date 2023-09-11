import json
import pickle
import random
import math

import numpy as np
import pandas as pd
import streamlit as st
# import seaborn as sns
# import matplotlib.pyplot as plt

from datetime import datetime
from modules.model_gensim import ModelGensim
# from modules.model_als import mode_als

# 
def display_product(result):
    idx = 0
    for _ in range(math.ceil(result.shape[0]/4)):
        cols = st.columns(4)
        i = 0 
        while idx < result.shape[0] and i < 4:
            with cols[i]:
                ele = result.iloc[idx]
                name = ele["name"] if len(ele["name"]) <= 80 else ele["name"][:80] + '...'
                st.write(name)
                st.image(ele["image"])
                st.write(str(ele['price']) + "đ")
                st.write('⭐'*int(ele['rating']))
            idx += 1
            i+= 1
    return None
# ------------ Load Mode ------------ # 
# --- Content Base --- #
with open('checkpoint/model_gensim.pkl', 'rb') as inp:
    model_gensim = pickle.load(inp)
path_gensim = 'data/ProductRaw.csv'
df_gensim = pd.read_csv(path_gensim)
df_gensim = df_gensim.dropna()
# -------------------- #
# --- Collaboration --- #
# dataframe
path_als = 'data/ReviewRaw.csv'
df_als = pd.read_csv(path_als)[["customer_id", "product_id", "name", "rating"]].dropna()
# model 
json_file_path = "checkpoint/model_als.json"
with open(json_file_path, 'r') as j:
    dict_als = json.loads(j.read())
# --------------------- #
# ----------------------------------- # 

# ------------ GUI ------------ #
st.title("Data Science Project 2 ")
# menu
menu = ["Business Objective", "Build Project", "Content-base", "Collaboration"]
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
        df_train = df_gensim[['item_id', 'name', 'description', 'group']]
        df_train = df_train.dropna()
        df_train = df_train.drop_duplicates()
        df_train["name_view_group"] =  df_train['name'] + ' ' + df_train['description'] + ' ' + df_train['group']
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
        
elif choice == 'Content-base':
    # Upload file
    st.subheader("Product")
    # paramenter
    col1, col2 = st.columns([1,3])
    with col1:
        search_type = st.radio(
            "Select Search Type",
            options=["Select", "Text"],
        )
    with col2:
        flat = True 
        text = ''
        if search_type == "Text":
            input_text = st.text_input('Text')
            text = input_text
            flat = False
        elif  search_type == "Select":
            input_text = st.selectbox('Choice product', df_gensim["name"].sample(20,random_state =42))
            # preprocess
            save = df_gensim[df_gensim['name'] == input_text].iloc[0]
            cont =st.container()
            cont.write(save["name"])
            cont.image(save["image"], width = 250)
            cont.write(str(save['price']) + "đ")
            cont.write('⭐'*int(save['rating']))
            text = save['name'] + save['description']
            # input_number 
        input_number = int( st.slider('Number', 0, 10, 5))
        clicked = st.button("Search")
    # Search 
    if text and clicked:
        output = model_gensim.predict(text,df_gensim)
        result = output.iloc[1:input_number+1] if flat else  output.iloc[:input_number]
        display_product(result)
            
elif choice == "Collaboration":        
    input_text = st.selectbox('Choice Name Customer', df_als['name'].sample(20,random_state =42))
    # preprocess
    save = df_als[df_als['name'] == input_text].iloc[0]
    id_cus = save['customer_id']
    input_number = int( st.slider('Number', 0, 10, 5))
    clicked = st.button("Search")
    if id_cus and clicked:
        recom = dict_als.get(str(id_cus)) 
        if recom:
            recom = [x[0] for x in recom][:input_number+1]
            result = df_gensim[df_gensim['item_id'].isin(recom)]
            display_product(result)
        else:
            st.write("Not ID Customer in Dataset")
        
        # sanr phaamr da mua 
        st.write("những sản phẩm đã mua ")
        list_pro = df_als[df_als["customer_id"] == id_cus]["product_id"].tolist()
        result2 = df_gensim[df_gensim['item_id'].isin(list_pro)]
        display_product(result2)
