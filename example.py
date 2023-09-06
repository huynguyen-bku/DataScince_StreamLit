import streamlit as st
import pandas as pd
import numpy as np

st.title("Hello - Chào các bạn")
st.header("welcome to our class")
st.subheader("GUI - Streamlit")

st.write('### Hello, *World!* :sunglasses:')

code = '''def hello():
    print("Hello, Streamlit!")'''
st.code(code, language='python')

df = pd.DataFrame(
   np.random.randn(50, 20),
   columns=('col %d' % i for i in range(20)))

st.dataframe(df.head())  # Same as st.write(df)

st.table(df.tail())

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

st.line_chart(chart_data)

# https://docs.streamlit.io/library/api-reference/widgets/st.radio
genre = st.radio(
    "What's your favorite movie genre",
    [":rainbow[Comedy]", "***Drama***", "Documentary :movie_camera:"])

if genre == ':rainbow[Comedy]':
    st.write('You selected comedy.')
else:
    st.write("You didn\'t select comedy.")
    
gender = st.radio(
    "Giới tính",
    ["Nữ", "Nam", "Khác"])

# if gender == 'Nữ':
#     st.write('Bạn là Nữ')
# else:
#     st.write("Bạn không phải là Nữ")
    
number = st.number_input('Insert a number')
# st.write('The current number is ', number)

if st.button("In kết quả"):
    if gender == 'Nữ':
        st.write('Bạn là Nữ')
    else:
        st.write("Bạn không phải là Nữ")
    st.write('The current number is ', number)
    
if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')
    
from PIL import Image
image = Image.open('ham_spam.jpg')
st.image(image, caption='Ham vs Spam')