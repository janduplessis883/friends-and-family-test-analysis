# example/st_app.py

import streamlit as st
from streamlit_gsheets import GSheetsConnection

st.header('Friend & Family Test Analysis')

url = "https://docs.google.com/spreadsheets/d/1K2d32XmZQMdGLslNzv2ZZoUquARl6yiKRT5SjUkTtIY/edit#gid=1323317089"

conn = st.experimental_connection("gsheets", type=GSheetsConnection)

data = conn.read(spreadsheet=url)
data.columns = ['time', 'score', 'freetext', 'freetext2']
st.dataframe(data)