import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud
import seaborn as sns
from datetime import datetime, timedelta
from datetime import date
from matplotlib.patches import Patch
import time
from openai import OpenAI
import streamlit_shadcn_ui as ui
import requests
import plotly.graph_objects as go
import plotly.express as px

client = OpenAI()

from utils import *

st.set_page_config(page_title="AI MedReview: FFT")

html = """
<style>
.gradient-text {
    background: linear-gradient(45deg, #284d74, #d8ad45, #ae4f4d);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-size: 3em;
    font-weight: bold;
}
</style>
<div class="gradient-text">AI MedReview v2</div>

"""
# Using the markdown function with HTML to center the text
ui.badges(badge_list=[("NEW", "destructive")], class_name="flex gap-2", key="badges1")
st.markdown(html, unsafe_allow_html=True)



st.markdown("""We've fine-tuned our platform for rapid performance, ensuring a swift and seamless experience. Now, we're excited to announce that AI MedReview v2 supports multiple Primary Care Networks (PCNs)! PDF Report coming soon.

**👉 Get Started with AI MedReview v2**
Ready to explore the new version? Click the button below for a direct journey to AI MedReview v2.""")

ui.link_button(text="Go To AI MedReview v2", url="https://ai-medreview.streamlit.app", key="link_btn1")

st.markdown("""### Support Resources

Updated supporting documentation. Including detail of the new **FFT Friends & Family Test Form** used with v2, and your surgery short URLs to use with SMS and emals. Download your new QR Codes.

For detailed insights and guidance, click the button below to visit our dedicated Notion page.
            """)
ui.link_button(text="Go To Notion Support Page", url="https://janduplessis.notion.site/AI-MedReview-v2-9c62df309c87463584b4d89252508d07?pvs=4", key="link_btn2")