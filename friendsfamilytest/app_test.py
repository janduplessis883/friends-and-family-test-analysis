import streamlit as st
import time
import pandas as pd
import numpy as np
from streamlit_extras.grid import grid
from streamlit_extras.app_logo import add_logo
from streamlit_extras.mention import mention
from streamlit_extras.buy_me_a_coffee import button


def example2():
    if st.checkbox("Use url", value=True):
        add_logo("http://placekitten.com/120/120")
    else:
        add_logo("gallery/kitty.jpeg", height=300)
    st.write("👈 Check out the cat in the nav-bar!")


example2()


def example():
    random_df = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])

    my_grid = grid(2, [2, 4, 1], 1, 4, vertical_align="bottom")

    # Row 1:
    my_grid.dataframe(random_df, use_container_width=True)
    my_grid.line_chart(random_df, use_container_width=True)
    # Row 2:
    my_grid.selectbox("Select Country", ["Germany", "Italy", "Japan", "USA"])
    my_grid.text_input("Your name")
    my_grid.button("Send", use_container_width=True)
    # Row 3:
    my_grid.text_area("Your message", height=40)
    # Row 4:
    my_grid.button("Example 1", use_container_width=True)
    my_grid.button("Example 2", use_container_width=True)
    my_grid.button("Example 3", use_container_width=True)
    my_grid.button("Example 4", use_container_width=True)
    # Row 5 (uses the spec from row 1):
    with my_grid.expander("Show Filters", expanded=True):
        st.slider("Filter by Age", 0, 100, 50)
        st.slider("Filter by Height", 0.0, 2.0, 1.0)
        st.slider("Filter by Weight", 0.0, 100.0, 50.0)
    my_grid.dataframe(random_df, use_container_width=True)


example()


def example4():
    button(username="janduplessis883", floating=False, width=221)


example4()
