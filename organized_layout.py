import pandas as pd
from helperfunctions import *
import streamlit as st
import random


def set_base():
    """
    Display the foundational configuration and styling
    """
    # Set page configuration, background and header
    st.set_page_config(page_title="Prediction of Wild Fires", page_icon='📊', layout='centered')
    set_background("images/background_image.png")
    display_header("ModelCraft Hub",
                   "select, clean and visualize data for AI modeling", is_sidebar=False)

    # Set sidebar background
    st.sidebar.markdown("""
        <style>
            [data-testid=stSidebar] {
                background-color: #F4E9F0;
            }
        </style>
        """, unsafe_allow_html=True)
    st.sidebar.image("images/logo.png", use_column_width=True)

    # Hide hamburger icon menu & streamlit header
    hide_st_style = """
                <style>
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                 """
    st.markdown(hide_st_style, unsafe_allow_html=True)


@st.cache_data
def load_data():
    file_demo = random.randint(0, 1)
    files = ['Data3/merged_fire_sample.csv', 'Data3/bikes_sharing.csv']
    uploaded_data = pd.read_csv(files[file_demo])
    return uploaded_data


def upload_data():
    """
    function to upload dataset
    -------
    :return pd.DataFrame or None -> The uploaded dataset or None if no dataset is uploaded.
    """

    pd.set_option("styler.render.max_elements", 10**10)

    # Display header for Step 1
    display_header("Step 1", "Upload data", is_sidebar=True)

    load = st.sidebar.checkbox('Use demo data', value=False, help="Click to use a dataset. The application will "
                                                                  "randomly choose between a fires dataset and a "
                                                                  "bike-sharing dataset")
    if load:
        # display header
        st.divider()
        st.subheader("1.  Preview Data")
        st.divider()

        # load and view data
        uploaded_data = load_data()
        st.dataframe(data=uploaded_data, width=700)
        st.session_state.data = uploaded_data
        return uploaded_data
    else:
        uploaded_data = st.sidebar.file_uploader("Choose a dataset", type="csv",)

        if uploaded_data:
            # display header
            st.divider()
            st.subheader("1.  Preview Data")
            st.divider()

            # load and view data
            uploaded_data = pd.read_csv(uploaded_data)
            st.write(uploaded_data)
            st.session_state.data = uploaded_data
            return uploaded_data

        else:
            st.sidebar.warning("Please upload a dataset")
            return None

