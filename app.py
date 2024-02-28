import streamlit

from organized_layout import *
from helperfunctions import *


intro_text = """
Welcome to the ModelCraft Hub for data analysis and ML Implementation.
EDA analysis is an important step 
in AI model development or Data Analysis. This app 
offers visualisation of descriptive statistics of a 
csv input file and further ML implementations. 
<br> There are 2 options you can use: 
<li>  Analyse 1 file using <a href = "https://docs.profiling.ydata.ai/latest/"> ydata-profiling </a>
<li>  Apply various classification or regression models, including logistic regression, random forest
&nbsp&nbsp&nbsp&nbsp&nbsp classifier, decision tree classifier, random forest regressor, decision tree regressor, and 
<br> &nbsp&nbsp&nbsp&nbsp&nbsp gradient boost regressor to derive insights and predictions from your data  </a> </li>
<br> The EDA PDF report is available for download upon completion of each analysis.
"""


def main():
    # Set the initial background
    set_base()

    # Create the container
    container = st.container()
    script = """<div id = 'chat_outer'></div>"""
    st.markdown(script, unsafe_allow_html=True)

    # Add an expander for more information
    with container:
        intro = st.expander("Click here for more info on this app section and packages used")
        with intro:
            sub_text(intro_text)

    with container:
        script = """<div id = 'chat_inner'></div>"""
        st.markdown(script, unsafe_allow_html=True)

    # apply container style
    chat_plh_style = """<style>
    div[data-testid='stVerticalBlock']:has(div#chat_inner):not(:has(div#chat_outer)) 
    {background-color:#FFFFFF;
    width: 730px;
    border-radius: 8px;
    padding: 10px;
    }; 
    </style>
    """

    st.markdown(chat_plh_style, unsafe_allow_html=True)

    # Upload data or use demo data
    with container:
        data = upload_data()

    # Display header for Step 2
    display_header("Step 2", "Apply EDA analysis to dataset", is_sidebar=True)

    # Apply EDA to file
    with container:
        # Set up the selectbox for ML models
        eda_option = st.sidebar.radio("Select an option", analysis_options, index=None)
        if eda_option is not None:
            eda(eda_option, data)

    # Display header for Step 4
    display_header("Step 4", "Choose a ML model", is_sidebar=True)
    # Select ML model option
    ml_option = select_insert(ml_options, "None")

    # Display model performance metrics
    with container:
        if ml_option and data is not None:
            st.session_state.ml_option_ = ml_option
            x_train, x_test, y_train, y_test = preprocessing(data)
            apply_ml_model(ml_option, x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
