# Import packages
import glob
import streamlit as st
import numpy as np
import base64
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import profile_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, \
    mean_squared_error


all_options = \
    ["yes", "no", "Logistical Regression", "Random Forest Classifier",
     "Decision Tree Classification", "Random Forest Regressor", "Decision Tree Regressor", "Gradient Boost Regressor"]
ml_options = all_options[2:]
analysis_options = all_options[:2]


def set_background(main_bg):
    """
    A function to unpack an image from root folder and set as background.
    The bg will be static and won't take resolution of device into account.
    -------
    :param main_bg: str -> The path to the background image.
    """
    background_style = f"""
         <style>
         .stApp {{
             background: url(data:image/{"png"};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: 100% 120%;
         }}
         </style>
         """
    st.markdown(background_style, unsafe_allow_html=True)


def display_header(main_txt, sub_txt, is_sidebar=False):
    """
    function to display major headers at user interface
    ----------
    :param main_txt: str -> the major text to be displayed
    :param sub_txt: str -> the minor text to be displayed
    :param is_sidebar: bool -> check if its side panel or major panel
    """

    html_text = f"""
    <h2 style = "color:#F74369; text-align:center; font-weight: bold;"> {main_txt} </h2>
    <p style = "color:#BB1D3F; text-align:center; font-weight: 100px;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_text, unsafe_allow_html=True)
    else:
        st.markdown(html_text, unsafe_allow_html=True)


def sub_text(text):
    """
    A function to neatly display expander text in app.
    ----------
    :param text : plain text.
    """

    html_temp = f"""
    <p style = "color:#1F4E79; text_align:justify;"> {text} </p>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)


def select_insert(options, holder):
    """
    function to display selectbox options at user interface
    ----------
    :param options: list -> labels for the selectbox options
    :param holder: str -> a string to display when no options are selected
    """
    # if "options" not in st.session_state:
    #     st.session_state["options"]":

    option = st.sidebar.selectbox("Select an option ", options, index=None, placeholder=holder)
    if option not in options:
        st.sidebar.warning("Please select step in sidebar")
        return option
    else:
        st.session_state.ml_option_ = option
        return option


@st.cache_data(experimental_allow_widgets=True)
def eda(eda_option, uploaded_file):
    """
    Return Pandas profiling analysis of the file.
    ----------
    :param eda_option: str -> user choice
    :param uploaded_file: a pd.dataframe
    """
    if eda_option == "yes" and uploaded_file is not None:
        st.subheader("Step 2: Pandas Profiling Report")
        pr = uploaded_file.profile_report()
        with st.expander("Click here for full report below", expanded=True):
            st_profile_report(pr)
        st.session_state.eda_option = eda_option

        # display header for step 4 and create an export file of report
        display_header("Step 3", "Download report", is_sidebar=True)
        export = pr.to_html()
        st.sidebar.download_button(label=":file_folder:", data=export, file_name='report.html')

def preprocessing(uploaded_file):
    """
    Split data into test-train
    ----------
    :param uploaded_file: a pd.DataFrame
    :return: tuple containing x_train, x_test, y_train, y_test.

    """
    # Splitting the merged dataframe into input and output
    if uploaded_file is not None:
        x = uploaded_file.iloc[:, :-1]
        y = uploaded_file.iloc[:, -1]

        # Splitting into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
        st.session_state.x_train_, st.session_state.x_test_, st.session_state.y_train_, st.session_state.y_test_ = \
            x_train, x_test, y_train, y_test
        return x_train, x_test, y_train, y_test

# Classifier Ml models


@st.cache_data
def logistical_regression(x_train, x_test, y_train, y_test):
    """
    Perform logistic regression and display performance metrics.
    ----------
    :param x_train: pd.DataFrame -> Training input data.
    :param x_test: pd.DataFrame -> Testing input data.
    :param y_train: pd.Series -> Training output data.
    :param y_test: pd.Series -> Testing output data.
    :returns Tuple containing accuracy and confusion matrix display.
    """

    # Create Random Forest Regressor model and train
    log_r = LogisticRegression()
    log_r.fit(x_train, y_train)

    # predict the response for the test data
    y_pred_log = log_r.predict(x_test)

    # View accuracy score and confusion matrix
    accuracy_log = accuracy_score(y_test, y_pred_log)
    log_cm = confusion_matrix(y_test, y_pred_log)
    display_log = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=log_r.classes_)
    st.session_state.model_result_ = accuracy_log, display_log
    fig_log, ax_log = plt.subplots()
    display_log.plot(ax=ax_log)
    st.pyplot(fig_log)

    return accuracy_log, display_log


@st.cache_data
def random_forest_classifier(x_train, x_test, y_train, y_test):
    """
    Perform random forest classifier and display performance metrics.
    -------
    :param x_train: pd.DataFrame -> Training input data.
    :param x_test: pd.DataFrame -> Testing input data.
    :param y_train: pd.Series -> Training output data.
    :param y_test: pd.Series -> Testing output data.
    :returns Tuple containing accuracy and confusion matrix display.
    """

    # Create model and train
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)

    # predict the response for the test data
    y_pred_rfc = rfc.predict(x_test)

    # View accuracy score and confusion matrix
    accuracy_rfc = accuracy_score(y_test, y_pred_rfc)
    rfc_cm = confusion_matrix(y_test, y_pred_rfc)
    display_rfc = ConfusionMatrixDisplay(confusion_matrix=rfc_cm, display_labels=rfc.classes_)
    st.session_state.model_result_ = accuracy_rfc, display_rfc
    fig, ax = plt.subplots()
    display_rfc.plot(ax=ax)
    st.pyplot(fig)

    return accuracy_rfc, display_rfc


@st.cache_data
def dtreeclassifier(x_train, x_test, y_train, y_test):
    """
    Perform decision tree classifier and display performance metrics.
    -------
    :param x_train: pd.DataFrame -> Training input data.
    :param x_test: pd.DataFrame -> Testing input data.
    :param y_train: pd.Series -> Training output data.
    :param y_test: pd.Series -> Testing output data.
    :returns Tuple containing accuracy and confusion matrix display.
    """

    # Create model and train
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)

    # Predict the response for the test data
    y_pred_dtc = dtc.predict(x_test)

    # View accuracy score and confusion matrix
    accuracy_dtc = accuracy_score(y_test, y_pred_dtc)
    dtc_cm = confusion_matrix(y_test, y_pred_dtc)
    display_dtc = ConfusionMatrixDisplay(confusion_matrix=dtc_cm, display_labels=dtc.classes_)
    st.session_state.model_result_ = accuracy_dtc, display_dtc
    fig, ax = plt.subplots()
    display_dtc.plot(ax=ax)
    st.pyplot(fig)

    return accuracy_dtc, display_dtc

# Regression Ml models


# DecisionTree Regressor
@st.cache_data
def dtreeregressor(x_train, x_test, y_train, y_test):
    """
    Perform decision tree regressor and displays performance metrics.

    ----------
    :param x_train: pd.DataFrame -> Training input data.
    :param x_test: pd.DataFrame -> Testing input data.
    :param y_train: pd.Series -> Training output data.
    :param y_test: pd.Series -> Testing output data.
    :returns Tuple containing MSE and RMSE.
    """

    # Create model and train
    dtr = DecisionTreeRegressor()
    dtr.fit(x_train, y_train)

    # Predict the response for the test data
    y_pred_dtr = dtr.predict(x_test)

    # View mean squared error and root mean squared error score
    tree_mse = mean_squared_error(y_test, y_pred_dtr)
    tree_rmse = np.sqrt(tree_mse)
    return tree_mse, tree_rmse


# Random Forest Regressor
@st.cache_data
def random_forest_regressor(x_train, x_test, y_train, y_test):
    """
    Perform random forest regressor and displays performance metrics.

    ----------
    :param x_train: pd.DataFrame -> Training input data.
    :param x_test: pd.DataFrame -> Testing input data.
    :param y_train: pd.Series -> Training output data.
    :param y_test: pd.Series -> Testing output data.
    :returns Tuple containing MSE and RMSE.
    """

    # Create Random Forest Regressor object and train
    rfr = RandomForestRegressor()
    rfr.fit(x_train, y_train)

    # Predict the response for the test data
    y_pred_rfr = rfr.predict(x_test)

    # View mean squared error and root mean squared error score
    rfr_mse = mean_squared_error(y_test, y_pred_rfr)
    rfr_rmse = np.sqrt(rfr_mse)
    return rfr_mse, rfr_rmse


@st.cache_data
def gboost_regressor(x_train, x_test, y_train, y_test):
    """
    Perform gradient boost regressor and displays performance metrics.

    ----------
    :param x_train: pd.DataFrame -> Training input data.
    :param x_test: pd.DataFrame -> Testing input data.
    :param y_train: pd.Series -> Training output data.
    :param y_test: pd.Series -> Testing output data.
    :returns Tuple containing MSE and RMSE.
    """

    # Create Gradient Boost Regressor object and train
    gbr = XGBRegressor(random_state=42)
    gbr.fit(x_train, y_train)

    # Predict the response for the test data
    y_pred_gbr = gbr.predict(x_test)

    # View mean squared error and root mean squared error score
    gbr_mse = mean_squared_error(y_test, y_pred_gbr)
    gbr_rmse = np.sqrt(gbr_mse)
    return gbr_mse, gbr_rmse


def apply_ml_model(ml_option, x_train, x_test, y_train, y_test):
    """
    Apply the selected ML model and display evaluation metrics.

    :param ml_option: str -> The selected ML model option.
    :param x_train: pd.DataFrame -> Training input data
    :param x_test: pd.DataFrame -> Testing input data
    :param y_train: pd.DataFrame -> Training output data
    :param y_test: pd.DataFrame -> Testing output data
    """
    try:
        # Set up the selectbox for ML models
        if ml_option in ml_options and x_train is not None:
            st.session_state.ml_option_ = ml_option

            ml_functions = {"Logistical Regression": logistical_regression,
                            "Random Forest Classifier": random_forest_classifier,
                            "Decision Tree Classification": dtreeclassifier,
                            "Random Forest Regressor": random_forest_regressor,
                            "Decision Tree Regressor": dtreeregressor,
                            "Gradient Boost Regressor": gboost_regressor}

            st.subheader("Step 3: Evaluating the ML model")
            st.divider()
            # st.write("Visualizing Classification Performance: Confusion Matrix Display")

            model_result = ml_functions[ml_option](x_train, x_test, y_train, y_test)
            st.session_state.model_result_ = model_result
            name = st.session_state.ml_option_.lstrip()

            if "Logistical" in ml_option or "Classification" in ml_option:
                st.write("Visualizing Classification Performance: Confusion Matrix Display")
                st.write(f"The accuracy score of this model: {name} is {round(model_result[0], 3)}")

            elif "Regressor" in ml_option:
                st.write(f"The mean square error of this model: {name} is {round(model_result[0], 3)}")
                st.write(f"The root mean square error of this model: {name} is {round(model_result[1], 3)}")
            st.divider()

    except:
        # display warning
        st.warning("Model is not suitable for this dataset")
