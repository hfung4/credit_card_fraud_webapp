import utils.predict as predict
import pandas as pd
from collections import OrderedDict
from pathlib import Path
import plotly.express as px
from PIL import Image
import streamlit as st
import fraud_analysis.evaluate as eval


st.title("Credit card fraud detection")
st.subheader("A Streamlit App by [Henry Fung](https://github.com/hfung4)")
st.text("")

# Get stock image
title_image = Image.open(Path("data", "fraud_detection_stock.jpg"))

st.image(title_image, width=500)

st.subheader("Background")
st.write(
    """ This is a Streamlit app that detects credit card fraudulent transactions. 
    To perform fraud detection, I trained a Random Forest classifier using a dataset containing transactions 
    made by credit cards in September 2013 by major European cardholders. The dataset was highly imbalanced
    with only 492 frauds out of 284,807 transactions.
    I created a [data pipeline](https://github.com/hfung4/credit_card_fraud_detection/tree/main/credit_card_fraud_production) to clean and process data-- including the use of resampling method (SMOTE) 
    to deal with class imabalance-- feature selection, and model training. The data pipeline is published on pypi as a package [here](https://pypi.org/project/tid-credit-card-fraud-prediction/).
"""
)

st.text("")
st.text("")
st.header("Upload Credit Card Transaction Data")
st.write(
    "The input data must contain PCA scores and attributes as described [here]('https://github.com/hfung4/credit_card_fraud_detection')."
)

X_test_file = st.file_uploader("Upload your credit card transaction data")

# Get data
if X_test_file is not None:
    X_test = pd.read_csv(X_test_file)

    st.markdown(f"**Your data is loaded! It has {X_test.shape[0]} records.**")

    # Run the rest of the code only after the user inputs test data
    st.text("")
    st.text("")
    st.header("Predictions")
    st.write(
        "A [pre-trained data pipeline](https://github.com/hfung4/credit_card_fraud_detection/tree/main/credit_card_fraud_production) will be used to process data and make predictions (Fraud or Not Fraud)"
    )
    res = predict.get_predictions(X_test)

    # Distribution of predictions

    # Predictions
    df_y_pred = pd.DataFrame(res["predictions"], columns=["predicted_is_fraud"])
    df_plot = df_y_pred.copy()
    label_mapping = OrderedDict({0: "No Fraud", 1: "Fraud"})
    df_plot["predicted_is_fraud"] = df_plot.predicted_is_fraud.astype("int").replace(
        label_mapping
    )

    fig = px.histogram(
        df_plot,
        x="predicted_is_fraud",
        barmode="overlay",
        marginal="box",
        labels="Transaction status",
    )

    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        font={"size": 18},
        xaxis_title="Transaction status",
    )

    st.plotly_chart(fig)

    st.text("")
    st.text("")

    # Model evaluation on test data
    st.header("Model evaluation using test data")
    st.write(
        """Ignore this section if the input data is new and unlabelled. If the
        labels of the input data is known (say, manually labeled by human analyst) and could
        be used to evaluate the trained model (e.g., to check if its performance has degraded),
        then please load the labels here.
        """
    )

    y_test_file = st.file_uploader(
        "Upload your credit card transaction labels (if available)"
    )
    # Get data
    if y_test_file is not None:
        y_test = pd.read_csv(y_test_file)
        st.markdown(f"**Your labels is loaded! There are {y_test.shape[0]} labels.**")

        # Evaluate the model on the test set
        eval.evaluate_on_test_set(
            y_test=y_test,
            y_pred=res["predictions"],
            y_pred_scores=res["predicted_proba"],
            save_dir=Path("outputs"),
        )

        st.text("")
        st.text("")

        # Confusion matrix
        st.subheader("Confusion matrix")

        # Get image
        cm_image = Image.open(Path("outputs", "confusion_matrix_test_set.png"))

        st.image(
            cm_image,
        )

        st.text("")
        st.text("")

        # Precision-Recall curve and average precision

        # PRC
        st.subheader("Precision-Recall Curve and Average Precision")

        # Get image
        prc_image = Image.open(Path("outputs", "prc_test_set.png"))

        st.image(prc_image, width=1000)
