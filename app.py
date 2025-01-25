import pickle
import numpy as np
import streamlit as st

filename = 'final_model.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .container {
        max-width: 700px;
        margin: 50px auto;
        background-color: #f8f9fa;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #3498db;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        font-weight: 400;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 30px;
    }
    .input-section {
        font-size: 16px;
        color: #7f8c8d;
        margin-bottom: 20px;
    }
    .predict-button {
        background-color: #3498db;
        border: none;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 20px;
        text-align: center;
        border-radius: 12px;
        cursor: pointer;
        transition: transform 0.3s ease;
        width: 100%;
    }
    .predict-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(255, 255, 255, 0.3);
    }
    .prediction {
        font-size: 24px;
        font-weight: bold;
        color: #1dd1a1;
        text-align: center;
        margin-top: 30px;
        animation: fadeIn 1s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .footer {
        text-align: center;
        color: #95a5a6;
        font-size: 14px;
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="container">', unsafe_allow_html=True)

st.markdown('<p class="title">Weight Prediction App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter your height in feet, and we will predict your weight accurately!</p>', unsafe_allow_html=True)

default_height = 5.8
height_input = st.number_input(
    "Height (in feet)", 
    value=default_height, 
    min_value=0.0, 
    max_value=10.0, 
    step=0.1, 
    format="%.1f"
)

if st.button("Predict Weight", key="predict", help="Click to predict your weight based on the height provided"):
    height_input_2d = np.array(height_input).reshape(1, -1)
    predicted_weight = loaded_model.predict(height_input_2d)

    if len(predicted_weight.shape) == 1:
        predicted_value = predicted_weight[0]
    elif len(predicted_weight.shape) == 2:
        predicted_value = predicted_weight[0, 0]
    else:
        st.error("Unexpected model output. Please check the model.")
        predicted_value = None

    if predicted_value is not None:
        st.markdown(
            f'<p class="prediction">Predicted Weight: {predicted_value:.2f} kg</p>',
            unsafe_allow_html=True
        )

st.markdown(
    '<p class="footer">Designed and Developed by <strong>Mohammed Shoeb</strong></p>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
