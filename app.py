# gradio app 

import gradio as gr
import pandas as pd
import pickle
import numpy as np

# =========================
# 1. Load the Model
# =========================
with open("water_potability.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# 2. Prediction Function
# =========================
def predict_water(ph, Hardness, Solids, Chloramines, Sulfate,
                  Conductivity, Organic_carbon, Trihalomethanes, Turbidity):
    
    # Pack inputs into DataFrame
    # Column names MUST match training data
    input_df = pd.DataFrame([[
        ph, Hardness, Solids, Chloramines, Sulfate,
        Conductivity, Organic_carbon, Trihalomethanes, Turbidity
    ]],
    columns=[
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
        'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ])
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    # Return result
    if prediction == 1:
        return "✅ Water is Safe for Drinking (Potable)"
    else:
        return "❌ Water is NOT Safe for Drinking"

# =========================
# 3. Input UI
# =========================
inputs = [
    gr.Number(label="pH", value=7),
    gr.Number(label="Hardness", value=200),
    gr.Number(label="Solids", value=10000),
    gr.Number(label="Chloramines", value=7),
    gr.Number(label="Sulfate", value=300),
    gr.Number(label="Conductivity", value=400),
    gr.Number(label="Organic Carbon", value=10),
    gr.Number(label="Trihalomethanes", value=80),
    gr.Number(label="Turbidity", value=3)
]

# =========================
# 4. Gradio Interface
# =========================
app = gr.Interface(
    fn=predict_water,
    inputs=inputs,
    outputs="text",
    title="💧 Water Potability Predictor",
    description="Enter water parameters to check if the water is safe for drinking."
)

# =========================
# 5. Launch App
# =========================
app.launch(share=True)