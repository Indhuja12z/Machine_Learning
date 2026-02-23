import gradio as gr
import numpy as np
import pickle

with open("diabetes_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(preg, glucose, bp, skin, insulin, bmi, pedigree, age):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, pedigree, age]])
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0].max()
    result = "DIABETIC" if pred == 1 else "HEALTHY"
    return f"{result} ({prob*100:.2f}%)"

with gr.Blocks(title="Diabetes Prediction") as demo:

    gr.Markdown("## Diabetes Prediction System")

    with gr.Row():
        with gr.Column():
            preg = gr.Number(label="Pregnancies")
            glucose = gr.Number(label="Glucose")
            bp = gr.Number(label="Blood Pressure")
            skin = gr.Number(label="Skin Thickness")

        with gr.Column():
            insulin = gr.Number(label="Insulin")
            bmi = gr.Number(label="BMI")
            pedigree = gr.Number(label="Pedigree")
            age = gr.Number(label="Age")

    btn = gr.Button("Predict")
    output = gr.Textbox(label="Result")

    btn.click(
        predict,
        inputs=[preg, glucose, bp, skin, insulin, bmi, pedigree, age],
        outputs=output
    )

demo.launch()
