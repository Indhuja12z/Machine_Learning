import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

# Load dataset
data = pd.read_csv("house_prices_practice.csv")

# Select feature and target
X = data["GrLivArea"].values.reshape(-1,1)
y = data["SalePrice"].values

# Gaussian kernel
def gaussian_kernel(x, x0, tau):
    return np.exp(-np.sum((x - x0)**2) / (2 * tau**2))

# Compute weights
def compute_weights(X, x0, tau):
    m = X.shape[0]
    weights = np.zeros(m)
    
    for i in range(m):
        weights[i] = gaussian_kernel(X[i], x0, tau)
        
    return np.diag(weights)

# Locally Weighted Regression
def locally_weighted_regression(X, y, x0, tau):

    X_b = np.c_[np.ones((X.shape[0],1)), X]
    x0_b = np.r_[1, x0]

    W = compute_weights(X, x0, tau)

    theta = np.linalg.inv(X_b.T @ W @ X_b) @ (X_b.T @ W @ y)

    return x0_b @ theta


# Prediction function for UI
def predict_price(area):

    area = np.array([area])
    price = locally_weighted_regression(X, y, area, tau=500)

    return f"Predicted House Price: {round(price,2)}"


# Gradio Interface
interface = gr.Interface(
    fn=predict_price,
    inputs=gr.Number(label="Enter House Area (GrLivArea)"),
    outputs="text",
    title="House Price Prediction using Locally Weighted Regression",
    description="Enter the house area to predict the price."
)

interface.launch()
