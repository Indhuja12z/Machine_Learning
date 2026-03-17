# Mall Customer Segmentation using KMeans with Visualization and Gradio UI

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score

import gradio as gr

# Step 1: Load Dataset

df = pd.read_csv("Mall_Customers.csv")

print("Dataset Loaded Successfully")
print(df.head())
print("Shape:", df.shape)


# Step 2: Preprocessing


X = df.drop("CustomerID", axis=1)

# Encode Gender
le = LabelEncoder()
X["Gender"] = le.fit_transform(X["Gender"])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train KMeans Model

kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nModel trained successfully.")

# Step 4: Clustering Performance

score = silhouette_score(X_scaled, df["Cluster"])

print("\nClustering Performance:")
print("Silhouette Score:", round(score, 3))

# Step 5: Cluster Labels

cluster_labels = {
    0: "Target Customer (High Income, High Spending)",
    1: "Frugal Shopper (High Income, Low Spending)",
    2: "Balanced Customer (Mid Income, Mid Spending)",
    3: "Impulsive Buyer (Low Income, High Spending)",
    4: "Sensible Shopper (Average Income, Average Spending)"
}

# Step 6: Visualization

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=df["Annual Income (k$)"],
    y=df["Spending Score (1-100)"],
    hue=df["Cluster"],
    palette="Set1",
    s=100
)

# Centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centroids[:, 2],
    centroids[:, 3],
    s=300,
    c="black",
    marker="X",
    label="Centroids"
)

plt.title("Customer Segmentation using KMeans")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.legend()

plt.show()

# Step 7: Prediction Function


def predict_customer(gender, age, income, score):

    input_data = pd.DataFrame(
        [[gender, age, income, score]],
        columns=[
            "Gender",
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    input_data["Gender"] = le.transform(input_data["Gender"])

    scaled_input = scaler.transform(input_data)

    cluster_id = kmeans.predict(scaled_input)[0]

    return cluster_labels[cluster_id]

# Step 8: Gradio UI

interface = gr.Interface(

    fn=predict_customer,

    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Number(label="Age"),
        gr.Number(label="Annual Income (k$)"),
        gr.Number(label="Spending Score (1-100)")
    ],

    outputs=gr.Textbox(label="Customer Segment"),

    title="Mall Customer Segmentation using KMeans",

    description="Enter customer details to predict customer type"
)

interface.launch()
