# Mall / Wholesale Clustering (Simplified: KMeans vs GMM)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# =========================
# 1. LOAD DATA + HEAD
# =========================

df = pd.read_csv("Wholesale customers data.csv")

print("\n===== DATASET HEAD =====")
print(df.head())

features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X = df[features]

# =========================
# 2. PREPROCESSING
# =========================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# =========================
# 3. KMEANS
# =========================

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X_scaled)

sil_km = silhouette_score(X_scaled, km_labels)

# =========================
# 4. GMM
# =========================

gmm = GaussianMixture(n_components=4, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

sil_gmm = silhouette_score(X_scaled, gmm_labels)

# =========================
# 5. SILHOUETTE SCORES
# =========================

print("\n===== SILHOUETTE SCORES =====")
print("KMeans:", round(sil_km, 3))
print("GMM   :", round(sil_gmm, 3))

# =========================
# 6. PLOTS (DISPLAY ONLY)
# =========================

# KMeans Plot
plt.figure(figsize=(6,5))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=km_labels, palette="Set1")
plt.title("KMeans Clustering")
plt.show()

# GMM Plot
plt.figure(figsize=(6,5))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=gmm_labels, palette="Set2")
plt.title("GMM Clustering")
plt.show()

# =========================
# 7. 5 TEST CASE PREDICTIONS
# =========================

test_cases = pd.DataFrame({
    'Fresh':            [50000,  1000,  5000, 20000,  200],
    'Milk':             [ 2000, 15000,  8000,  3000, 8000],
    'Grocery':          [ 3000, 20000, 12000,  5000,18000],
    'Frozen':           [10000,   500,  1500,  8000,  300],
    'Detergents_Paper': [  500,  8000,  4000,   400, 9000],
    'Delicassen':       [ 1000,  2000,  1500,  1000, 1200],
})

# Scale
test_scaled = scaler.transform(test_cases)

# Predictions
km_preds = kmeans.predict(test_scaled)
gmm_preds = gmm.predict(test_scaled)

# Table output
results = pd.DataFrame({
    "Test Case": ["TC1","TC2","TC3","TC4","TC5"],
    "KMeans Cluster": km_preds,
    "GMM Cluster": gmm_preds
})

print("\n===== TEST CASE COMPARISON =====")
print(results)