import numpy as np
import pandas as pd

# Step 1: Read dataset from CSV
data = pd.read_csv("input.csv")
print("Dataset:\n", data)

# Convert to numpy array
X = data.values

# Step 2: Original dimensions
original_dim = X.shape[1]

# Step 3: Mean centering
mean = np.mean(X, axis=0)
X_centered = X - mean

# Step 4: Covariance matrix
cov_matrix = np.cov(X_centered.T)

print("\nCovariance Matrix:\n", cov_matrix)

# Step 5: Eigen values and Eigen vectors
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

print("\nEigen Values:\n", eigen_values)
print("\nEigen Vectors:\n", eigen_vectors)

# Step 6: Sort eigen values in descending order
idx = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[idx]
eigen_vectors = eigen_vectors[:, idx]

print("\nSorted Eigen Values:\n", eigen_values)
print("\nSorted Eigen Vectors:\n", eigen_vectors)

# Step 7: Choose number of components
# Eigen value > 1 rule
k = np.sum(eigen_values > 1)

# Step 8: Dimensionality reduction
principal_components = eigen_vectors[:, :k]
X_reduced = np.dot(X_centered, principal_components)

# Step 9: Final output
print("\nNumber of selected Principal Components:", k)
print("Original Dimensions:", original_dim)
print("Reduced Dimensions:", k)

print("\nFinal Reduced Data:\n", X_reduced)
