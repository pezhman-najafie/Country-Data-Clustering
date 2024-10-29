# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator

# Load dataset
Data = pd.read_csv('Country_data.csv')

# Preview the dataset
df = pd.DataFrame(Data)

# Find the total number of missing values in the dataframe
df.isnull().sum()

# Drop rows with missing values
df.dropna(inplace=True)

# Drop 'country' column as it is not needed for clustering
df.drop(['country'], axis=1, inplace=True)

# Standardize the data
scaler = StandardScaler()
df_scale = scaler.fit_transform(df)

# Apply PCA with n_components=5
pca = PCA(n_components=5).fit(df_scale)
reduced_df = pca.transform(df_scale)

# Display the explained variance ratio
print("Explained Variance Ratio:")
print(np.cumsum(pca.explained_variance_ratio_))

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(reduced_df)

# Add the cluster labels to the original dataframe
df['Cluster'] = kmeans.labels_

# Display the dataframe with cluster labels
print(df)

# Perform PCA with a valid number of components
final_pca = PCA(n_components=9).fit(df_scale)
reduced_cr = final_pca.fit_transform(df_scale)


country_pca = pd.DataFrame(data=reduced_cr, columns=[f'principal component {i}' for i in range(1, 10)])




# Now we are going to implement Elbow method to find the optimal number of clusters
kmeans_set = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
cluster_range = range(2, 10)
cluster_errors = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, n_init=kmeans_set["n_init"], max_iter=kmeans_set["max_iter"], random_state=kmeans_set["random_state"])
    kmeans.fit(country_pca)
    cluster_errors.append(kmeans.inertia_)



# Elbow Method to find the optimal k
k1 = KneeLocator(range(2, 10), cluster_errors, curve='convex', direction='decreasing')


# Plot the Elbow Method without identified elbow point
plt.figure(figsize=(7, 5))
plt.style.use("fivethirtyeight")
plt.plot(range(2, 10), cluster_errors)
plt.xticks(range(2, 10))
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Plot the Elbow Method with the identified elbow point
plt.figure(figsize=(7, 5))
plt.style.use("fivethirtyeight")
plt.plot(range(2, 10), cluster_errors)
plt.xticks(range(2, 10))
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.axvline(x=k1.elbow, color='b', label='axvline-full height', ls='--')
plt.legend()
plt.show()

# Plot clusters in 2D space based on the first two principal components
plt.figure(figsize=(10, 6))
plt.scatter(reduced_df[:, 0], reduced_df[:, 1], c=kmeans.labels_, cmap='viridis', edgecolor='k', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-means Clustering Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# pass through the scaled data set into our PCA class object
pca = PCA().fit(df_scale)

# plot the Cumulative Summation of the Explained Variance
plt.figure(figsize=(10, 6))
plt.step(list(range(1, df_scale.shape[1] + 1)), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components', fontsize=15)
plt.ylabel('Variance (%)', fontsize=15)
plt.title('Explained Variance', fontsize=20)
plt.show()



plt.tight_layout()
plt.show()

