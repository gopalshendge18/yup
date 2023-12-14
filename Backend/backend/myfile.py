import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, DBSCAN
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

# Function to calculate clustering accuracy
def calculate_accuracy(true_labels, predicted_labels):
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    return ari, nmi

# Generate sample data with ground truth labels
X, true_labels = make_blobs(n_samples=300, centers=4, random_state=42)

# Initialize DataFrame to store results
results = pd.DataFrame(columns=['Algorithm', 'ARI', 'NMI'])

# AGNES (Hierarchical clustering)
agnes = AgglomerativeClustering(n_clusters=4, linkage='ward')
agnes_labels = agnes.fit_predict(X)
ari, nmi = calculate_accuracy(true_labels, agnes_labels)
results = results._append({'Algorithm': 'AGNES', 'ARI': ari, 'NMI': nmi}, ignore_index=True)

# DIANA (Hierarchical clustering)
diana = AgglomerativeClustering(n_clusters=4, linkage='single')
diana_labels = diana.fit_predict(X)
ari, nmi = calculate_accuracy(true_labels, diana_labels)
results = results._append({'Algorithm': 'DIANA', 'ARI': ari, 'NMI': nmi}, ignore_index=True)

# Perform k-Means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
ari, nmi = calculate_accuracy(true_labels, kmeans_labels)
results = results._append({'Algorithm': 'k-Means', 'ARI': ari, 'NMI': nmi}, ignore_index=True)

# Perform k-Medoids (PAM)
distance_matrix = calculate_distance_matrix(X)
medoids_instance = kmedoids(distance_matrix, initial_index_medoids=np.random.randint(0, len(X), 4))
medoids_instance.process()
kmedoids_labels = [cluster_id for cluster_id, cluster in enumerate(medoids_instance.get_clusters()) for _ in cluster]
ari, nmi = calculate_accuracy(true_labels, kmedoids_labels)
results = results.append({'Algorithm': 'k-Medoids', 'ARI': ari, 'NMI': nmi}, ignore_index=True)

# BIRCH
from sklearn.cluster import Birch
birch = Birch(n_clusters=4)
birch_labels = birch.fit_predict(X)
ari, nmi = calculate_accuracy(true_labels, birch_labels)
results = results.append({'Algorithm': 'BIRCH', 'ARI': ari, 'NMI': nmi}, ignore_index=True)

# Perform DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)
ari, nmi = calculate_accuracy(true_labels, dbscan_labels)
results = results.append({'Algorithm': 'DBSCAN', 'ARI': ari, 'NMI': nmi}, ignore_index=True)

# Display the results table
print(results)
