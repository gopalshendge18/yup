import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import HttpResponse, render
from .forms import CSVUploadForm
from .models import CSVData
from django.core.files.storage import FileSystemStorage
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from django.urls import path
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from django.conf import settings
from .models import CSVData
from sklearn.cluster import Birch
from scipy.cluster.hierarchy import dendrogram, linkage
from django.http import JsonResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import json
#####
def plot_dendrogram(Z):
    plt.figure(figsize=(10, 7))
    dendrogram(Z, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.savefig("C:\\Users\\Saurabh\\Desktop\\DM assignments\\dm_assignments\\dm_assignments\\dm_assignments\\static\\AGNES.png")


    # Save dendrogram plot to a BytesIO object
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    # Convert the plot to base64 for embedding in the API response
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    image_path = os.path.join(settings.MEDIA_ROOT, 'birch_plot.png')
    plt.savefig(image_path)
    plt.close()

    media_url = os.path.join(settings.MEDIA_URL, 'agnes_plot.png')
    return JsonResponse({'image_path': media_url})
    # Close the plot to release resources


    


def agnes_clustering(data):
    from sklearn.cluster import AgglomerativeClustering

    clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = clustering.fit_predict(data)

    # Generate dendrogram
    linked = linkage(data, 'ward')
    dendrogram_plot = plot_dendrogram(linked)

    return labels, dendrogram_plot


######

def plot_dendrogram_diana(data, labels):
    plt.figure(figsize=(10, 7))
    linkage_matrix = np.zeros((len(data) - 1, 4))

    # Custom linkage matrix for DIANA
    for i in range(1, len(data)):
        parent = np.unique(labels[:i])[-1]
        children = np.unique(labels[i:])
        linkage_matrix[i - 1, 0] = parent
        linkage_matrix[i - 1, 1] = children[0]
        linkage_matrix[i - 1, 2] = i
        linkage_matrix[i - 1, 3] = len(np.where(labels == parent)[0]) + len(np.where(labels == children[0])[0])

    dendrogram(linkage_matrix, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram (DIANA)')
    plt.xlabel('sample index')
    plt.ylabel('distance')

    # Save dendrogram plot to a BytesIO object
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)

    # Convert the plot to base64 for embedding in the API response
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

    # Close the plot to release resources
    plt.close()

    return img_base64


def perform_diana_clustering(data, n_clusters):
    def split_cluster(cluster_data):
        # Find the feature with maximum variance
        max_variance_feature = np.argmax(np.var(cluster_data, axis=0))

        # Sort data based on the chosen feature
        sorted_indices = np.argsort(cluster_data[:, max_variance_feature])

        # Split the cluster into two
        split_index = len(sorted_indices) // 2
        cluster1_indices = sorted_indices[:split_index]
        cluster2_indices = sorted_indices[split_index:]

        return cluster1_indices, cluster2_indices

    def recursive_diana(cluster_data, cluster_labels, remaining_clusters):
        if remaining_clusters == 1:
            return

        # Find the cluster with the highest variance
        max_variance_cluster = np.argmax([np.var(cluster_data[cluster_labels == cluster], axis=0).sum()
                                          for cluster in np.unique(cluster_labels)])

        # Split the cluster into two
        split_indices1, split_indices2 = split_cluster(cluster_data[cluster_labels == max_variance_cluster])

        # Update cluster labels
        cluster_labels[cluster_labels == max_variance_cluster] = max(cluster_labels) + 1

        # Recursively split the clusters
        recursive_diana(cluster_data[split_indices1], cluster_labels[split_indices1], remaining_clusters - 1)
        recursive_diana(cluster_data[split_indices2], cluster_labels[split_indices2], remaining_clusters - 1)

    # Initial clustering
    initial_labels = np.zeros(len(data))
    recursive_diana(data, initial_labels, n_clusters)

    # Generate dendrogram
    dendrogram_plot = plot_dendrogram_diana(data, initial_labels)

    return initial_labels, dendrogram_plot

@api_view(['GET'])
def hierarchical_clustering(request, method, format=None):
    # Sample data (replace this with your dataset)

    csv_data_id = 19  # Replace with the appropriate CSVData ID
    csv_data = CSVData.objects.get(id=csv_data_id)

    # Extract the file path from the model
    file_path = os.path.join(settings.MEDIA_ROOT, str(csv_data.csv_file))
    

    iris_data = pd.read_csv(file_path)

    # Extract features (assuming the first 4 columns are the features)
    data = iris_data.iloc[:, :4].values
    print(data)
    # data = np.array([
    #     [5.1, 3.5, 1.4, 0.2],
    #     [4.9, 3.0, 1.4, 0.2],
    #     [4.7, 3.2, 1.3, 0.2],
    #     [7.0, 3.2, 4.7, 1.4],
    #     [6.4, 3.2, 4.5, 1.5],
    #     [6.9, 3.1, 4.9, 1.5],
    #     [6.3, 3.3, 6.0, 2.5],
    #     [5.8, 2.7, 5.1, 1.9],
    #     [7.1, 3.0, 5.9, 2.1]
    # ])

    if method == 'agnes':
        labels, dendrogram_plot = agnes_clustering(data)
    elif method == 'diana':
        labels, dendrogram_plot = perform_diana_clustering(data,3)
    else:
        return Response({'error': 'Invalid clustering method'}, status=status.HTTP_400_BAD_REQUEST)

    return Response({'labels': labels.tolist(), 'dendrogram': dendrogram_plot,"name":node[0].name}, status=status.HTTP_200_OK)





def kmeans_clustering_algorithm(data, n_clusters, max_iters=100):
    num_points, num_features = data.shape

    # Initialize centroids randomly
    centroids = data[np.random.choice(num_points, n_clusters, replace=False)]

    for _ in range(max_iters):
        # Assign each point to the nearest centroid
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis, :] - centroids, axis=-1), axis=-1)

        # Update centroids
        new_centroids = np.array([data[labels == k].mean(axis=0) for k in range(n_clusters)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


def plot_kmeans_clusters(data, labels, centroids):
    plt.figure(figsize=(10, 7))
    colors = ['red', 'green', 'blue']

    # Plot data points
    for k in range(len(centroids)):
        cluster_points = data[labels == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k], label=f'Cluster {k + 1}')

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='black', label='Centroids')

    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig("C:\\Users\\Saurabh\\Desktop\\DM assignments\\dm_assignments\\dm_assignments\\dm_assignments\\static\\KMEANS.png")
    return
    


@api_view(['GET'])
def kmeans_clustering(request, format=None):
    # Sample data (replace this with your dataset)
    node=CSVFile.objects.all()
    print("..............")
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    

    iris_data = pd.read_csv(node[0].file)

    # Extract features (assuming the first 4 columns are the features)
    data = iris_data.iloc[:, :4].values
    print(data)
    

    n_clusters = 3

    labels, centroids = kmeans_clustering_algorithm(data, n_clusters)

    # Plot the results
    plot_kmeans_clusters(data, labels, centroids)

    return Response({'labels': labels.tolist(), 'centroids': centroids.tolist(),"name":node[0].name,}, status=status.HTTP_200_OK)










import numpy as np
import matplotlib.pyplot as plt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

def assign_points_to_medoids(data, medoids):
    distances = np.linalg.norm(data[:, np.newaxis, :] - medoids, axis=-1)
    labels = np.argmin(distances, axis=-1)
    return labels

def calculate_total_cost(data, labels, medoids):
    total_cost = 0
    for i, medoid in enumerate(medoids):
        cluster_points = data[labels == i]
        cluster_cost = np.linalg.norm(cluster_points - medoid, axis=-1).sum()
        total_cost += cluster_cost
    return total_cost

def plot_kmedoids_clusters(data, labels, medoids):
    plt.figure(figsize=(10, 7))
    colors = ['red', 'green', 'blue']

    # Plot data points
    for k in range(len(medoids)):
        cluster_points = data[labels == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k], label=f'Cluster {k + 1}')

    # Plot medoids
    medoids = np.array(medoids)
    plt.scatter(medoids[:, 0], medoids[:, 1], marker='X', s=200, c='black', label='Medoids')

    plt.title('k-Medoids Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig("C:\\Users\\Saurabh\\Desktop\\DM assignments\\dm_assignments\\dm_assignments\\dm_assignments\\static\\KMEDOID.png")
    return


def kmedoids_clustering_algorithm(data, n_clusters, max_iters=100):
    num_points, num_features = data.shape

    # Initialize medoids randomly
    medoids = data[np.random.choice(num_points, n_clusters, replace=False)]
    labels = assign_points_to_medoids(data, medoids)

    for _ in range(max_iters):
        # Find the cost of the current clustering
        current_cost = calculate_total_cost(data, labels, medoids)

        # Randomly select a non-medoid point
        non_medoid_indices = np.setdiff1d(np.arange(num_points), medoids)
        random_non_medoid = np.random.choice(non_medoid_indices)

        for i, medoid in enumerate(medoids):
            # Swap the medoid with the non-medoid point
            medoids[i] = random_non_medoid

            # Recalculate labels and cost
            new_labels = assign_points_to_medoids(data, medoids)
            new_cost = calculate_total_cost(data, new_labels, medoids)

            # If the new clustering has a lower cost, accept the swap
            if new_cost < current_cost:
                labels = new_labels
                current_cost = new_cost
            else:
                # Revert the medoid swap
                medoids[i] = medoid

    return medoids, labels





@api_view(['GET'])
def kmedoids_clustering(request, format=None):
    # Sample data (replace this with your dataset)
    node=CSVFile.objects.all()
    print("..............")
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    

    iris_data = pd.read_csv(node[0].file)

    # Extract features (assuming the first 4 columns are the features)
    data = iris_data.iloc[:, :4].values
    print(data)
    # data = np.array([
    #     [5.1, 3.5, 1.4, 0.2],
    #     [4.9, 3.0, 1.4, 0.2],
    #     [4.7, 3.2, 1.3, 0.2],
    #     [7.0, 3.2, 4.7, 1.4],
    #     [6.4, 3.2, 4.5, 1.5],
    #     [6.9, 3.1, 4.9, 1.5],
    #     [6.3, 3.3, 6.0, 2.5],
    #     [5.8, 2.7, 5.1, 1.9],
    #     [7.1, 3.0, 5.9, 2.1]
    # ])

    n_clusters = 3

    medoids, labels = kmedoids_clustering_algorithm(data, n_clusters)

    # Plot the results
    plot_kmedoids_clusters(data, labels, medoids)

    return Response({'medoids': medoids.tolist(), 'labels': labels.tolist(),"name":node[0].name,}, status=status.HTTP_200_OK)





def merge_clusters(clusters, cluster_centers, branching_factor):
    # Calculate pairwise distances between cluster centers
    distances = cdist(cluster_centers, cluster_centers)
    np.fill_diagonal(distances, np.inf)

    # Find the pair of clusters with the smallest distance
    min_indices = np.unravel_index(np.argmin(distances), distances.shape)
    cluster1, cluster2 = min_indices

    # Merge the two clusters
    clusters[cluster1] += clusters[cluster2]
    cluster_centers[cluster1] = np.mean(clusters[cluster1], axis=0)

    # Remove the merged cluster
    del clusters[cluster2]
    del cluster_centers[cluster2]

    return clusters, cluster_centers

def plot_birch_clusters(data, clusters, cluster_centers):
    plt.figure(figsize=(10, 7))
    colors = ['red', 'green', 'blue']

    for i, cluster in enumerate(clusters):
        cluster_points = np.array(cluster)
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i + 1}')

    cluster_centers = np.array(cluster_centers)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', s=200, c='black', label='Cluster Centers')

    plt.title('BIRCH Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig("C:\\Users\\Saurabh\\Desktop\\DM assignments\\dm_assignments\\dm_assignments\\dm_assignments\\static\\BIRCH.png")



def birch_clustering_algorithm(data, threshold, branching_factor):
    num_points, num_features = data.shape

    clusters = []
    cluster_centers = []  # Initialize as a list

    # Initialize the first cluster
    clusters.append([data[0]])
    cluster_centers.append(data[0])

    for i in range(1, num_points):
        point = data[i]

        # Find the nearest cluster center
        distances = np.linalg.norm(np.array(cluster_centers) - point, axis=1)
        nearest_cluster = np.argmin(distances)

        # Check if the point is within the threshold distance of the nearest cluster
        if distances[nearest_cluster] <= threshold:
            clusters[nearest_cluster].append(point)

            # Update the cluster center
            cluster_centers[nearest_cluster] = np.mean(clusters[nearest_cluster], axis=0)
        else:
            # If not, create a new cluster
            clusters.append([point])
            cluster_centers.append(point)

            # Adjust the number of clusters if the branching factor is exceeded
            if len(clusters) > branching_factor:
                clusters, cluster_centers = merge_clusters(clusters, cluster_centers, branching_factor)
    return clusters, np.array(cluster_centers) 





from scipy.spatial.distance import cdist


@api_view(['GET'])
def birch_clustering(request, format=None):
    # Sample data (replace this with your dataset)
    node=CSVFile.objects.all()
    print("..............")
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    

    iris_data = pd.read_csv(node[0].file)

    # Extract features (assuming the first 4 columns are the features)
    data = iris_data.iloc[:, :4].values
    print(data)

    threshold = 1.0
    branching_factor = 3

    clusters, cluster_centers = birch_clustering_algorithm(data, threshold, branching_factor)

    # Plot the results
    plot_birch_clusters(data, clusters, cluster_centers)

    return Response({'clusters': clusters, 'cluster_centers': cluster_centers,"name":node[0].name,}, status=status.HTTP_200_OK)







def find_neighbors(data, point_index, eps):
    distances = np.linalg.norm(data - data[point_index], axis=1)
    return np.where(distances <= eps)[0]

def expand_cluster(data, labels, point_index, neighbors, cluster_id, eps, min_samples):
    labels[point_index] = cluster_id

    i = 0
    while i < len(neighbors):
        current_point = neighbors[i]

        if labels[current_point] == -1:
            labels[current_point] = cluster_id
        elif labels[current_point] == 0:
            labels[current_point] = cluster_id

            new_neighbors = find_neighbors(data, current_point, eps)

            if len(new_neighbors) >= min_samples:
                neighbors = np.concatenate([neighbors, new_neighbors])
        
        i += 1  # Move the increment inside the loop

def group_clusters(labels):
    unique_labels = np.unique(labels)
    clusters = {}

    for label in unique_labels:
        if label == -1:
            continue

        cluster_points = np.where(labels == label)[0]
        clusters[label] = cluster_points.tolist()

    return clusters

def plot_dbscan_clusters(data, clusters):
    plt.figure(figsize=(10, 7))
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']

    for i, cluster_points in enumerate(clusters.values()):
        cluster_points = np.array([data[index] for index in cluster_points])
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i + 1}')

    plt.title('DBSCAN Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig("C:\\Users\\Saurabh\\Desktop\\DM assignments\\dm_assignments\\dm_assignments\\dm_assignments\\static\\DBSCAN.png")



def dbscan_clustering_algorithm(data, eps, min_samples):
    num_points, num_features = data.shape

    labels = np.zeros(num_points, dtype=int)
    cluster_id = 0

    for i in range(num_points):
        if labels[i] != 0:
            continue

        neighbors = find_neighbors(data, i, eps)

        if len(neighbors) < min_samples:
            labels[i] = -1  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(data, labels, i, neighbors, cluster_id, eps, min_samples)

    return labels, group_clusters(labels)



@api_view(['GET'])
def dbscan_clustering(request, format=None):
    # Sample data (replace this with your dataset)

    node=CSVFile.objects.all()
    print("..............")
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    

    iris_data = pd.read_csv(node[0].file)

    # Extract features (assuming the first 4 columns are the features)
    data = iris_data.iloc[:, :4].values
    print(data)

    eps = 0.5
    min_samples = 3

    labels, clusters = dbscan_clustering_algorithm(data, eps, min_samples)

    # Plot the results
    plot_dbscan_clusters(data, clusters)

    return Response({'labels': labels.tolist(), 'clusters': clusters,"name":node[0].name,}, status=status.HTTP_200_OK)


import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering, KMeans, Birch, DBSCAN

@api_view(['GET'])
def clustering_evaluation(request, format=None):
    # Load Iris dataset from CSV

    node=CSVFile.objects.all()
    print("..............")
    print(node[0].name)

    if len(node)==0 :
        return HttpResponse("No csv file in database !!")
    

    iris_data = pd.read_csv(node[0].file)

    # Extract features (assuming the first 4 columns are the features)
    data = iris_data.iloc[:, :4].values
    print(data)

    # Extract ground truth labels (assuming the last column is the ground truth labels)
    ground_truth_labels = iris_data.iloc[:, -1].values

    # Hierarchical Clustering (Agglomerative)
    agnes = AgglomerativeClustering(n_clusters=3)
    agnes_labels = agnes.fit_predict(data)
    agnes_silhouette = silhouette_score(data, agnes_labels)
    agnes_rand_index = adjusted_rand_score(ground_truth_labels, agnes_labels)

    # K-Means
    kmeans = KMeans(n_clusters=3)
    kmeans_labels = kmeans.fit_predict(data)
    kmeans_silhouette = silhouette_score(data, kmeans_labels)
    kmeans_rand_index = adjusted_rand_score(ground_truth_labels, kmeans_labels)

    # Birch
    birch = Birch(n_clusters=3)
    birch_labels = birch.fit_predict(data)
    birch_silhouette = silhouette_score(data, birch_labels)
    birch_rand_index = adjusted_rand_score(ground_truth_labels, birch_labels)

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    dbscan_labels = dbscan.fit_predict(data)
    dbscan_silhouette = silhouette_score(data, dbscan_labels)
    dbscan_rand_index = adjusted_rand_score(ground_truth_labels, dbscan_labels)

    # Create a dictionary with the results
    results = {
        "name":node[0].name,
        "Agglomerative (AGNES)": {"Silhouette Score": agnes_silhouette, "Adjusted Rand Index": agnes_rand_index},
        "K-Means": {"Silhouette Score": kmeans_silhouette, "Adjusted Rand Index": kmeans_rand_index},
        "Birch": {"Silhouette Score": birch_silhouette, "Adjusted Rand Index": birch_rand_index},
        "DBSCAN": {"Silhouette Score": dbscan_silhouette, "Adjusted Rand Index": dbscan_rand_index}
    }


    return Response(results, status=status.HTTP_200_OK)