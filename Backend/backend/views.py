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
@csrf_exempt
@api_view(['POST', 'GET'])
  # Make sure CSRF exemption is included
def upload_csv_view(request):
    if request.method == 'GET':
        return Response({"message": "JSFSFSLKFJ"})
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_data = CSVData(csv_file=request.FILES['csv_file'])
            csv_data.save()
            request.session['csv_data_id'] = csv_data.id  # Save the CSVData ID in the session
            return Response({"message": "File has been successfully uploaded to the database."})
        else:
            return Response({"error": "Form is not valid."}, status=400)
    else:
        return Response({"error": "Only POST requests are allowed."}, status=405)

def process_csv_data(request):
    csv_data_id = 3 # Retrieve the CSVData ID from the session
    if csv_data_id is not None:
        try:
            csv_data = CSVData.objects.get(id=csv_data_id)
            # Process the CSV data here
            print(csv_data_id)
            data_str = csv_data.csv_file.read().decode('utf-8')
            
            # Example processing - split data by lines
            lines = data_str.split("\n")
            return HttpResponse(f"CSV data has been processed. Number of lines: {data_str}")
        except CSVData.DoesNotExist:
            return HttpResponse("CSV data not found.", status=404)
    else:
        return HttpResponse("No CSV data has been uploaded.", status=404)

def home(request):
    return HttpResponse("Hello, DM!")


def hierarchical_clustering(request):
    csv_data_id = request.session.get('csv_data_id')
    csv_data = CSVData.objects.get(id=csv_data_id)
    data = pd.read_csv(csv_data)
    # Perform hierarchical clustering here
    # Example code for creating a dendrogram plot
    from scipy.cluster.hierarchy import dendrogram, linkage
    Z = linkage(data, 'ward')
    plt.figure(figsize=(10, 7))
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(Z)
    # Save the plot to an image buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8').replace('\n', '')
    buf.close()
    plt.close()

    return HttpResponse(plot_data)

def custom_kmeans(X, n_clusters=4, max_iter=100):
    # Initialize centroids randomly
    centroids = X.sample(n_clusters).values

    for _ in range(max_iter):
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(X.values[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids based on the mean of the assigned points
        for i in range(n_clusters):
            centroids[i] = X[labels == i].mean(axis=0)

    return labels, centroids

def k_means_clustering(request):
    csv_data_id = 19  # Replace with the appropriate CSVData ID
    csv_data = CSVData.objects.get(id=csv_data_id)

    # Extract the file path from the model
    file_path = os.path.join(settings.MEDIA_ROOT, str(csv_data.csv_file))

    # Read the CSV data using the constructed file path
    data = pd.read_csv(file_path)

    # Perform custom k-means clustering
    X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalWidthCm']]  # Specify your features
    labels, centroids = custom_kmeans(X, n_clusters=4)

    # Plot the clusters
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)

    # Save the plot to an image buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8').replace('\n', '')
    buf.close()

    # Save the plot to the media directory
    image_path = os.path.join(settings.MEDIA_ROOT, 'kmeans_plot.png')

    plt.savefig(image_path)
    plt.close()

    media_url = os.path.join(settings.MEDIA_URL, 'kmeans_plot.png')
    return JsonResponse({'image_path': media_url})
    # return JsonResponse ({'image_path': image_path})
class BIRCHNode:
    def __init__(self, threshold, feature_values=None, children=None, data_points=None):
        self.threshold = threshold
        self.feature_values = np.asarray(feature_values) if feature_values is not None else np.zeros_like(np.asarray(threshold))
        self.children = children or []
        self.data_points = data_points or []





def custom_birch(X, threshold=3, branching_factor=50):
    root = BIRCHNode(threshold)

    for i, data_point in X.iterrows():
        insert_data_point(root, data_point, threshold, branching_factor)

    return root

def insert_data_point(node, data_point, threshold, branching_factor):
    if len(node.children) == 0:
        node.data_points.append(data_point)
        if len(node.data_points) > branching_factor:
            split_node(node, threshold, branching_factor)
    else:
        best_child = find_best_child(node, data_point)
        insert_data_point(best_child, data_point, threshold, branching_factor)

def find_best_child(node, data_point):
    min_distance = float('inf')
    best_child = None

    for child in node.children:
        distance = np.linalg.norm(data_point.values - child.feature_values)
        if distance < min_distance:
            min_distance = distance
            best_child = child

    return best_child

def split_node(node, threshold, branching_factor):
    new_children = []
    for child_data_points in np.array_split(node.data_points, branching_factor):
        new_child = BIRCHNode(threshold, data_points=child_data_points)
        update_feature_values(new_child)
        new_children.append(new_child)

    node.children = new_children
    node.data_points = []

def update_feature_values(node):
    if len(node.data_points) > 0:
        node.feature_values = np.mean([data_point.values for data_point in node.data_points], axis=0)
    else:
        for child in node.children:
            update_feature_values(child)

def traverse_tree(node, labels, cluster_index):
    if len(node.children) == 0:
        labels[node.data_points.index] = cluster_index
    else:
        for child in node.children:
            traverse_tree(child, labels, cluster_index)

def custom_birch_clustering(request):
    csv_data_id = 19  # Replace with the appropriate CSVData ID
    csv_data = CSVData.objects.get(id=csv_data_id)

    # Extract the file path from the model
    file_path = os.path.join(settings.CSV_FILES, str(csv_data.csv_file))

    # Read the CSV data using the constructed file path
    data = pd.read_csv(file_path)

    # Perform custom BIRCH clustering
    X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalWidthCm']]  # Specify your features
    birch_tree = custom_birch(X, threshold=3, branching_factor=50)

    # Assign cluster labels to data points
    labels = np.full(len(X), -1)
    for cluster_index, child in enumerate(birch_tree.children):
        traverse_tree(child, labels, cluster_index)

    # Plot the clusters
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, s=50, cmap='viridis')

    # Save the plot to an image buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8').replace('\n', '')
    buf.close()

    # Save the plot to the media directory
    image_path = os.path.join(settings.MEDIA_ROOT, 'birch_plot.png')
    plt.savefig(image_path)
    plt.close()

    media_url = os.path.join(settings.MEDIA_URL, 'birch_plot.png')
    return JsonResponse({'image_path': media_url})


def birch_clustering(request):
    csv_data_id = 19  # Replace with the appropriate CSVData ID
    csv_data = CSVData.objects.get(id=csv_data_id)

    # Extract the file path from the model

    file_path = os.path.join(settings.MEDIA_ROOT, str(csv_data.csv_file))

    # Read the CSV data using the constructed file path
    data = pd.read_csv(file_path)

    # Specify your features
    X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalWidthCm']]

    # Perform BIRCH clustering
    birch = Birch(n_clusters=4)
    labels = birch.fit_predict(X)

    # Plot the clusters
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, s=50, cmap='viridis')

    # Save the plot to an image buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8').replace('\n', '')
    buf.close()

    # Save the plot to the media directory
    image_path = os.path.join(settings.MEDIA_ROOT, 'birch_plot.png')
    plt.savefig(image_path)
    plt.close()

    media_url = os.path.join(settings.MEDIA_URL, 'birch_plot.png')
    return JsonResponse({'image_path': media_url})


def custom_dbscan(X, eps, min_samples):
    # Initialize labels as unvisited (-1)
    labels = np.full(X.shape[0], -1)

    # Initialize cluster label
    cluster_label = 0

    for i in range(X.shape[0]):
        if labels[i] != -1:
            continue

        # Find neighbors
        neighbors = find_neighbors(X, i, eps)

        if len(neighbors) < min_samples:
            # Assign as noise
            labels[i] = 0
        else:
            # Expand cluster
            cluster_label += 1
            labels[i] = cluster_label
            expand_cluster(X, labels, i, neighbors, cluster_label, eps, min_samples)

    return labels

def find_neighbors(X, index, eps):
    # Calculate Euclidean distances
    distances = np.linalg.norm(X - X[index], axis=1)
    # Return indices of points within eps distance
    return np.where(distances <= eps)[0]

def expand_cluster(X, labels, index, neighbors, cluster_label, eps, min_samples):
    for neighbor in neighbors:
        if labels[neighbor] == -1:
            labels[neighbor] = cluster_label
            new_neighbors = find_neighbors(X, neighbor, eps)

            if len(new_neighbors) >= min_samples:
                neighbors = np.union1d(neighbors, new_neighbors)

def dbscan_clustering(request):
    csv_data_id = 19  # Replace with the appropriate CSVData ID
    csv_data = CSVData.objects.get(id=csv_data_id)

    # Extract the file path from the model
    file_path = os.path.join(settings.MEDIA_ROOT, str(csv_data.csv_file))

    # Read the CSV data using the constructed file path
    data = pd.read_csv(file_path)

    # Specify your features
    X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalWidthCm']].values

    # Perform DBSCAN clustering
    eps = 0.5
    min_samples = 5
    labels = custom_dbscan(X, eps, min_samples)

    # Plot the clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

    # Save the plot to an image buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8').replace('\n', '')
    buf.close()

    # Save the plot to the media directory
    image_path = os.path.join(settings.MEDIA_ROOT, 'dbscan_plot.png')
    plt.savefig(image_path)
    plt.close()

    media_url = os.path.join(settings.MEDIA_URL, 'dbscan_plot.png')
    return JsonResponse({'image_path': media_url})

def custom_dbscan(X, eps, min_samples):
    # Initialize labels as unvisited (-1)
    labels = np.full(X.shape[0], -1)

    # Initialize cluster label
    cluster_label = 0

    for i in range(X.shape[0]):
        if labels[i] != -1:
            continue

        # Find neighbors
        neighbors = find_neighbors(X, i, eps)

        if len(neighbors) < min_samples:
            # Assign as noise
            labels[i] = 0
        else:
            # Expand cluster
            cluster_label += 1
            labels[i] = cluster_label
            expand_cluster(X, labels, i, neighbors, cluster_label, eps, min_samples)

    return labels

def find_neighbors(X, index, eps):
    # Calculate Euclidean distances
    distances = np.linalg.norm(X - X[index], axis=1)
    # Return indices of points within eps distance
    return np.where(distances <= eps)[0]

def expand_cluster(X, labels, index, neighbors, cluster_label, eps, min_samples):
    for neighbor in neighbors:
        if labels[neighbor] == -1:
            labels[neighbor] = cluster_label
            new_neighbors = find_neighbors(X, neighbor, eps)

            if len(new_neighbors) >= min_samples:
                neighbors = np.union1d(neighbors, new_neighbors)




def agnes_clustering(request):
    csv_data_id = 25  # Replace with the appropriate CSVData ID
    csv_data = CSVData.objects.get(id=csv_data_id)

    # Extract the file path from the model
    file_path = os.path.join(settings.MEDIA_URL, str(csv_data.csv_file))

    # Read the CSV data using the constructed file path
    data = pd.read_csv(file_path)

    # Specify your features
    X = data[['dAge', 'dAncstry1', 'iCitizen']].values

    # Perform AGNES clustering
    Z = agnes(X)

    # Plot dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title('AGNES Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Distance')
    
    # Save the plot to an image buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8').replace('\n', '')
    buf.close()

    # Save the plot to the media directory
    image_path = os.path.join(settings.MEDIA_ROOT, 'agnes_dendrogram.png')
    plt.savefig(image_path)
 
    plt.close()
    return HttpResponse(plot_data)

def agnes(X):
    # Initialize clusters as individual data points
    clusters = [[i] for i in range(X.shape[0])]

    # Calculate pairwise distances
    distances = np.linalg.norm(X[:, np.newaxis, :] - X, axis=2)

    # Perform agglomerative clustering
    while len(clusters) > 1:
        min_distance = float('inf')
        merge_indices = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = calculate_distance(clusters[i], clusters[j], distances)
                if distance < min_distance:
                    min_distance = distance
                    merge_indices = (i, j)

        # Merge clusters
        clusters[merge_indices[0]] += clusters[merge_indices[1]]
        del clusters[merge_indices[1]]

    # Generate linkage matrix
    Z = generate_linkage_matrix(clusters, distances)

    return Z

def calculate_distance(cluster1, cluster2, distances):
    # Calculate distance between two clusters (single-linkage)
    min_distance = float('inf')
    for i in cluster1:
        for j in cluster2:
            if distances[i, j] < min_distance:
                min_distance = distances[i, j]
    return min_distance

def generate_linkage_matrix(clusters, distances):
    # Generate linkage matrix from final clusters
    Z = []
    cluster_mapping = {point: idx for idx, cluster in enumerate(clusters) for point in cluster}

    for i in range(len(clusters) - 1):
        cluster1 = clusters[i]
        cluster2 = clusters[i + 1]

        min_distance = float('inf')
        for i in cluster1:
            for j in cluster2:
                if distances[i, j] < min_distance:
                    min_distance = distances[i, j]

        Z.append([cluster_mapping[cluster1[0]], cluster_mapping[cluster2[0]], min_distance, len(cluster1) + len(cluster2)])

    return np.array(Z)

def pam(X, k, max_iter=100):
    # Initialize medoids randomly
    medoids = np.random.choice(X.shape[0], k, replace=False)
    prev_medoids = np.copy(medoids)

    for _ in range(max_iter):
        # Assign each point to the nearest medoid
        labels = np.argmin(np.linalg.norm(X - X[medoids][:, np.newaxis, :], axis=2), axis=0)

        # Update medoids
        for i in range(k):
            cluster_points = X[labels == i]
            distances = np.sum(np.abs(cluster_points[:, np.newaxis, :] - cluster_points), axis=2)
            new_medoid_index = np.argmin(np.sum(distances, axis=1))
            medoids[i] = np.where(labels == i)[0][new_medoid_index]

        # Check for convergence
        if np.array_equal(prev_medoids, medoids):
            break

        prev_medoids = np.copy(medoids)

    return labels, medoids

def pam_clustering(request):
    csv_data_id = 19  # Replace with the appropriate CSVData ID
    csv_data = CSVData.objects.get(id=csv_data_id)

    # Extract the file path from the model
    file_path = os.path.join(settings.MEDIA_ROOT, str(csv_data.csv_file))

    # Read the CSV data using the constructed file path
    data = pd.read_csv(file_path)

    # Specify your features
    X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalWidthCm']].values

    # Specify the number of clusters (k)
    k = 3

    # Perform k-Medoids clustering (PAM)
    labels, medoids = pam(X, k)

    # Plot the clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

    # Highlight medoids
    plt.scatter(X[medoids, 0], X[medoids, 1], c='red', marker='X', s=200, label='Medoids')

    plt.legend()
    
    # Save the plot to an image buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8').replace('\n', '')
    buf.close()

    # Save the plot to the media directory
    image_path = os.path.join(settings.MEDIA_ROOT, 'pam_plot.png')
    plt.savefig(image_path)
    plt.close()

    media_url = os.path.join(settings.MEDIA_URL, 'pam_plot.png')
    return JsonResponse({'image_path': media_url})

from scipy.cluster.hierarchy import linkage, dendrogram
from django.http import JsonResponse
from io import BytesIO
import base64
import os
import matplotlib.pyplot as plt
import pandas as pd
from django.conf import settings
from .models import CSVData

def agness_clustering(request):
    csv_data_id = 25  # Replace with the appropriate CSVData ID
    csv_data = CSVData.objects.get(id=csv_data_id)

    # Extract the file path from the model
    file_path = os.path.join(settings.MEDIA_ROOT, str(csv_data.csv_file))

    # Read the CSV data using the constructed file path
    data = pd.read_csv(file_path)

    # Specify your features
    X = data[['dAge', 'iCitizen', 'iEnglish']].values

    # Perform AGNES clustering
    agnes_labels = perform_agnes(X)

    # Plot dendrogram
    dendrogram_plot = plot_dendrogram(X)

    # Save the plot to an image buffer
    buf = BytesIO()
    dendrogram_plot.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8').replace('\n', '')
    buf.close()

    # Save the plot to the media directory
    image_path = os.path.join(settings.MEDIA_ROOT, 'agnes_plot.png')
    dendrogram_plot.savefig(image_path)
    plt.close()  # Close the plot to avoid rendering issues

    media_url = os.path.join(settings.MEDIA_URL, 'agnes_plot.png')
    return JsonResponse({'image_path': media_url})

def perform_agnes(X):
    # Perform AGNES clustering
    agnes = linkage(X, method='ward')
    # You can extract cluster labels using a method like fcluster
    # agnes_labels = fcluster(agnes, t=3, criterion='maxclust')
    # For demonstration purposes, assuming you have a clustering function
    agnes_labels = [0] * len(X)
    return agnes_labels

def plot_dendrogram(X):
    # Plot dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linkage(X, method='ward'))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.close()

    return plt
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, Birch, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.impute import SimpleImputer
def calculate_ari(true_labels, predicted_labels):
    return adjusted_rand_score(true_labels, predicted_labels)

# Function to tabulate results and return JSON
def tabulate_results_json(request):
    # Load your data (replace 'your_data.csv' with the actual file path)
    csv_data_id =20 # Replace with the appropriate CSVData ID
    csv_data = CSVData.objects.get(id=csv_data_id)

    # Extract the file path from the model
    file_path = os.path.join(settings.MEDIA_ROOT, str(csv_data.csv_file))

    # Read the CSV data using the constructed file path
    data = pd.read_csv(file_path)
    # Read the CSV data using the constructed file path
   

    # Impute missing values with mean (you can choose another strategy)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(data.drop('diagnosis', axis=1))

    # Feature scaling
    X = StandardScaler().fit_transform(X_imputed)

    # True labels
    true_labels = data['diagnosis'].map({'M': 1, 'B': 0})

    # 1. KMeans
    km = KMeans(n_clusters=2, init="k-means++", n_init=10)
    km_pred = km.fit_predict(X)
    km_ari = calculate_ari(true_labels, km_pred)

    # 2. Birch
    birch = Birch(n_clusters=2)
    birch_pred = birch.fit_predict(X)
    birch_ari = calculate_ari(true_labels, birch_pred)

    # 3. DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_pred = dbscan.fit_predict(X)
    dbscan_ari = calculate_ari(true_labels, dbscan_pred)

    # 4. AGNES (Agglomerative Clustering)
    agnes = AgglomerativeClustering(n_clusters=2)
    agnes_pred = agnes.fit_predict(X)
    agnes_ari = calculate_ari(true_labels, agnes_pred)

    # Tabulate the results to JSON
    algorithms = ['KMeans', 'Birch', 'DBSCAN', 'AGNES']
    ari_scores = [km_ari, birch_ari, dbscan_ari, agnes_ari]
    results_table = pd.DataFrame({'Algorithm': algorithms, 'ARI Score': ari_scores})
    results_json = results_table.to_json(orient='records')

    return JsonResponse({'results': results_json})

# Print or return the JSON
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
# from helpers.helpers import check_df, crm_data_prep

  # Ensure that the helpers module is available
@csrf_exempt
@api_view(['POST', 'GET'])
def association_rules_api(request):
    try:
        # Get data from the request body
        csv_data_id1 =22 # Replace with the appropriate CSVData ID
        csv_data_id2 =23
        csv_data_id3 =24  
        csv_data1 = CSVData.objects.get(id=csv_data_id1)
        csv_data2 = CSVData.objects.get(id=csv_data_id2)
        csv_data3 = CSVData.objects.get(id=csv_data_id3)
    # Extract the file path from the model
        file_path1 = os.path.join(settings.MEDIA_ROOT, str(csv_data1.csv_file))
        file_path2 = os.path.join(settings.MEDIA_ROOT, str(csv_data2.csv_file))
        file_path3 = os.path.join(settings.MEDIA_ROOT, str(csv_data3.csv_file))
    # Read the CSV data using the constructed file path
        
        data = json.loads(request.body.decode('utf-8'))
        confidence = float(data['confidence'])
        support = float(data['support'])

        # ... rest of your code (from loading data to generating rules)
        sparseVector = pd.read_csv(file_path1, header=None, names=["Receipt_No.", "Food_1", "Food_2", "Food_3", "Food_4", "Food_5", "Food_6", "Food_7", "Food_8"])
        fullBinaryVector = pd.read_csv(file_path2, header=None)
        itemTable = pd.read_csv(file_path3, header=None, names=["Receipt_No.", "Quantity", "Food_No."])

        # Create lookup table
        foodNames = ["Ladoo", "Chakali", "Karanji", "shankarpali", "Besan Ladoo", "Kaju katli", "Chivada", "Kalyche ladoo", "Anarase", "Milk cake", "Almond Tart", "Apple Pie", "Apple Tart", "Apricot Tart", "Berry Tart", "Blackberry Tart", "Blueberry Tart", "Chocolate Tart", "Cherry Tart", "Lemon Tart", "Pecan Tart", "Ganache Cookie", "Gongolais Cookie", "Raspberry Cookie", "Lemon Cookie", "Chocolate Meringue", "Vanilla Meringue", "Marzipan Cookie", "Tulie Cookie", "Walnut Cookie", "Almond Croissant", "Apple Croissant", "Apricot Croissant", "Cheese Croissant", "Chocolate Croissant", "Apricot Danish", "Apple Danish", "Almond Twist", "Almond Bear Claw", "Blueberry Danish", "Lemon Lemonade", "Raspberry Lemonade", "Orange Juice", "Green Tea", "Bottled Water", "Hot Coffee", "Chocolate Coffee", "Vanilla Frappuccino", "Cherry Soda", "Single Espresso"]
        foodID = list(range(50))
        lookUpTable = pd.DataFrame({"foodID": foodID, "foodNames": foodNames})

        # Rename some columns
        itemTable.columns = ["Receipt_No.", "Quantity", "Food_No."]
        fullBinaryVector.columns = ["Receipt_No."] + foodNames

        # To make sure empty value is NA
        sparseVector = sparseVector.applymap(lambda x: pd.to_numeric(x, errors='coerce'))

        # Drop rows with missing values in Receipt_No. and Food_No.
        itemTable = itemTable.dropna(subset=["Receipt_No.", "Food_No."])

        # Reset index before creating a transaction table
        itemTable.reset_index(drop=True, inplace=True)

        # Create a transaction table
        transTable = itemTable.groupby("Receipt_No.")["Food_No."].apply(list).reset_index(name='Food')
        print(transTable)
        # Ensure all arrays have the same length
        max_len = transTable['Food'].apply(len).max()
        transTable['Food'] = transTable['Food'].apply(lambda x: x + [None] * (max_len - len(x)) if len(x) < max_len else x[:max_len])

        # Ensure 'Food' column contains only valid lists
        transTable = transTable[transTable['Food'].apply(lambda x: isinstance(x, list))]

        # Create a binary matrix with the correct number of columns
        binary_matrix = pd.DataFrame(transTable['Food'].tolist(), columns=foodNames[:max_len]).fillna(0).astype(int)
        print(binary_matrix)
        # Ensure all values in the binary matrix are either 0 or 1
        binary_matrix = binary_matrix.applymap(lambda x: 1 if x != 0 else 0)
        print(binary_matrix)

        # Run Apriori Algo
        frequent_itemsets = apriori(binary_matrix, min_support= support, use_colnames=True)

        # Print the frequent itemsets to check if any were found
        print("Frequent Itemsets:")
        print(frequent_itemsets)

        # Check the shape of the frequent_itemsets DataFrame
        print("Shape of frequent_itemsets DataFrame:", frequent_itemsets.shape)

        # Check if any frequent itemsets were found
        if frequent_itemsets.empty:
            print("No frequent itemsets found. Adjust min_support or check your data.")
        else:
            # Generate association rules
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)

            # Print the rules
            print("Association Rules:")
            print(rules)

            # Plotting the graph
            

            # Convert rules DataFrame to JSON
            rules_json = rules.to_json(orient='records')

            return JsonResponse({'rules': rules_json})

    except Exception as e:
        return JsonResponse({'error': str(e)})
def chi_square(observed, expected):
    observed = [
        [row['support'], row['consequent support'] - row['support']],
        [row['antecedent support'] - row['support'], 1 - (row['antecedent support'] + row['consequent support'] - row['support'])]
    ]
    _, p_value, _, _ = chi2_contingency(observed)
    return p_value

def all_confidence(rule):
    return min(rule['confidence'], 1 - rule['confidence'])

def max_confidence(rule):
    return max(rule['confidence'], 1 - rule['confidence'])

def kulczynski(rule):
    return 0.5 * (rule['confidence'] + (1 - rule['confidence']))

def cosine(rule):
    return rule['confidence'] / math.sqrt(rule['antecedent support'] * rule['consequent support'])

@csrf_exempt
@api_view(['POST', 'GET'])
def rules_api(request):
    try:
        # Get data from the request body
        csv_data_id1 =22 # Replace with the appropriate CSVData ID
        csv_data_id2 =23
        csv_data_id3 =24  
        csv_data1 = CSVData.objects.get(id=csv_data_id1)
        csv_data2 = CSVData.objects.get(id=csv_data_id2)
        csv_data3 = CSVData.objects.get(id=csv_data_id3)
    # Extract the file path from the model
        file_path1 = os.path.join(settings.MEDIA_ROOT, str(csv_data1.csv_file))
        file_path2 = os.path.join(settings.MEDIA_ROOT, str(csv_data2.csv_file))
        file_path3 = os.path.join(settings.MEDIA_ROOT, str(csv_data3.csv_file))
    # Read the CSV data using the constructed file path
        
        data = json.loads(request.body.decode('utf-8'))
        confidence = float(data['confidence'])
        support = float(data['support'])
        print(data)
        # ... rest of your code (from loading data to generating rules)
        sparseVector = pd.read_csv(file_path1, header=None, names=["Receipt_No.", "Food_1", "Food_2", "Food_3", "Food_4", "Food_5", "Food_6", "Food_7", "Food_8"])
        fullBinaryVector = pd.read_csv(file_path2, header=None)
        itemTable = pd.read_csv(file_path3, header=None, names=["Receipt_No.", "Quantity", "Food_No."])

        # Create lookup table
        foodNames = ["Ladoo", "Chakali", "Karanji", "shankarpali", "Besan Ladoo", "Kaju katli", "Chivada", "Kalyche ladoo", "Anarase", "Milk cake", "Almond Tart", "Apple Pie", "Apple Tart", "Apricot Tart", "Berry Tart", "Blackberry Tart", "Blueberry Tart", "Chocolate Tart", "Cherry Tart", "Lemon Tart", "Pecan Tart", "Ganache Cookie", "Gongolais Cookie", "Raspberry Cookie", "Lemon Cookie", "Chocolate Meringue", "Vanilla Meringue", "Marzipan Cookie", "Tulie Cookie", "Walnut Cookie", "Almond Croissant", "Apple Croissant", "Apricot Croissant", "Cheese Croissant", "Chocolate Croissant", "Apricot Danish", "Apple Danish", "Almond Twist", "Almond Bear Claw", "Blueberry Danish", "Lemon Lemonade", "Raspberry Lemonade", "Orange Juice", "Green Tea", "Bottled Water", "Hot Coffee", "Chocolate Coffee", "Vanilla Frappuccino", "Cherry Soda", "Single Espresso"]
        foodID = list(range(50))
        lookUpTable = pd.DataFrame({"foodID": foodID, "foodNames": foodNames})

        # Rename some columns
        itemTable.columns = ["Receipt_No.", "Quantity", "Food_No."]
        fullBinaryVector.columns = ["Receipt_No."] + foodNames

        # To make sure empty value is NA
        sparseVector = sparseVector.applymap(lambda x: pd.to_numeric(x, errors='coerce'))

        # Drop rows with missing values in Receipt_No. and Food_No.
        itemTable = itemTable.dropna(subset=["Receipt_No.", "Food_No."])

        # Reset index before creating a transaction table
        itemTable.reset_index(drop=True, inplace=True)

        # Create a transaction table
        transTable = itemTable.groupby("Receipt_No.")["Food_No."].apply(list).reset_index(name='Food')
        # print(transTable)
        # Ensure all arrays have the same length
        max_len = transTable['Food'].apply(len).max()
        transTable['Food'] = transTable['Food'].apply(lambda x: x + [None] * (max_len - len(x)) if len(x) < max_len else x[:max_len])

        # Ensure 'Food' column contains only valid lists
        transTable = transTable[transTable['Food'].apply(lambda x: isinstance(x, list))]

        # Create a binary matrix with the correct number of columns
        binary_matrix = pd.DataFrame(transTable['Food'].tolist(), columns=foodNames[:max_len]).fillna(0).astype(int)
        
        # Ensure all values in the binary matrix are either 0 or 1
        binary_matrix = binary_matrix.applymap(lambda x: 1 if x != 0 else 0)
        

        # Run Apriori Algo
        frequent_itemsets = apriori(binary_matrix, min_support= support, use_colnames=True)

        # Print the frequent itemsets to check if any were found
        

        # Check the shape of the frequent_itemsets DataFrame
        

        # Check if any frequent itemsets were found
        if frequent_itemsets.empty:
            print("No frequent itemsets found. Adjust min_support or check your data.")
        else:
            # Generate association rules
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)

            # Print the rules
    
            rules_lift = rules[rules['lift'] > 1]
            print(rules_lift)
            rules['cosine'] = rules.apply(lambda row: cosine(row), axis=1)

# Adjust the threshold value based on your analysis
            threshold_cosine = 0.4

# Filter interesting rules based on the cosine measure
            interesting_cosine_rules = rules[rules['cosine'] > threshold_cosine]
            rules['kulczynski'] = rules.apply(lambda row: kulczynski(row), axis=1)

# Adjust the threshold value based on your analysis
            threshold_kulczynski = 0.4
            rules['all_confidence'] = rules.apply(lambda row: all_confidence(row), axis=1)

# Adjust the threshold value based on your analysis
            threshold_all_confidence = 0.4

# Filter interesting rules based on the All-Confidence measure
            interesting_all_confidence_rules = rules[rules['all_confidence'] > threshold_all_confidence]
            rules['max_confidence'] = rules.apply(lambda row: max_confidence(row), axis=1)

# Adjust the threshold value based on your analysis
            threshold_max_confidence = 0.4

# Filter interesting rules based on the Max-Confidence measure
            interesting_max_confidence_rules = rules[rules['max_confidence'] > threshold_max_confidence]
            # rules['chi2'] = rules.apply(chi_square, axis=1)

# Decide a threshold for Chi-Square Test
            threshold_chi2 = 0.05  # You can adjust this threshold based on your significance level

# Filter interesting rules based on Chi-Square Test
            # interesting_chi2_rules = rules[rules['chi2'] < threshold_chi2]
# Convert the filtered rules DataFrame to JSON
        interesting_max_confidence_rules_json = interesting_max_confidence_rules.to_json(orient='records')

# Convert the filtered rules DataFrame to JSON
        interesting_all_confidence_rules_json = interesting_all_confidence_rules.to_json(orient='records')
# Filter interesting rules based on the Kulczynski measure
        interesting_kulczynski_rules = rules[rules['kulczynski'] > threshold_kulczynski]
# Convert the filtered rules DataFrame to JSON
        interesting_cosine_rules_json = interesting_cosine_rules.to_json(orient='records')

        interesting_kulczynski_rules_json = interesting_kulczynski_rules.to_json(orient='records')

        # Convert the filtered rules DataFrames to JSON
        rules_lift_json = rules_lift.to_json(orient='records')
       

# Convert the filtered rules DataFrame to JSON
        # interesting_chi2_rules_json = interesting_chi2_rules.to_json(orient='records')

        return JsonResponse({
            'rules_lift': rules_lift_json,
            'rules_cosine': interesting_cosine_rules_json,
            'rules_kulczynski': interesting_kulczynski_rules_json,
            'rules_all_confidence': interesting_all_confidence_rules_json,
            'rules_max_confidence': interesting_max_confidence_rules_json,
            
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)})

import math
from rest_framework.parsers import FileUploadParser
from rest_framework.views import APIView
from rest_framework import status
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

import pandas as pd
import numpy as np
from django.http import JsonResponse
from scipy.stats import chi2_contingency
from scipy.stats import chi2

class Chi_Analyze(APIView):
    @method_decorator(csrf_exempt)
    def post(self, request, *args, **kwargs):
        if request.method == 'POST':
            try:
                file = request.FILES.get('file')
                df = pd.read_csv(file)

                attribute1 = request.POST['attribute1']
                attribute2 = request.POST['attribute2']

              

                contingency_table = self.create_contingency_table(df, attribute1, attribute2)

                print(type(contingency_table))
            

                chi_square, p, dof, expected = self.calculate_chi_square(contingency_table)

                result = {
                    'contingency_table': contingency_table,
                    'chi2_value': chi_square,
                    'p_value': p,
                    'correlation_result': 'Correlated' if p < 0.05 else 'Not Correlated'
                }

                return JsonResponse(result)
            except Exception as e :
                return JsonResponse({"error => ": str(e)}, status=status.HTTP_200_OK)

    def create_contingency_table(self, data, attribute1, attribute2):
        unique_values_attr1 = set(data[attribute1])
        unique_values_attr2 = set(data[attribute2])
        
        contingency_table = {}
        
        for value_attr1 in unique_values_attr1:
            contingency_table[value_attr1] = {}
            for value_attr2 in unique_values_attr2:
                contingency_table[value_attr1][value_attr2] = 0
        
        for index, row in data.iterrows():
            contingency_table[row[attribute1]][row[attribute2]] += 1
        
        return contingency_table

    def calculate_chi_square(self, contingency_table):
        rows = len(contingency_table)
        cols = len(contingency_table[next(iter(contingency_table))])

        total_observed = 0
        row_sums = [0] * rows
        col_sums = [0] * cols

        for i, row_key in enumerate(contingency_table):
            for j, col_key in enumerate(contingency_table[row_key]):
                frequency = contingency_table[row_key][col_key]
                total_observed += frequency
                row_sums[i] += frequency
                col_sums[j] += frequency

        expected_values = []
        for i in range(rows):
            row_expected = []
            for j in range(cols):
                expected = (row_sums[i] * col_sums[j]) / total_observed
                row_expected.append(expected)
            expected_values.append(row_expected)

        chi_square = 0
        for i, row_key in enumerate(contingency_table):
            for j, col_key in enumerate(contingency_table[row_key]):
                observed = contingency_table[row_key][col_key]
                expected = expected_values[i][j]
                chi_square += ((observed - expected) ** 2) / expected

        dof = (rows - 1) * (cols - 1)

        p_value = 1 - chi2.cdf(chi_square, dof)
        
        return chi_square, p_value, dof, expected_values


from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .helpers import is_url_valid, get_clean_url, is_link_internal
from bs4 import BeautifulSoup, SoupStrainer
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from ordered_set import OrderedSet

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from bs4 import BeautifulSoup, SoupStrainer
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from ordered_set import OrderedSet

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'
}


# def crawl(request):
#     if request.method == 'POST':
#         data = request.POST
#         seed_url = data.get('url')
#         print(data)
#         depth = int(data.get('depth', 5))  # Default depth is 25

#         if not seed_url:
#             return JsonResponse({'error': 'Please provide a valid URL'})

#         crawled_urls = crawl_url(seed_url, depth)
#         return JsonResponse({'links': list(crawled_urls)})

#     return JsonResponse({'error': 'Invalid request method'})

# def crawl_url(url, depth):
#     crawled_urls = OrderedSet([])
#     crawl_recursive(url, crawled_urls, depth)
#     return crawled_urls

# def crawl_recursive(url, crawled_urls, depth):
#     found_urls = []

#     try:
#         # Create a Request object with headers
#         req = Request(url)
#         page = urlopen(req)
#         content = page.read()
#         soup = BeautifulSoup(content, 'lxml', parse_only=SoupStrainer('a'))

#         for anchor in soup.find_all('a'):
#             link = anchor.get('href')
#             if is_url_valid(link):
#                 link = get_clean_url(url, link)
#                 if is_link_internal(link, url):
#                     found_urls.append(link)

#     except HTTPError as e:
#         print('HTTPError:' + str(e.code) + ' in ', url)
#     except URLError as e:
#         print('URLError: ' + str(e.reason) + ' in ', url)
#     except Exception:
#         import traceback
#         print('Generic exception: ' + traceback.format_exc() + ' in ', url)

#     cleaned_found_urls = set(found_urls)
#     crawled_urls |= cleaned_found_urls

#     if len(crawled_urls) > depth:
#         crawled_urls = crawled_urls[:depth]
#         return
#     else:
#         for found_url in cleaned_found_urls:
#             crawl_recursive(found_url, crawled_urls, depth)
from collections import deque
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from bs4 import BeautifulSoup, SoupStrainer
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from ordered_set import OrderedSet
@csrf_exempt
def crawl(request):
    if request.method == 'POST':
        data = request.POST
        seed_url = data.get('url')
        print(data)
        depth = int(data.get('depth', 5))  # Default depth is 25

        if not seed_url:
            return JsonResponse({'error': 'Please provide a valid URL'})

        dfs_crawled_urls = crawl_url(seed_url, depth, dfs=True)
        bfs_crawled_urls = crawl_url(seed_url, depth, dfs=False)

        return JsonResponse({'dfs_links': list(dfs_crawled_urls), 'bfs_links': list(bfs_crawled_urls)})

    return JsonResponse({'error': 'Invalid request method'})

def crawl_url(url, depth, dfs=True):
    crawled_urls = OrderedSet([])
    if dfs:
        crawl_recursive_dfs(url, crawled_urls, depth)
    else:
        crawl_recursive_bfs(url, crawled_urls, depth)
    return crawled_urls

def crawl_recursive_dfs(url, crawled_urls, depth):
    found_urls = []
    try:
        req = Request(url)
        page = urlopen(req)
        content = page.read()
        soup = BeautifulSoup(content, 'lxml', parse_only=SoupStrainer('a'))

        for anchor in soup.find_all('a'):
            link = anchor.get('href')
            if is_url_valid(link):
                link = get_clean_url(url, link)
                if is_link_internal(link, url):
                    found_urls.append(link)

    except HTTPError as e:
        print('HTTPError:' + str(e.code) + ' in ', url)
    except URLError as e:
        print('URLError: ' + str(e.reason) + ' in ', url)
    except Exception:
        import traceback
        print('Generic exception: ' + traceback.format_exc() + ' in ', url)

    cleaned_found_urls = set(found_urls)
    crawled_urls |= cleaned_found_urls

    if len(crawled_urls) > depth:
        crawled_urls = crawled_urls[:depth]
        return

    for found_url in cleaned_found_urls:
        crawl_recursive_dfs(found_url, crawled_urls, depth)

def crawl_recursive_bfs(seed_url, crawled_urls, depth):
    queue = deque([(seed_url, 0)])

    while queue:
        current_url, current_depth = queue.popleft()

        if current_depth > depth:
            break

        if current_url not in crawled_urls:
            try:
                req = Request(current_url)
                page = urlopen(req)
                content = page.read()
                soup = BeautifulSoup(content, 'lxml', parse_only=SoupStrainer('a'))

                found_urls = [
                    get_clean_url(current_url, anchor.get('href'))
                    for anchor in soup.find_all('a') if is_url_valid(anchor.get('href'))
                ]

                internal_urls = [
                    link for link in found_urls if is_link_internal(link, current_url)
                ]

                crawled_urls.add(current_url)
                crawled_urls.update(internal_urls)

                queue.extend((url, current_depth + 1) for url in internal_urls)

            except HTTPError as e:
                print('HTTPError:' + str(e.code) + ' in ', current_url)
            except URLError as e:
                print('URLError: ' + str(e.reason) + ' in ', current_url)
            except Exception:
                import traceback
                print('Generic exception: ' + traceback.format_exc() + ' in ', current_url)

    return crawled_urls

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import networkx as nx

# Function to read Google web graph and calculate HITS
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import networkx as nx

# Function to read Google web graph and calculate HITS
def calculate_hits(file_path):
    G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int)
    hits_result = nx.hits(G)
    return hits_result

@csrf_exempt
def hits_api(request):
    if request.method == 'GET':
        # Specify the path to the downloaded file
        file_path = 'C:/Users/Rahul -hp/Desktop/seventh sem/data mining/DM assigenments/dm_assignments/media/csv_files/web-NotreDame.txt'

        # Calculate HITS
        hits_data = calculate_hits(file_path)

        # Get the top 10 hub and authority scores
        top_hub_data = pd.DataFrame(list(hits_data[0].items()), columns=['Page', 'Hub']).nlargest(10, 'Hub').to_dict(orient='records')
        top_authority_data = pd.DataFrame(list(hits_data[1].items()), columns=['Page', 'Authority']).nlargest(10, 'Authority').to_dict(orient='records')

        response_data = {
            'top_hub_scores': top_hub_data,
            'top_authority_scores': top_authority_data
        }

        return JsonResponse(response_data)

    return JsonResponse({'error': 'Invalid request method'})



# views.py
import pandas as pd
import numpy as np
from django.http import JsonResponse

def pagerank_apis(request):
    if request.method == 'GET':
        # Specify the path to the downloaded file
        file_path = 'C:/Users/Rahul -hp/Desktop/seventh sem/data mining/DM assigenments/dm_assignments/media/csv_files/web-NotreDame.txt'

        # Calculate PageRank
        pagerank_data = calculate_pagerank(file_path)

        # Prepare the data for JSON response
        pagerank_data = pd.DataFrame(list(pagerank_data.items()), columns=['Page', 'Rank']).to_dict(orient='records')

        response_data = {
            'pagerank_scores': pagerank_data[:10]  # Return the top 10 pages
        }

        return JsonResponse(response_data)

    return JsonResponse({'error': 'Invalid request method'})

def calculate_pagerank(file_path, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
    # Read the edge list from the file
    edges = pd.read_csv(file_path, delimiter='\t', comment='#', header=None, names=['FromNode', 'ToNode'])

    # Create a dictionary to store the outgoing links for each node
    outgoing_links = {node: set() for node in set(edges['FromNode'])}

    # Populate the outgoing_links dictionary
    for _, row in edges.iterrows():
        outgoing_links[row['FromNode']].add(row['ToNode'])

    # Initialize PageRank scores
    num_nodes = len(outgoing_links)
    pagerank = {node: 1 / num_nodes for node in outgoing_links}

    # Iterative PageRank calculation
    for iteration in range(max_iterations):
        new_pagerank = {node: (1 - damping_factor) / num_nodes for node in outgoing_links}

        for node in outgoing_links:
            for incoming_node in outgoing_links:
                if node in outgoing_links[incoming_node]:
                    new_pagerank[node] += damping_factor * (pagerank[incoming_node] / len(outgoing_links[incoming_node]))

        # Check for convergence
        if all(abs(new_pagerank[node] - pagerank[node]) < tolerance for node in outgoing_links):
            break

        pagerank = new_pagerank

    return pagerank

def pagerank_api(request):
    if request.method == 'GET':
        # Specify the path to the downloaded file
        file_path = 'C:/Users/Rahul -hp/Desktop/seventh sem/data mining/DM assigenments/dm_assignments/media/csv_files/web-NotreDame.txt'


        # Calculate PageRank
        pagerank_data = calculate_pagerank(file_path)

        # Prepare the data for JSON response
        pagerank_data = pd.DataFrame(list(pagerank_data.items()), columns=['Page', 'Rank']).to_dict(orient='records')

        response_data = {
            'pagerank_scores': pagerank_data[:10]  # Return the top 10 pages
        }

        return JsonResponse(response_data)

    return JsonResponse({'error': 'Invalid request method'})

def calculate_pagerank(file_path):
    G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int)
    pagerank_result = nx.pagerank(G)

    return pagerank_result

from django.http import JsonResponse
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def plot_dendrograms(Z):
    plt.figure(figsize=(10, 7))
    dendrogram(Z, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')

    # Save dendrogram plot to an image buffer
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8').replace('\n', '')
    buf.close()

    # Save the plot to the media directory
    image_path = os.path.join(settings.MEDIA_ROOT, 'agnes_plot.png')
    plt.savefig(image_path)
    plt.close()

    media_url = os.path.join(settings.MEDIA_URL, 'agnes_plot.png')
    return media_url

def agnes_clusterings(data):
    clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = clustering.fit_predict(data)

    # Generate dendrogram
    linked = linkage(data, 'ward')
    dendrogram_path = plot_dendrograms(linked)

    return labels, dendrogram_path

@api_view(['GET'])
def hierarchical_clusterings(request):
    # Sample data (replace this with your dataset)
    csv_data_id = 19  # Replace with the appropriate CSVData ID
    csv_data = CSVData.objects.get(id=csv_data_id)

    # Extract the file path from the model
    file_path = os.path.join(settings.MEDIA_ROOT, str(csv_data.csv_file))

    iris_data = pd.read_csv(file_path)

    # Extract features (assuming the first 4 columns are the features)
    data = iris_data.iloc[:, :4].values

    
    labels, dendrogram_path = agnes_clusterings(data)

       
    return JsonResponse({'image_path': dendrogram_path}, status=status.HTTP_200_OK)