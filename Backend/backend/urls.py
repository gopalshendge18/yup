from django.urls import path, include
from backend import views
from .views import *




urlpatterns = [

    # path('', views.home, name="home"),
    path('home/', views.process_csv_data),
    path('upload', views.upload_csv_view),
    path('api/', include('rest_framework.urls')),
    path('kmeans', views.k_means_clustering),
    path('birch', views.birch_clustering),
    path('dbscan', views.dbscan_clustering),
    path('agnes', views.hierarchical_clusterings),
    path('pam', views.pam_clustering),
    path('acc', views.tabulate_results_json),
    path('apriori', views.association_rules_api),
    path('rules', views.rules_api),
    path('api/chi2_analyze/', views.Chi_Analyze.as_view()),
    path('crawl', views.crawl),

    path('hits', views.hits_api),
    path('pagerank', views.pagerank_apis),



]