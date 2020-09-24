# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#This is coming from "dont_edit_this_file"!- only for Python and LaTex connection
import dont_edit_this_file
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#Put your Machine learining algorithms or any other Mathematical Manipulations here
#Example for Creating  tables
import pandas as pd
product_data=pd.read_excel('./test_data.xls', index_col='Index')
categories=product_data['Category'].unique()
product_list = {k:product_data[product_data['Category']==k].T.to_dict() for k in categories}
#Example for Creating a figure by a simple KNN cluserting (unsupervised learning methods)
# K-Means Clustering
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values#the first column is not needed+for simplicity we only consider annual income and spending score (2 features to plot in 2-D)
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):#To construct WCSS 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)#'k-means++' is used to prevent from initialization trap
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('figure_1.pdf')#Make sure your figure name is same as in tex file
plt.show()
# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)#creating dependent variables (cluster for each feature) 
# Visualising the clusters 
#feature 1 Vs feature 2
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='yellow',label='Cluster 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='green',label='Cluster 2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='blue',label='Cluster 3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='pink',label='Cluster 4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='red',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='black',label='Centroids')#centroids feature 1 Vs centroids feature 2
plt.title('Customers CLustering')
plt.xlabel('Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.savefig('figure_2.pdf')#Make sure your figure name is same as in tex file
plt.show()
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Generating the pdf from (Tex_file.tex) based on the new data (numbers/figures/tables)
template = dont_edit_this_file.get_template('./Tex_file.tex')
variable_dict = {'names': product_list}
dont_edit_this_file.compile_pdf_from_template(template, variable_dict, './Result.pdf')
