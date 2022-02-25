'''
a) Implement the PSO algorithm for clustering
b) Implement K-means
c) Generate Artificial dataset 1
d) Compare performance of PSO and K-means in terms of quantization error 
        on Artificial dataset 1 and on the Iris dataset
e) plot the clusters
'''

from re import X
from turtle import xcor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pso import ParticleSwarmOptimizedClustering
from particle import quantization_error
from kmeans import KMeans
from sklearn import datasets
import random
import matplotlib.pyplot as plt



def main():
    x, iris = generate_data()

    # artificial dataset 1
    # pso_result, pso = pso_clustering(x)
    # kmeans_result, kmeans = get_kmeans(x)
    # compare(x, pso_result, pso, kmeans_result, kmeans)
    #plot(x, kmeans_result, kmeans)
    
    
    # Iris dataset
    pso_result, pso = pso_clustering(iris)
    kmeans_result, kmeans = get_kmeans(iris)
    compare(iris, pso_result, pso, kmeans_result, kmeans)
    plot(iris, kmeans_result, kmeans)

    
    
    

def pso_clustering(x):
    # 1) initialize
        # 2) for t = 1 to tmax do:
            # 2a) for each particle i do:
                # 2b) for each data vector Zp:
                    # 2b I) calculate Euclidean distance
                    # 2b II) assign Zp to cluster
                    # 2b III) calculate fitness
                
                # 2c) update global and local best positions
                # 2d) update cluster centroids
        
    pso = ParticleSwarmOptimizedClustering(
        n_cluster=2, 
        n_particles=10, 
        data=x, 
        hybrid=False, 
        max_iter=100, 
        )

    return pso.run(), pso
            

def get_kmeans(x):
    kmeans = KMeans(n_cluster=3, init_pp=False, seed=2018)
    kmeans.fit(x)
    return kmeans.predict(x), kmeans


def generate_data():
    
    # Dummy data
    data = pd.read_csv('seed.txt', sep='\t', header=None)
    # print(data.head())
    x = data.drop([7], axis=1)
    x = x.values
    x_normalized = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)) # normalization step

    
    
    # Generate artificial problem 1
    df = pd.DataFrame([], columns=list('xyc'),)
    for i in range(400):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        label = 1 if (x >= 0.7) or ((x <= 0.3) and (y >= -0.2 - x)) else 0
        df = df.append({'x': x, 'y': y, 'c' : label}, ignore_index=True)
    x = df.values
    # print(df)
    # df.plot(x = 'x', y = 'y', c= 'c', kind='scatter')
    # plt.show()
    
    
    # Iris dataset
    iris = datasets.load_iris()
    iris_df = pd.DataFrame(iris.data)
    iris_df['class']=iris.target
    iris_df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
    iris_df.dropna(how="all", inplace=True) # remove any empty lines
    
    #selecting only first 4 columns as they are the independent(X) variable
    # any kind of feature selection or correlation analysis should be first done on these
    iris_X = iris_df.iloc[:,[0,1,2,3]]
    iris_X = iris_X.values
    iris_X = (iris_X - iris_X.min(axis=0)) / (iris_X.max(axis=0) - iris_X.min(axis=0))
    
    return x, iris_X



def compare(x, pso_result, pso, kmeans_result, kmeans):
    print('Quantization Kmeans:'
          , quantization_error(centroids=kmeans.centroid, data=x, labels=kmeans_result))
    print('Quantization PSO:',
          pso.gbest_score)

def plot(x, kmeans_result, kmeans):
    # plot kmeans result
    label = kmeans_result
    filtered_label0 = x[label == 0]
    filtered_label1 = x[label == 1]
    filtered_label2 = x[label == 2]
    
    plt.scatter(filtered_label0[:,0] , filtered_label0[:,1] , color = 'red')
    plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'black')
    plt.scatter(filtered_label2[:,0] , filtered_label2[:,1] , color = 'blue')
    plt.show()
    
    # plot PSO result
    

if __name__ == '__main__':
    positions = main()