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

    pso_result = pso_clustering(x)
    kmeans_result = kmeans(x)

    compare()
    plot()
    
    

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
        n_cluster=3, 
        n_particles=10, 
        data=x, 
        hybrid=False, 
        max_iter=20, 
        )

    return pso.run()
            

def kmeans(x):
    kmeans = KMeans(n_cluster=3, init_pp=False, seed=2018)
    kmeans.fit(x)
    predicted_kmeans = kmeans.predict(x)
    return predicted_kmeans


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
        if (x >= 0.7) or ((x <= 0.3) and (y >= -0.2 - x)):
            label = 1
        else:
            label = 0
    
        df = df.append({'x': x, 'y': y, 'c' : label}, ignore_index=True)
    
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
    
    return x_normalized, iris_X



def compare():
    pass

def plot():
    pass

if __name__ == '__main__':
    positions = main()