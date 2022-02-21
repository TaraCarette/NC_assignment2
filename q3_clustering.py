'''
a) Implement the PSO algorithm for clustering
b) Implement K-means
c) Generate Artificial dataset 1
d) Compare performance of PSO and K-means in terms of quantization error 
        on Artificial dataset 1 and on the Iris dataset
e) plot the clusters
'''

from turtle import xcor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pso import ParticleSwarmOptimizedClustering
from particle import quantization_error
from kmeans import KMeans


def main():
    x, iris = generate_data()

    pso_clustering(x)
    kmeans(x)

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
    data = pd.read_csv('seed.txt', sep='\t', header=None)
    # print(data.head())
    x = data.drop([7], axis=1)
    x = x.values
    x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0)) # normalization step
    
    #TODO: get iris dataset
    
    return x, 'iris'



def compare():
    pass

def plot():
    pass

if __name__ == '__main__':
    positions = main()