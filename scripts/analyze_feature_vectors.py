#!/usr/bin/env python

"""
Analyze the feature vectors
"""

import os
import argparse


import numpy as np
import matplotlib.pyplot as plt

import IPython

data_folder = '../data/'

class AnalyzeFeatureVectors:
    def __init__(self, file_paths):
        """
        :param file_names: list of .npy containing feature vectors
        """
        self.file_names = [f.replace(data_folder,'').replace('.npy','') for f in file_paths]
        names_and_files = zip(self.file_names, file_paths)
        self.fvs = { n : np.load(f) for n, f in names_and_files }
        
    @property
    def data(self):
        return np.array([v for v in self.fvs.values()]).T
        
    @property
    def mean(self):
        return np.mean(self.data)        
        
    @property
    def covariance(self):
        return np.cov(self.data)
        
    @property
    def distance_matrix(self):
        """
        :return file_names, matrix
        """
        N = len(self.file_names)
        
        inv_cov = np.linalg.inv(self.covariance)
        D = np.zeros(2*[N])
        for i, f_i in enumerate(self.file_names):
            for j, f_j in enumerate(self.file_names):
                v_i, v_j = self.fvs[f_i], self.fvs[f_j]
                D[i,j] = np.sqrt((v_i - v_j).T.dot(inv_cov).dot(v_i - v_j))
                
        return D
            
    ###################
    # display methods #
    ###################
    
    def display_distance_matrix(self):
        D = self.distance_matrix
        N = len(D)
        
        # similarity matrix
        S = D.max() - D
        
        plt.figure()
        plt.pcolor(S)
        plt.colorbar()
        plt.yticks(np.arange(0.5,N+0.5), self.file_names)
        plt.xticks(np.arange(0.5,N+0.5), self.file_names, rotation='vertical')
        plt.show(block=False)
        
if __name__ == '__main__':
    file_names = [data_folder+f for f in os.listdir(data_folder) if '.npy' in f]
    file_names = sorted(sorted(file_names), key=lambda x : x.split('_')[-1])
    afv = AnalyzeFeatureVectors(file_names)
    
    D = afv.distance_matrix
    afv.display_distance_matrix()
    
    print('Press enter to exit')
    raw_input()
        
