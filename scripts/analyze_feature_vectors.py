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
image_folder = '../figs/'

from play import big_cloths, hards, rugs, small_cloths

class AnalyzeFeatureVectors:
    def __init__(self, file_paths):
        """
        :param file_names: list of .npy containing feature vectors
        """
        self.file_names = [f.replace('push_','').replace('sound_','').replace(data_folder,'').replace('.npy','') for f in file_paths]
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
        
    @property
    def similarity_matrix(self):
        D = self.distance_matrix
        return D.max() - D
            
    ###################
    # display methods #
    ###################
    
    def display_similarity_matrix(self, save_file=None):
        N = len(self.file_names)

        plt.figure()
        plt.pcolor(self.similarity_matrix)
        plt.colorbar()
        plt.yticks(np.arange(0.5,N+0.5), self.file_names)
        plt.xticks(np.arange(0.5,N+0.5), self.file_names, rotation='vertical')
        plt.title('Similarity matrix')
        plt.show(block=False)
        
        if save_file is not None:
            plt.savefig(save_file)
            
    def display_2d(self):
        cov = self.covariance
        U, S, V = np.linalg.svd(cov)
        
        P = U[:,:2].T.dot(self.data)
        
        f = plt.figure()
        ax = f.add_subplot(111)
        
        for i, file_name in enumerate(self.file_names):
            s='rx'
            if file_name in rugs:
                s = 'bo'
            elif file_name in big_cloths:
                s = 'g^'
            elif file_name in hards:
                s = 'rx'
            elif file_name in small_cloths:
                s = 'gx'
    
            ax.plot(P[0,i], P[1,i], s, markersize=8.0)
        
        #plt.plot(P[0,:], P[1,:], 'rx', markersize=10.0)
        plt.show(block=False)
        
        IPython.embed()
        
    #################
    # print methods #
    #################
    
    def print_most_similar(self):
        S = self.similarity_matrix
        np.fill_diagonal(S, 0)
        for i, file_name in enumerate(self.file_names):
            j = S[i,:].argmax()
            print('{0:<15} : {1}'.format(file_name, self.file_names[j]))
        
#########
# TESTS #
#########

def analyze_push():
    file_names = [data_folder+f for f in os.listdir(data_folder) if '.npy' in f and 'push' in f]
    file_names = sorted(sorted(file_names), key=lambda x : x.split('_')[-1])
    afv = AnalyzeFeatureVectors(file_names)
    
    afv.display_similarity_matrix(save_file=image_folder+'push_similarity.jpg')
    afv.print_most_similar()
    
    print('Press enter to exit')
    raw_input()
        
def analyze_sound():
    file_names = sorted([data_folder+f for f in os.listdir(data_folder) if '.npy' in f and 'sound' in f])
    file_names = [data_folder+'sound_'+f+'.npy' for f in small_cloths + big_cloths + hards + rugs]
    afv = AnalyzeFeatureVectors(file_names)
    
    #afv.display_similarity_matrix(save_file=image_folder+'sound_similarity.jpg')
    #afv.print_most_similar()
    
    afv.display_2d()
    
    print('Press enter to exit')
    raw_input()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=('push','sound'))
    args = parser.parse_args()
    
    if args.type == 'push':
        analyze_push()
    elif args.type == 'sound':
        analyze_sound()
