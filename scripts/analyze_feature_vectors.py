#!/usr/bin/env python

"""
Analyze the feature vectors
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors
from matplotlib.colors import ListedColormap

import IPython

data_folder = '../data/'
image_folder = '../figs/'

from play import big_cloths, hards, rugs, small_cloths, all_materials
from extract_feature_vector import test_files

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
        
    #####################
    # NN classification #
    #####################
    
    def classify(self, test_afv):
        def label(name):
            if 'r' in name: return 0
            elif 'h' in name: return 1
            elif 'bc' in name: return 2
            elif 'sc' in name: return 3
            else: raise Exception()
                
        def name(num):
            if num == 0: return 'rugs'
            elif num == 1: return 'hards'
            elif num == 2: return 'big cloths'
            elif num == 3: return 'small cloths'
            else: raise Exception()
    
        X = self.data.T
        y = np.array([label(n) for n in self.file_names])
    
        n_neighbors = 3 # 3
        weights = 'uniform' # uniform
            
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)
        
        
        for file_name in test_afv.file_names:
            x_test = test_afv.fvs[file_name]
            l = clf.predict(x_test)[0]
        
            print('{0}'.format(file_name))
            print('Class : {0}'.format(name(l)))
            closest, min_dist = None, np.inf
            for f in self.file_names:
                if label(f) == l:
                    dist = np.linalg.norm(self.fvs[f] - x_test)
                    if dist < min_dist:
                        min_dist = dist
                        closest = f
            print('Closest : {0}'.format(closest))
            print('')
    
            
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
            
    def display_materials_2d(self, save_file=None, plot_nn=False, nn_res=1, loc='upper left'):
        cov = self.covariance
        U, S, V = np.linalg.svd(cov)
        
        P = U[:,:2].T.dot(self.data)
        
        f = plt.figure(figsize=(15,15))
        ax = f.add_subplot(111)

        material = defaultdict(list)
        categories = ['rugs', 'hards', 'big cloths', 'small cloths']
        #styles = ['bo', 'g^', 'rx', 'gx']
        styles = ['o', '^', 'x', 'x']
        for i, file_name in enumerate(self.file_names):
            if file_name in rugs:
                material['rugs'].append(list(P[:,i]))
            elif file_name in big_cloths:
                material['big cloths'].append(list(P[:,i]))
            elif file_name in hards:
                material['hards'].append(list(P[:,i]))
            elif file_name in small_cloths:
                material['small cloths'].append(list(P[:,i]))

        cmap_light = ListedColormap(['#AAFFAA', '#AAAAFF', '#FFAAAA', '#FFAAFF'])
        cmap_bold = ListedColormap(['#00FF00', '#0000FF', '#FF0000', '#FF00FF'])
    
        for m, style, color in zip(categories, styles, cmap_bold.colors):
            arr = np.array(material[m])
            ax.plot(arr[:,0], arr[:,1], style, color=color, markersize=8.0, label=m)

        #plt.title('Projected sound feature vectors')
        
        if plot_nn:
            n_neighbors = 5 # 3
            weights = 'distance' # uniform
            
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            X, y = list(), list()
            for i, m in enumerate(categories):
                X += material[m]
                y += len(material[m])*[i]
            
            X, y = np.array(X), np.array(y)
                        
            clf.fit(X, y)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            x_min, x_max = X[:, 0].min() - nn_res, X[:, 0].max() + nn_res
            y_min, y_max = X[:, 1].min() - nn_res, X[:, 1].max() + nn_res
            xx, yy = np.meshgrid(np.arange(x_min, x_max, nn_res),
                                 np.arange(y_min, y_max, nn_res))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            #plt.figure()
            ax.pcolormesh(xx, yy, Z, cmap=cmap_light)


        plt.legend(loc=loc)
        plt.show(block=False)
        
        if save_file is not None:
            plt.savefig(save_file)
        
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
    
    afv.display_materials_2d(save_file=image_folder+'sound_projected.jpg', plot_nn=True, nn_res=10)
    
    print('Press enter to exit')
    raw_input()
    
def analyze_images():
    file_names = sorted([data_folder + material + '.npy' for material in all_materials])
    afv = AnalyzeFeatureVectors(file_names)
    
    afv.display_materials_2d(save_file=image_folder+'image_projected.jpg', plot_nn=True, nn_res=0.0002, loc='lower right')
    
    IPython.embed()
        
def classify():
    file_names = sorted([data_folder + material + '.npy' for material in all_materials])
    afv = AnalyzeFeatureVectors(file_names)
    
    test_file_names = sorted([data_folder + f + '.npy' for f in test_files])
    test_afv = AnalyzeFeatureVectors(test_file_names)
    
    afv.classify(test_afv)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', choices=('push','sound','images','classify'))
    args = parser.parse_args()
    
    if args.type == 'push':
        analyze_push()
    elif args.type == 'sound':
        analyze_sound()
    elif args.type == 'images':
        analyze_images()
    elif args.type == 'classify':
        classify()
