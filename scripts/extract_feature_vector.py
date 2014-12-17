#!/usr/bin/env python

"""
Extract feature vectors from data
"""

from collections import defaultdict

import rospy, rosbag, roslib
roslib.load_manifest('tfx')
import tfx
#roslib.load_manifest('cv_bridge')
#import cv_bridge

import scipy as sci
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.filters import gaussian_filter1d

import argparse
import IPython

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb, rgb2gray
#import os

import scipy.io.wavfile
from scikits.talkbox.features import mfcc

from play import all_materials, object_materials, floor_materials

data_folder = '../data/'
image_folder = '../figs/'
push_files = ['push_{0}_on_{1}'.format(o, f) for o in object_materials for f in floor_materials]
sound_files = ['sound_{0}'.format(o) for o in all_materials]

class ExtractFeatureVector:
    def __init__(self, npz_name):
        self.name = npz_name[:-4]
        npzfile = np.load(npz_name)
        
        self.l_finger_tip = npzfile['l_finger_tip']
        self.r_finger_tip = npzfile['r_finger_tip']
        self.t_pressure = npzfile['t_pressure']
        
        self.joint_positions = npzfile['joint_positions']
        self.joint_velocities = npzfile['joint_velocities']
        self.joint_efforts = npzfile['joint_efforts']
        self.t_joint = npzfile['t_joint']
        
        self.poses = [tfx.pose(p) for p in npzfile['poses']]
        
        if 'push' in self.name:
            self.object_material = self.name.split('_')[1]
            self.floor_material = self.name.split('_')[-1]
        
            self.object_img = sci.misc.imread(data_folder + self.object_material + '.jpg')
            self.floor_img = sci.misc.imread(data_folder + self.floor_material + '.jpg')
            
            self.object_texture = ImageTexture(self.object_img)
            self.floor_texture = ImageTexture(self.floor_img)
        elif 'sound' in self.name:
            self.object_material = self.name.split('_')[1]
            #self.audio = npzfile['audio']
            self.rate, self.audio = scipy.io.wavfile.read(self.name + '.wav')
        else:
            self.object_im = None
            self.floor_img = None
            
    ##########################
    # feature vector methods #
    ##########################
    
    @property
    def feature_vector(self):
        """
        Generate feature vector from data
        """
        # energy is no good! depends on time of first contact
        
        v = list()
        v += self.max_finger_tip_forces # good
        #v.append(self.finger_tip_forces_energy)
        #v.append(self.max_forces_derivative)
        #v.append(self.distance_traveled)
        #v.append(self.max_joint_velocity)
        v.append(self.max_joint_effort) # good
        #v.append(self.joint_effort_energy)
        #v.append(self.texture_similarity)
        return np.array(v)
                
    @property
    def max_finger_tip_forces(self):
        return [self.l_finger_tip.max(), self.r_finger_tip.max()]

    @property
    def finger_tip_forces_energy(self):
        energy = 0
        for a in [self.l_finger_tip, self.r_finger_tip]:
            for col in xrange(a.shape[1]):
                energy += np.linalg.norm(np.convolve(a[:,col], [1,-1]))
        return energy

    @property
    def max_forces_derivative(self):
        d_max = -np.inf
        for a in [self.l_finger_tip, self.r_finger_tip]:
            for col in xrange(a.shape[1]):
                #d_max = max(d_max, np.convolve(a[:,col], [1,-1]).max())
                d_max = max(d_max, (a[:,col] - gaussian_filter1d(a[:,col],1)).max())
        return d_max

    @property
    def forces_entropy(self):
        """
        Histogram the forces, then compute the entropy
        """
        finger_bins = [np.linspace(0, m, 10) for m in self.max_finger_tip_forces]
        
        entropy = 0
        for a, bins in zip([self.l_finger_tip, self.r_finger_tip], finger_bins):
            for col in xrange(a.shape[1]):
                h, _ = np.histogram(a[:,col], normed=False, bins=bins)
                h = np.array(h, dtype=float)
                h /= float(np.sum(h))
                filt = h != 0
                entropy += -np.sum(h[filt] * np.log2(h[filt]))
        return entropy
        

    @property
    def distance_traveled(self):
        return sum([self.poses[i].position.distance(self.poses[i+1].position) for i in xrange(len(self.poses)-1)])

    @property        
    def max_joint_velocity(self):
        return self.joint_velocities.max()
    
    @property
    def max_joint_effort(self):
        return self.joint_efforts.max()
        
    @property
    def joint_effort_energy(self):
        energy = 0
        for col in xrange(self.joint_efforts.shape[1]):
            energy += np.linalg.norm(np.convolve(self.joint_efforts[:,col], [1, -1]))
        return energy
        
    @property
    def texture_similarity(self):
        return self.object_texture.kl_divergence(self.floor_texture)
                
        
    def __str__(self):
        """
        Print relevant statistics, for guidance
        """
        s = ''
        #s += 'Max finger tip forces: {0}\n'.format(self.max_finger_tip_forces)
        #s += 'Finger tip forces energy: {0:.3f}\n'.format(self.finger_tip_forces_energy)
        #s += 'Max forces derivative: {0:.3f}\n'.format(self.max_forces_derivative)
        s += 'Forces entropy: {0:.3f}\n'.format(self.forces_entropy)
        #s += 'Distance traveled: {0:.3f}\n'.format(self.distance_traveled)
        #s += 'Max joint velocity: {0:.3f}\n'.format(self.max_joint_velocity)
        #s += 'Max joint effort: {0:.3f}\n'.format(self.max_joint_effort)
        #s += 'Joint effort energy: {0:.3f}\n'.format(self.joint_effort_energy)
        #s += 'Object-floor texture similarity: {0:.3f}\n'.format(self.texture_similarity)
        return s
        
    #################
    # sound methods #
    #################
    
    @property
    def sound_feature_vector(self):
        v = list()
        v += self.sound_peaks
        #v += [np.max(self.sound_peaks), np.mean(self.sound_peaks)] # GOOD
        #v.append(self.avg_sound_peak)
        #v += list(self.mfcc_sound_features)
        #v.append(self.sound_energy)
        return np.array(v)
    
    @property
    def sound_peaks(self):
        audio = self.audio[:,0]
        N = len(audio)
        
        bw = 0.5*self.rate
        peak_indices = list()
        audio_peaks = audio.copy()
        for i in xrange(3):
            peak = audio_peaks.argmax()
            peak_indices.append(peak)
            
            audio_peaks[max(0,peak-bw):min(N-1,peak+bw)] = 0
            
        peaks = [audio[p] for p in peak_indices]
        
        """    
        plt.figure(1)
        plt.plot(xrange(N), audio, 'b-', peak_indices, peaks, 'g^')
        
        plt.figure(2)
        g = np.abs(np.convolve(audio, [1,-1]))
        plt.plot(g)
            
        print('peak_indices: {0}'.format(peak_indices))
        #audio_fft = np.fft.rfft(audio)
        #audio_fft = abs(audio_fft)
        #audio_fft = 10*np.log10(audio_fft)
        #plt.plot(audio_fft[:,0])
        plt.show(block=False)
        """
        return peaks
    
    @property
    def avg_sound_peak(self):
        return np.mean(self.sound_peaks)
        
    @property
    def mfcc_sound_features(self):
        ceps, mspec, spec = mfcc(self.audio)
        
        num_ceps = len(ceps)
        #v = np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0)
        v = np.mean(ceps, axis=0)
        
        return v
        
    @property
    def sound_energy(self):
        audio = self.audio[:,0]
        #return np.linalg.norm(np.convolve(audio, [1,-1]))
        return np.linalg.norm(np.fft.rfft(audio)) / float(len(audio))
        
    ###################
    # display methods #
    ###################
        
    def display_forces(self, save_file=None):
        """
        Display fingertip pressure forces
        :param file to save figure to
        """
        N = self.l_finger_tip.shape[1]
        
        f, axes = plt.subplots(N, 2, sharex=True, sharey=True)
        axes_left = axes[:,0]
        axes_right = axes[:,1]

        axes_left[0].set_title('Left finger')
        axes_right[0].set_title('Right finger')
    
        for i in xrange(N):
            axes_left[i].plot(self.t_pressure, self.l_finger_tip[:,i], 'r')
            axes_right[i].plot(self.t_pressure, self.r_finger_tip[:,i], 'b')
            
        f.subplots_adjust(hspace=0)
        f.suptitle(self.name)
            
        plt.show(block=False)
        
        if save_file is not None:
            plt.savefig(save_file)
        
        return f

    def display_materials(self):
        """
        Display images of materials
        """
        if self.object_image is not None:
            plt.figure(1)
            plt.imshow(self.object_image)
            plt.title('Object material')
        if self.floor_img is not None:
            plt.figure(2)
            plt.imshow(self.floor_img)
            plt.title('Floor material')
            
        plt.show(block=False)
        
    def display_sound(self, save_file=None):
        rate, audio = scipy.io.wavfile.read(self.name + '.wav')
        plt.plot(audio)
        plt.show(block=False)
        
        if save_file is not None:
            plt.savefig(save_file)
        plt.close()
        
class ImageTexture:
    def __init__(self, img):
        self.img = rgb2gray(img)
        
    @property
    def histogram(self):
        radius = 5 # 3
        n_points = 20*radius # 8 * radius
        method = 'uniform'
        
        lbp = local_binary_pattern(self.img, n_points, radius, method)
        
        n_bins = lbp.max() + 1
        histogram, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
        return histogram
        
        
    def kl_divergence(self, other):
        h, h_o = np.asarray(self.histogram), np.asarray(other.histogram)
        filt = np.logical_and(h != 0, h_o != 0)
        return np.sum(h[filt] * np.log2(h[filt] / h_o[filt]))



#########
# TESTS #
#########

def save_all_display_forces():
    for push_file in push_files:
        try:
            efd = ExtractFeatureVector(data_folder + push_file + '.npz')
            f = efd.display_forces(save_file=image_folder + push_file + '.jpg')
            print('Saved image for push_file: {0}'.format(push_file))
            plt.close()
        except:
            pass
        
def print_push_files():
    efds = list()
    for push_file in sorted(sorted(push_files), key=lambda x : x.split('_')[-1]):
        try:
            efds.append(ExtractFeatureVector(data_folder + push_file + '.npz'))
        except Exception as e:
            pass
                        
    for efd in efds:
        print(efd.name)
        print(str(efd) + '\n')
            
def save_push_feature_vectors():
    print('Saving push feature vectors...')
    efds = list()
    for push_file in push_files:
        try:
            efds.append(ExtractFeatureVector(data_folder + push_file + '.npz'))
            print('{0}'.format(efds[-1].name))
        except Exception as e:
            pass
                        
    for efd in efds:
        np.save(efd.name+'.npy', efd.feature_vector)
            
def save_sound_feature_vectors():
    print('Saving sound feature vectors...')
    efds = list()
    for sound_file in sorted(sound_files):
        try:
            efd = ExtractFeatureVector(data_folder + sound_file + '.npz')
            print('{0}'.format(efd.name))
            #efd.display_sound(save_file=image_folder + sound_file + '.jpg')
            efds.append(efd)
        except Exception as e:
            pass
     
    for efd in efds:
        np.save(efd.name+'.npy', efd.sound_feature_vector)
            
    #names_peaks = [(efd.name, efd.avg_sound_peak) for efd in efds]
    #for name, peak in sorted(names_peaks, key=lambda x: x[1]):
    #    print('{0} : {1}'.format(name, peak))
     
    #names_E = [(efd.name, efd.sound_energy) for efd in efds]
    #for name, e in sorted(names_E, key=lambda x: x[1]):
    #    print('{0} : {1}'.format(name, e))
        
    #mfccs = [efd.mfcc_sound_features for efd in efds]
    #for i, f in enumerate(mfccs):
    #    dists = [np.linalg.norm(f - other) for j, other in enumerate(mfccs) if i != j]
    #    min_ind = np.argmin(dists)
    #    print('{0:<10} : {1}'.format(efds[i].name, efds[min_ind].name))           
                    
def save_image_feature_vectors():
    for material in all_materials:
        img = sci.misc.imread(data_folder + material + '.jpg')
        texture = ImageTexture(img)
        hist = texture.histogram
        npy_name = data_folder + material + '.npy'
        print('Saving histogram to {0}'.format(npy_name))
        np.save(npy_name, hist)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('npz_name')
    parser.add_argument('--save-all-display-forces', action='store_true')
    parser.add_argument('--print-push-files', action='store_true')
    parser.add_argument('--save-push', action='store_true')
    parser.add_argument('--save-sound', action='store_true')
    parser.add_argument('--save-images', action='store_true')

    args = parser.parse_args()
    
    if args.npz_name == 'all':
        if args.save_all_display_forces:
            save_all_display_forces()
        elif args.print_push_files:
            print_push_files()
        elif args.save_push:
            save_push_feature_vectors()
        elif args.save_sound:
            save_sound_feature_vectors()
        elif args.save_images:
            save_image_feature_vectors()
    else:
        efd = ExtractFeatureVector(data_folder + args.npz_name)
        #efd.display_forces()
        #efd.display_materials()
        #print(efd)
        #efd.avg_sound_peak
        efd.mfcc_sound_features
        #IPython.embed()
        
    
    #print('Press enter to exit')
    #raw_input()
    

