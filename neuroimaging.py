# -*- coding: utf-8 -*-
"""
Created on Wed May 27 09:17:07 2020

@author: MIT-DGMIF
"""

import os
import pandas as pd
import numpy as np
import scipy as sp
import random
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

os.listdir('E:/Dataset/trends-assessment-prediction/')

features = pd.read_csv('E:/Dataset/trends-assessment-prediction/train_scores.csv')
loading = pd.read_csv('E:/Dataset/trends-assessment-prediction/loading.csv')
submission = pd.read_csv('E:/Dataset/trends-assessment-prediction/sample_submission.csv')
fnc = pd.read_csv("E:/Dataset/trends-assessment-prediction/fnc.csv")
reveal = pd.read_csv('E:/Dataset/trends-assessment-prediction/reveal_ID_site2.csv')
numbers = pd.read_csv('E:/Dataset/trends-assessment-prediction/ICN_numbers.csv')
fmri_mask = 'E:/Dataset/trends-assessment-prediction/fMRI_mask.nii'


import nilearn as nl
import nilearn.plotting as nlplt
import nibabel as nib
from nilearn import image
from nilearn import plotting
from nilearn import datasets
from nilearn import surface

smri = 'E:/Dataset/trends-assessment-prediction/ch2better.nii'
mask_img = nl.image.load_img(fmri_mask)

def load_subject(filename, mask_img):
    subject_data = None
    with h5py.File(filename, 'r') as f:
        subject_data = f['SM_feature'][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0,1,2,3], [3,2,1,0])
    subject_img = nl.image.new_img_like(mask_img, subject_data, affine=mask_img.affine, copy_header=True)

    return subject_img



files = random.choices(os.listdir('E:/Dataset/trends-assessment-prediction/fMRI_train/'), k = 3)
for file in files:
    subject = os.path.join('E:/Dataset/trends-assessment-prediction/fMRI_train/', file)
    subject_img = load_subject(subject, mask_img)
    print("Image shape is %s" % (str(subject_img.shape)))
    num_components = subject_img.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))
    nlplt.plot_prob_atlas(subject_img, bg_img=smri, view_type='filled_contours',
                          draw_cross=False, title='All %d spatial maps' % num_components, threshold='auto')
    print("-" * 50)
    
#%%
    
files = random.choices(os.listdir('E:/Dataset/trends-assessment-prediction/fMRI_train/'), k = 3)
for file in files:
    subject = os.path.join('E:/Dataset/trends-assessment-prediction/fMRI_train/', file)
    subject_img = load_subject(subject, mask_img)
    print("Image shape is %s" % (str(subject_img.shape)))
    num_components = subject_img.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))
    rsn = subject_img
    #convert to 3d image
    first_rsn = image.index_img(rsn, 0)
    print(first_rsn.shape)
    plotting.plot_stat_map(first_rsn)
    print("-"*50)
    
#%%
files = random.choices(os.listdir('E:/Dataset/trends-assessment-prediction/fMRI_train/'), k = 1)
for file in files:
    subject = os.path.join('E:/Dataset/trends-assessment-prediction/fMRI_train/', file)
    subject_img = load_subject(subject, mask_img)
    print("Image shape is %s" % (str(subject_img.shape)))
    num_components = subject_img.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))
    rsn = subject_img
    #convert to 3d image
    first_rsn = image.index_img(rsn, 0)
    print(first_rsn.shape)
    for img in image.iter_img(rsn):
        # img is now an in-memory 3D img
        plotting.plot_stat_map(img, threshold=3)
    print("-"*50)


#%%
files = random.choices(os.listdir('E:/Dataset/trends-assessment-prediction/fMRI_train/'), k = 1)
for file in files:
    subject = os.path.join('E:/Dataset/trends-assessment-prediction/fMRI_train/', file)
    subject_img = load_subject(subject, mask_img)
    print("Image shape is %s" % (str(subject_img.shape)))
    num_components = subject_img.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))
    rsn = subject_img
    #convert to 3d image
    first_rsn = image.index_img(rsn, 0)
    print(first_rsn.shape)
    plotting.plot_epi(first_rsn)
    print("-"*50)

#%%
files = random.choices(os.listdir('E:/Dataset/trends-assessment-prediction/fMRI_train/'), k = 3)
for file in files:
    subject = os.path.join('E:/Dataset/trends-assessment-prediction/fMRI_train/', file)
    subject_img = load_subject(subject, mask_img)
    print("Image shape is %s" % (str(subject_img.shape)))
    num_components = subject_img.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))
    rsn = subject_img
    #convert to 3d image
    first_rsn = image.index_img(rsn, 0)
    print(first_rsn.shape)
    plotting.plot_anat(first_rsn)
    print("-"*50)

#%%
files = random.choices(os.listdir('E:/Dataset/trends-assessment-prediction/fMRI_train/'), k = 3)
for file in files:
    subject = os.path.join('E:/Dataset/trends-assessment-prediction/fMRI_train/', file)
    subject_img = load_subject(subject, mask_img)
    print("Image shape is %s" % (str(subject_img.shape)))
    num_components = subject_img.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))
    rsn = subject_img
    #convert to 3d image
    first_rsn = image.index_img(rsn, 0)
    print(first_rsn.shape)
    plotting.plot_roi(first_rsn)
    print("-"*50)
    
#%%
files = random.choices(os.listdir('E:/Dataset/trends-assessment-prediction/fMRI_train/'), k = 1)
for file in files:
    subject = os.path.join('E:/Dataset/trends-assessment-prediction/fMRI_train/', file)
    subject_img = load_subject(subject, mask_img)
    print("Image shape is %s" % (str(subject_img.shape)))
    num_components = subject_img.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))
    rsn = subject_img
    #convert to 3d image
    first_rsn = image.index_img(rsn, 0)
    print(first_rsn.shape)     
    plotting.plot_glass_brain(first_rsn,display_mode='lyrz')
    print("-"*50)
    
    
    
#%%
motor_images = datasets.fetch_neurovault_motor_task()
stat_img = motor_images.images[0]
view = plotting.view_img_on_surf(first_rsn, threshold='90%')
view.open_in_browser()
view