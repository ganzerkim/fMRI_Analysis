#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:02:26 2020

@author: mg
"""
from nilearn import datasets

dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
# cort-maxprob-thr0-1mm
# cort-maxprob-thr0-2mm
# cort-maxprob-thr25-1mm
# cort-maxprob-thr25-2mm
# cort-maxprob-thr50-1mm
# cort-maxprob-thr50-2mm
# sub-maxprob-thr0-1mm
# sub-maxprob-thr0-2mm
# sub-maxprob-thr25-1mm
# sub-maxprob-thr25-2mm
# sub-maxprob-thr50-1mm
# sub-maxprob-thr50-2mm
# cort-prob-1mm
# cort-prob-2mm
# sub-prob-1mm
# sub-prob-2mm

atlas_filename = dataset.maps
labels = dataset.labels



#%%
from nilearn import datasets
atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas['maps']
# Loading atlas data stored in 'labels'
labels = atlas['labels']

#%%

print('Atlas ROIs are located in nifti image (4D) at: %s' %
      atlas_filename)  # 4D data

# One subject of brain development fmri data
#data = datasets.fetch_development_fmri(n_subjects=1)
fmri_filenames = '/home/bk/Desktop/bkrest/sub-ID01/func/sub-ID01_task-rest1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
confounds = '/home/bk/Desktop/bkrest/sub-ID01/func/sub-ID01_task-rest1_desc-confounds_regressors.tsv'

from nilearn.input_data import NiftiLabelsMasker
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                           memory='nilearn_cache', verbose=5)

# Here we go from nifti files to the signal time series in a numpy
# array. Note how we give confounds to be regressed out during signal
# extraction
time_series = masker.fit_transform(fmri_filenames)#, confounds=confounds)


from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series])[0]



# Plot the correlation matrix
import numpy as np
from nilearn import plotting
import cv2 as cv

correlation_matrix_norm = cv.normalize(correlation_matrix, None, -1, 1, cv.NORM_MINMAX)
# Make a large figure
# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix_norm, 0)
# The labels we have start with the background (0), hence we skip the
# first label
# matrices are ordered for block-like representation
plotting.plot_matrix(correlation_matrix_norm, figure=(10, 8), labels=labels[1:],
                     vmax=1, vmin=-1, reorder=False)



coords = atlas.region_coords

# We threshold to keep only the 20% of edges with the highest value
# because the graph is very dense
plotting.plot_connectome(correlation_matrix_norm, coords,
                         edge_threshold="80%", colorbar=True)

plotting.show()



view = plotting.view_connectome(correlation_matrix, coords, edge_threshold='80%')

# In a Jupyter notebook, if ``view`` is the output of a cell, it will
# be displayed below the cell
view

time_series = masker.fit_transform(fmri_filenames)
# Note how we did not specify confounds above. This is bad!

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                     vmax=0.8, vmin=-0.8, reorder=True)

plotting.show()
