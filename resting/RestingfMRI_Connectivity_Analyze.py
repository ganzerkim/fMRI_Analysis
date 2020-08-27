#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 13:57:38 2020

@author: Mingeon Kim
"""
################msdl############################
from nilearn import datasets

atlas = datasets.fetch_atlas_msdl()

# Loading atlas image stored in 'maps'
atlas_filename = atlas['maps']
# Loading atlas data stored in 'labels'
labels = atlas['labels']

# Load the functional datasets
data = datasets.fetch_development_fmri(n_subjects=1)
print('First subject resting-state nifti image (4D) is located at: %s' %
      data.func[0])

func = '/home/bk/Desktop/bkrest/sub-ID07/func/sub-ID07_task-rest1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'


from nilearn.input_data import NiftiMapsMasker
masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                         memory='nilearn_cache', verbose=5)

time_series = masker.fit_transform(func,
                                   confounds=None)

print(time_series.shape)

from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Display the correlation matrix
import numpy as np
from nilearn import plotting
import cv2 as cv
import matplotlib.pyplot as plt

# Mask out the major diagonal
np.fill_diagonal(correlation_matrix, 0)
correlation_matrix_norm = cv.normalize(correlation_matrix, None, -1, 1, cv.NORM_MINMAX)
plotting.plot_matrix(correlation_matrix_norm, labels=labels, colorbar=True,
                     vmax=0.8, vmin=-0.8)

plt.imshow(correlation_matrix_norm, cmap = 'rainbow', vmin = -0.8, vmax = 0.8)

from nilearn import plotting
coords = atlas.region_coords

# We threshold to keep only the 20% of edges with the highest value
# because the graph is very dense
plotting.plot_connectome(correlation_matrix, coords,
                         edge_threshold="95%", colorbar=True)

plotting.show()



view = plotting.view_connectome(correlation_matrix, coords, edge_threshold='80%')

# In a Jupyter notebook, if ``view`` is the output of a cell, it will
# be displayed below the cell
view

