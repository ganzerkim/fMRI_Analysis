#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:22:21 2020

@author: mingeon kim
"""


import numpy as np # linear algebra
import nilearn as nl
import nilearn.plotting as nlplt
import nibabel as nib
import h5py
import matplotlib.pyplot as plt
from nilearn.plotting import plot_anat, show

from nilearn import datasets
atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas['maps']
# Loading atlas data stored in 'labels'
labels = atlas['labels']


import csv
# Load the functional datasets
data = '/home/bk/Desktop/bkrest/sub-ID01/func/sub-ID01_task-rest1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
confounds = '/home/bk/Desktop/bkrest/sub-ID01/func/sub-ID01_task-rest1_desc-confounds_regressors.tsv'
d = 'home/bk/Desktop/regressor.tsv'
# confounds = open(regressor, 'r', encoding='utf-8')
# rdr = csv.reader(confounds, delimiter = '\t')

print('First subject resting-state nifti image (4D) is located at: %s' %
      data)

from nilearn.input_data import NiftiMapsMasker
masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                         memory='nilearn_cache', verbose=5)

time_series = masker.fit_transform(data,
                                   confounds=confounds)

print(time_series.shape)


from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')
dd = correlation_measure.fit_transform([time_series])

correlation_matrix = dd[0, :, :]

# Display the correlation matrix
import numpy as np
from nilearn import plotting
# Mask out the major diagonal
np.fill_diagonal(correlation_matrix, 1)
plotting.plot_matrix(correlation_matrix, labels=labels, colorbar=True,
                     vmax=1.8, vmin=0.8)


from nilearn import plotting
coords = atlas.region_coords

# We threshold to keep only the 20% of edges with the highest value
# because the graph is very dense
plotting.plot_connectome(correlation_matrix, coords,
                         edge_threshold="80%", colorbar=True)

plotting.show()



view = plotting.view_connectome(correlation_matrix, coords, edge_threshold='80%')

# In a Jupyter notebook, if ``view`` is the output of a cell, it will
# be displayed below the cell
view
