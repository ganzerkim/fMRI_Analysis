# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:34:20 2020

@author: MIT-DGMIF
"""
from nilearn.datasets import fetch_neurovault
from nilearn.image import smooth_img

from nilearn.datasets import load_mni152_brain_mask
from nilearn.input_data import NiftiMasker

from nilearn import plotting

import nilearn as nl
import nilearn.plotting as nlplt
import nibabel as nib
from nilearn import image

from nilearn import datasets
from nilearn import surface

import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


import os
import pandas as pd
import numpy as np
import scipy as sp
import random
import h5py
import matplotlib.pyplot as plt
import seaborn as sns




bold = 'C:/Users/MIT-DGMIF/Desktop/S01/BOLD1/swraBOLD_4D.nii'
fmri_mask = 'C:/Users/MIT-DGMIF/Desktop/S01/Stat/mask.nii'
regressor = 'C:/Users/MIT-DGMIF/Desktop/S01/BOLD1/rp_aBOLD_4D.txt'


smri = 'E:/Dataset/trends-assessment-prediction/ch2better.nii'

bold_img = nl.image.load_img(bold)
mask_img = nl.image.load_img(fmri_mask)
sss = bold_img.get_fdata()

'''
def load_subject(filename, mask_img):
    subject_data = None
    with h5py.File(filename, 'r') as f:
        subject_data = f['SM_feature'][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0,1,2,3], [3,2,1,0])
    subject_img = nl.image.new_img_like(mask_img, subject_data, affine=mask_img.affine, copy_header=True)

    return subject_img
'''

'''
files = random.choices(os.listdir('E:/Dataset/trends-assessment-prediction/fMRI_train/'), k = 3)
for file in files:
    subject = os.path.join('E:/Dataset/trends-assessment-prediction/fMRI_train/', file)
    subject_img = load_subject(subject, mask_img)
    print("Image shape is %s" % (str(subject_img.shape)))
    num_components = subject_img.shape[-1]
'''
#subject_data = np.moveaxis(sss, [0,1,2,3], [3,2,1,0])
subject_img = nl.image.new_img_like(mask_img, sss, affine=mask_img.affine, copy_header=True)

from nilearn import datasets
atlas = datasets.fetch_atlas_msdl()
# Loading atlas image stored in 'maps'
atlas_filename = atlas['maps']
# Loading atlas data stored in 'labels'
labels = atlas['labels']

# Load the functional datasets
'''
data = datasets.fetch_development_fmri(n_subjects=1)

print('First subject resting-state nifti image (4D) is located at: %s' %
      data.func[0])

test = nl.image.load_img(data.func[0])
header = test.get_header()
print(header)
img = test.get_fdata()
my_img = subject_img.get_fdata()
'''


from nilearn.input_data import NiftiMapsMasker
masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                         memory='nilearn_cache', verbose=5)

time_series = masker.fit_transform(subject_img,
                                   confounds=regressor)


print(time_series.shape)

#%%
from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Display the correlation matrix
# Mask out the major diagonal
np.fill_diagonal(correlation_matrix, 0)
plotting.plot_matrix(correlation_matrix, labels=labels, colorbar=True,
                     vmax=0.8, vmin=-0.8)

#%%
from nilearn import plotting
coords = atlas.region_coords

# We threshold to keep only the 20% of edges with the highest value
# because the graph is very dense
plotting.plot_connectome(correlation_matrix, coords,
                         edge_threshold="80%", colorbar=True)

plotting.show()

#3D
view = plotting.view_connectome(correlation_matrix, coords, edge_threshold='80%')

# In a Jupyter notebook, if ``view`` is the output of a cell, it will
# be displayed below the cell
view
view.open_in_browser()


#%%%
from nilearn.masking import apply_mask
masked_data = apply_mask(subject_img, mask_img)

# masked_data shape is (timepoints, voxels). We can plot the first 150
# timepoints from two voxels

# And now plot a few of these

plt.figure(figsize=(7, 5))
plt.plot(masked_data[:150, :2])
plt.xlabel('Time [TRs]', fontsize=16)
plt.ylabel('Intensity', fontsize=16)
plt.xlim(0, 150)
plt.subplots_adjust(bottom=.12, top=.95, right=.95, left=.12)

show()






