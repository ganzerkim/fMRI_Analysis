# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np # linear algebra
import nilearn as nl
import nilearn.plotting as nlplt
import nibabel as nib
import h5py
import matplotlib.pyplot as plt
from nilearn.plotting import plot_anat, show

mask_filename = '/home/bk/Desktop/bkrest/sub-ID01/func/sub-ID01_task-rest1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
subject_filename = '/home/bk/Desktop/bkrest/sub-ID01/func/sub-ID01_task-rest1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
spm_filename = '/home/bk/Desktop/bkrest/sub-ID01/func/sub-ID01_task-rest1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
smri_filename = '/home/bk/Desktop/bkrest/sub-ID01/anat/sub-ID01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
mask_niimg = nl.image.load_img(mask_filename)

def load_subject(filename, mask_niimg):
    """
    Load a subject saved in .mat format with
        the version 7.3 flag. Return the subject
        niimg, using a mask niimg as a template
        for nifti headers.
        
    Args:
        filename    <str>            the .mat filename for the subject data
        mask_niimg  niimg object     the mask niimg object used for nifti headers
    """
    subject_data = None
    with h5py.File(subject_filename, 'r') as f:
        subject_data = f['SM_feature'][()]
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0,1,2,3], [3,2,1,0])
    subject_niimg = nl.image.new_img_like(mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)
    
    return subject_niimg

subject_niimg = load_subject(subject_filename, mask_niimg)

func_filenames = subject_filename  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      func_filenames)  # 4D data


from nilearn.decomposition import CanICA

canica = CanICA(n_components=20,
                memory="nilearn_cache", memory_level=2,
                verbose=10,
                mask_strategy='template',
                random_state=0)
canica.fit(func_filenames)

# Retrieve the independent components in brain space. Directly
# accesible through attribute `components_img_`.
canica_components_img = canica.components_img_
# components_img is a Nifti Image object, and can be saved to a file with
# the following line:
canica_components_img.to_filename('/home/bk/Desktop/bkrest/canica_resting_state.nii.gz')


from nilearn.plotting import plot_prob_atlas

# Plot all ICA components together
plot_prob_atlas(canica_components_img, title='All ICA components')


from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map

for i, cur_img in enumerate(iter_img(canica_components_img)):
    plot_stat_map(cur_img, display_mode="z", title="IC %d" % i,
                  cut_coords=1, colorbar=False)


from nilearn.decomposition import DictLearning

dict_learning = DictLearning(n_components=20,
                             memory="nilearn_cache", memory_level=2,
                             verbose=1,
                             random_state=0,
                             n_epochs=1,
                             mask_strategy='template')

print('[Example] Fitting dicitonary learning model')
dict_learning.fit(func_filenames)
print('[Example] Saving results')
# Grab extracted components umasked back to Nifti image.
# Note: For older versions, less than 0.4.1. components_img_
# is not implemented. See Note section above for details.
dictlearning_components_img = dict_learning.components_img_
dictlearning_components_img.to_filename('/home/bk/Desktop/bkrest/dictionary_learning_resting_state.nii.gz')

plot_prob_atlas(dictlearning_components_img,
                title='All DictLearning components')


for i, cur_img in enumerate(iter_img(dictlearning_components_img)):
    plot_stat_map(cur_img, display_mode="z", title="Comp %d" % i,
                  cut_coords=1, colorbar=False)

show()


from nilearn.regions import RegionExtractor

extractor = RegionExtractor(dictlearning_components_img, threshold=0.5,
                            thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            standardize=True, min_region_size=1350)

# Just call fit() to process for regions extraction
extractor.fit(func_filenames)

# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_
# Each region index is stored in index_
regions_index = extractor.index_
# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]

# Visualization of region extraction results
title = ('%d regions are extracted from %d components.'
         '\nEach separate color of region indicates extracted region'
         % (n_regions_extracted, 8))
plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                         title=title)


from nilearn.connectome import ConnectivityMeasure

correlations = []
# Initializing ConnectivityMeasure object with kind='correlation'
connectome_measure = ConnectivityMeasure(kind='correlation')
for filename, confound in zip(func_filenames, confounds):
    # call transform from RegionExtractor object to extract timeseries signals
    timeseries_each_subject = extractor.transform(filename, confounds=confound)
    # call fit_transform from ConnectivityMeasure object
    correlation = connectome_measure.fit_transform([timeseries_each_subject])
    # saving each subject correlation to correlations
    correlations.append(correlation)

# Mean of all correlations

mean_correlations = np.mean(correlations, axis=0).reshape(n_regions_extracted,
                                                          n_regions_extracted)

title = 'Correlation between %d regions' % n_regions_extracted

# First plot the matrix
display = plotting.plot_matrix(mean_correlations, vmax=1, vmin=-1,
                               colorbar=True, title=title)

# Then find the center of the regions and plot a connectome
regions_img = regions_extracted_img
coords_connectome = plotting.find_probabilistic_atlas_cut_coords(regions_img)

plotting.plot_connectome(mean_correlations, coords_connectome,
                         edge_threshold='90%', title=title)













