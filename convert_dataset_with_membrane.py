import matplotlib.pyplot as plt
from skimage import io
import os
import numpy as np
import h5py
from scipy import ndimage
import skimage.feature
import skimage.measure
import skimage.morphology
import sys

in_root = '/media/data_cifs/andreas/connectomics'
out_root = '/media/data_cifs/connectomics/datasets/third_party/wide_fov'

# BERSON 384 384 384
fullpath = '/media/data_cifs/connectomics/datasets/berson.npz'
data = np.load(fullpath)
membrane = np.expand_dimes(1 - data['label'], axis=3)
volume = np.expand_dimes(data['volume'], axis=3)
volume_n_membrane = np.concatenate([volume, membrane], axis=3)

print(name + ' vol: ' + str(volume.shape))

write_dir = os.path.join(out_root,name,'train')
if not os.path.isdir(write_dir):
    os.makedirs(write_dir)
writer = h5py.File(os.path.join(out_root,name,'train','grayscale_n_memb_maps.h5'), 'w')
writer.create_dataset('raw', data=volume_n_membrane, dtype='|u1')
writer.close()

# # WRITE VAL
# write_dir = os.path.join(out_root,name,'val')
# if not os.path.isdir(write_dir):
# 	os.makedirs(write_dir)
# writer = h5py.File(os.path.join(write_dir,'groundtruth.h5'), 'w')
# writer.create_dataset('stack', data=instances_new[:,192:,:], dtype='<i8')
# writer.close()
# writer = h5py.File(os.path.join(out_root,name,'val','grayscale_maps.h5'), 'w')
# writer.create_dataset('raw', data=volume[:,192:,:], dtype='|u1')
# writer.close()
