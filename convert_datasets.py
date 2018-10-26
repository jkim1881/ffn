import matplotlib.pyplot as plt
from skimage import io
import os
import numpy as np 
import h5py
from scipy import ndimage
import skimage.feature
import skimage.measure
import skimage.morphology 

in_root = '/media/data_cifs/andreas/connectomics'
out_root = '/media/data_cifs/connectomics/datasets/third_party/traditional'

# ISBI2012
# name = 'isbi2012'
# path = os.path.join(in_root, 'ISBI_2012_data/train')
# files = ['train-labels.tif','train-volume.tif']
# instances = io.imread(os.path.join(path,files[0]))
# volume = io.imread(os.path.join(path,files[1]))
# print(name + ' inst: ' + str(instances.shape))
# print(name + ' vol: ' + str(volume.shape))

# for i in range(1):
# 	plt.subplot(3,1,1)
# 	plt.imshow(segments1[i,:,:])
# 	plt.subplot(3,1,2)
# 	plt.imshow(segments2[i,:,:])
# 	plt.subplot(3,1,3)
# 	plt.imshow(volume[i,:,:], cmap='gray')
# 	plt.show()

# ISBI2013
name = 'isbi2013'
path = os.path.join(in_root, 'ISBI_2013_data/train')
files = ['train-labels.tif','train-input.tif']
instances = io.imread(os.path.join(path,files[0]))
volume = io.imread(os.path.join(path,files[1]))
print(name + ' inst: ' + str(instances.shape))
print(name + ' vol: ' + str(volume.shape))

# RELABEL FROM ONE
print('before re-labeling: ' + str(len(np.unique(instances))))
# replacement_label = 0
# instances_new = np.zeros_like(instances, dtype=)
# for label in unique_labels:
# 	print('replacing label: ' + str(replacement_label))
# 	instances_new[instances==label] = replacement_label
# 	replacement_label += 1
instances_new, _, _ = skimage.segmentation.relabel_sequential(instances, offset=1)
print('after re-labeling: ' + str(len(np.unique(instances_new))))

# for i in range(1):
# 	plt.subplot(2,2,1)
# 	plt.imshow(instances[i,:,:])
# 	plt.subplot(2,2,2)
# 	plt.imshow(volume[i,:,:], cmap='gray')
# 	plt.subplot(2,2,3)
# 	plt.imshow(instances[:,:,i])
# 	plt.subplot(2,2,4)
# 	plt.imshow(volume[:,:,i], cmap='gray')
# 	plt.show()

# # WRITE TRAIN
write_dir = os.path.join(out_root,name,'train')
if not os.path.isdir(write_dir):
	os.makedirs(write_dir)
writer = h5py.File(os.path.join(write_dir,'groundtruth.h5'), 'w')
writer.create_dataset('stack', data=instances_new[:,:512,:], dtype='<i8')
writer.close()
writer = h5py.File(os.path.join(out_root,name,'train','grayscale_maps.h5'), 'w')
writer.create_dataset('raw', data=volume[:,:512,:], dtype='|u1')
writer.close()
# # WRITE VAL
write_dir = os.path.join(out_root,name,'val')
if not os.path.isdir(write_dir):
	os.makedirs(write_dir)
writer = h5py.File(os.path.join(write_dir,'groundtruth.h5'), 'w')
writer.create_dataset('stack', data=instances_new[:,512:,:], dtype='<i8')
writer.close()
writer = h5py.File(os.path.join(out_root,name,'val','grayscale_maps.h5'), 'w')
writer.create_dataset('raw', data=volume[:,512:,:], dtype='|u1')
writer.close()


# CREMI
names = ['cremi_a', 'cremi_b', 'cremi_c']
path = os.path.join(in_root, 'CREMI_data/train')
files = ['sample_A_20160501.hdf', 'sample_B_20160501.hdf', 'sample_C_20160501.hdf']
for (file, name) in zip(files, names):
	data = h5py.File(os.path.join(path, file), 'r')
	instances = np.array(data['volumes']['labels']['neuron_ids'])
	volume = data['volumes']['raw']
	print(name + ' inst: ' + str(instances.shape))
	print(name + ' vol: ' + str(volume.shape))

	# RELABEL FROM ONE
	print('before re-labeling: ' + str(len(np.unique(instances))))
	# replacement_label = 0
	# instances_new = np.zeros_like(instances, dtype=)
	# for label in unique_labels:
	# 	print('replacing label: ' + str(replacement_label))
	# 	instances_new[instances==label] = replacement_label
	# 	replacement_label += 1
	instances_new, _, _ = skimage.segmentation.relabel_sequential(instances, offset=1)
	print('after re-labeling: ' + str(len(np.unique(instances_new))))

	# for i in range(1):
	# 	plt.subplot(1,2,1)
	# 	plt.imshow(instances[i,:,:])
	# 	plt.subplot(1,2,2)
	# 	plt.imshow(volume[i,:,:], cmap='gray')
	# 	plt.show()

	write_dir = os.path.join(out_root,name,'train')
	if not os.path.isdir(write_dir):
		os.makedirs(write_dir)
	writer = h5py.File(os.path.join(write_dir,'groundtruth.h5'), 'w')
	writer.create_dataset('stack', data=instances_new[:,:625,:], dtype='<i8')
	writer.close()
	writer = h5py.File(os.path.join(out_root,name,'train','grayscale_maps.h5'), 'w')
	writer.create_dataset('raw', data=volume[:,:625,:], dtype='|u1')
	writer.close()
	# WRITE VAL
	write_dir = os.path.join(out_root,name,'val')
	if not os.path.isdir(write_dir):
		os.makedirs(write_dir)
	writer = h5py.File(os.path.join(write_dir,'groundtruth.h5'), 'w')
	writer.create_dataset('stack', data=instances_new[:,625:,:], dtype='<i8')
	writer.close()
	writer = h5py.File(os.path.join(out_root,name,'val','grayscale_maps.h5'), 'w')
	writer.create_dataset('raw', data=volume[:,625:,:], dtype='|u1')
	writer.close()

#  "<i8, <i4 "|u1""

# BERSON
name = 'berson'
path = os.path.join(in_root, 'Berson')
file = 'updated_Berson.h5'
data = h5py.File(os.path.join(path, file), 'r')
instances = np.array(data['masks'])
volume = data['volume']
print(name + ' inst: ' + str(instances.shape))
print(name + ' vol: ' + str(volume.shape))

# RELABEL FROM ONE
print('before re-labeling: ' + str(len(np.unique(instances))))
# replacement_label = 0
# instances_new = np.zeros_like(instances, dtype=)
# for label in unique_labels:
# 	print('replacing label: ' + str(replacement_label))
# 	instances_new[instances==label] = replacement_label
# 	replacement_label += 1
instances_new, _, _ = skimage.segmentation.relabel_sequential(instances, offset=1)
print('after re-labeling: ' + str(len(np.unique(instances_new))))

# for i in range(1):
# 	plt.subplot(2,2,1)
# 	plt.imshow(instances[i,:,:])
# 	plt.subplot(2,2,2)
# 	plt.imshow(volume[i,:,:], cmap='gray')
# 	plt.subplot(2,2,3)
# 	plt.imshow(instances[:,:,i])
# 	plt.subplot(2,2,4)
# 	plt.imshow(volume[:,:,i], cmap='gray')
# 	plt.show()

# # WRITE TRAIN
write_dir = os.path.join(out_root,name,'train')
if not os.path.isdir(write_dir):
	os.makedirs(write_dir)
writer = h5py.File(os.path.join(write_dir,'groundtruth.h5'), 'w')
writer.create_dataset('stack', data=instances_new[:,:192,:], dtype='<i8')
writer.close()
writer = h5py.File(os.path.join(out_root,name,'train','grayscale_maps.h5'), 'w')
writer.create_dataset('raw', data=volume[:,:192,:], dtype='|u1')
writer.close()
# # WRITE VAL
write_dir = os.path.join(out_root,name,'val')
if not os.path.isdir(write_dir):
	os.makedirs(write_dir)
writer = h5py.File(os.path.join(write_dir,'groundtruth.h5'), 'w')
writer.create_dataset('stack', data=instances_new[:,192:,:], dtype='<i8')
writer.close()
writer = h5py.File(os.path.join(out_root,name,'val','grayscale_maps.h5'), 'w')
writer.create_dataset('raw', data=volume[:,192:,:], dtype='|u1')
writer.close()


# PREP TRAINING DATA
# 1. get partition-ized coordinates
  # python compute_partitions.py \
  #   --input_volume /media/data_cifs/connectomics/datasets/third_party/public/berson/train/groundtruth.h5:stack \
  #   --output_volume /media/data_cifs/connectomics/datasets/third_party/public/berson/train/af.h5:af \
  #   --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  #   --lom_radius 24,24,24 \
  #   --min_size 10000

# 2. generate tfrecords of coordinates from which to sample subvolumes
  # python build_coordinates.py \
  #    --partition_volumes jk:third_party/media/data_cifs/connectomics/datasets/third_party/public/berson/train/af.h5:af \
  #    --coordinate_output /media/data_cifs/connectomics/datasets/third_party/public/berson/train/tf_record_file \
  #    --margin 24,24,24