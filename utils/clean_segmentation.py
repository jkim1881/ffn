"""Clean segmentations and reasign dust to nearest neighbor segment."""
import os
# import re
import h5py
import numpy as np
from skimage import measure, morphology, segmentation
from tqdm import tqdm
from scipy import stats
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy import ndimage
from utils.NeuronIds import NeuronIds


def clean_segments(segments, connectivity=2, extent=1, background=0):
    """Run segment cleaning routines.

    1. Label segments to get connected components
    2. Measure region props and sort components by area
    """
    labeled_segments = measure.label(
        segments, connectivity=connectivity, background=background)
    props = np.asarray(measure.regionprops(labeled_segments, coordinates='rc'))

    # Sort in ascending order for greedy agglomeration
    areas = np.asarray([x.area for x in props])
    sort_areas = np.argsort(areas)
    props = props[sort_areas]
    areas = areas[sort_areas]

    # Gather cluster IDs
    ids = []
    for p in props:
        ids += [
            labeled_segments[p.coords[0][0], p.coords[0][1], p.coords[0][2]]]
    ids = np.asarray(ids)
    reassign = np.where(areas < threshold)[0]
    keep = np.where(areas >= threshold)[0]
    keep_ids = []
    for k in keep:
        coords = props[k].coords
        keep_ids += [
            segments[coords[0, 0], coords[0, 1], coords[0, 2]]]
    keep_ids = np.asarray(keep_ids)
    max_vals = (np.asarray(segments.shape) - 1).repeat(1)
    min_vals = np.asarray([0]).repeat(3)
    for pidx in tqdm(reassign, total=len(reassign)):
        coords = props[pidx].coords

        # Heuristic: Reassign to segment closest to NN
        z_min_idx = np.argmin(coords, axis=0)[2]
        z_max_idx = np.argmax(coords, axis=0)[2]
        z_search_seed_min = [
            coords[z_min_idx, 0],
            coords[z_min_idx, 1],
            coords[z_min_idx, 2] - extent]
        z_search_seed_max = [
            coords[z_max_idx, 0],
            coords[z_max_idx, 1],
            coords[z_max_idx, 2] + extent]
        y_min_idx = np.argmin(coords, axis=0)[1]
        y_max_idx = np.argmax(coords, axis=0)[1]
        y_search_seed_min = [
            coords[y_min_idx, 0],
            coords[y_min_idx, 1] - extent,
            coords[y_min_idx, 2]]
        y_search_seed_max = [
            coords[y_max_idx, 0],
            coords[y_max_idx, 1] + extent,
            coords[y_max_idx, 2]]
        x_min_idx = np.argmin(coords, axis=0)[0]
        x_max_idx = np.argmax(coords, axis=0)[0]
        x_search_seed_min = [
            coords[x_min_idx, 0] - extent,
            coords[x_min_idx, 1],
            coords[x_min_idx, 2]]
        x_search_seed_max = [
            coords[x_max_idx, 0] + extent,
            coords[x_max_idx, 1],
            coords[x_max_idx, 2]]

        x_search_seed_max = np.asarray(x_search_seed_max)
        y_search_seed_max = np.asarray(y_search_seed_max)
        z_search_seed_max = np.asarray(z_search_seed_max)
        x_search_seed_min = np.asarray(x_search_seed_min)
        y_search_seed_min = np.asarray(y_search_seed_min)
        z_search_seed_min = np.asarray(z_search_seed_min)

        x_search_seed_max = np.minimum(x_search_seed_max, max_vals)
        y_search_seed_max = np.minimum(y_search_seed_max, max_vals)
        z_search_seed_max = np.minimum(z_search_seed_max, max_vals)
        x_search_seed_max = np.maximum(x_search_seed_max, min_vals)
        y_search_seed_max = np.maximum(y_search_seed_max, min_vals)
        z_search_seed_max = np.maximum(z_search_seed_max, min_vals)

        x_search_seed_min = np.minimum(x_search_seed_min, max_vals)
        y_search_seed_min = np.minimum(y_search_seed_min, max_vals)
        z_search_seed_min = np.minimum(z_search_seed_min, max_vals)
        x_search_seed_min = np.maximum(x_search_seed_min, min_vals)
        y_search_seed_min = np.maximum(y_search_seed_min, min_vals)
        z_search_seed_min = np.maximum(z_search_seed_min, min_vals)

        # Get nearest neighbors in the x/y/z
        plus_x = labeled_segments[
            x_search_seed_max[0], x_search_seed_max[1], x_search_seed_max[2]]
        minus_x = labeled_segments[
            x_search_seed_min[0], x_search_seed_min[1], x_search_seed_min[2]]
        plus_y = labeled_segments[
            y_search_seed_max[0], y_search_seed_max[1], y_search_seed_max[2]]
        minus_y = labeled_segments[
            y_search_seed_min[0], y_search_seed_min[1], y_search_seed_min[2]]
        plus_z = labeled_segments[
            z_search_seed_max[0], z_search_seed_max[1], z_search_seed_max[2]]
        minus_z = labeled_segments[
            z_search_seed_min[0], z_search_seed_min[1], z_search_seed_min[2]]
        all_labels = [
            plus_x,
            minus_x,
            plus_y,
            minus_y,
            plus_z,
            minus_z,
        ]
        all_labels = np.asarray(all_labels)

        # Get NN counts
        area_plus_x = areas[ids == all_labels[0]]
        area_minus_x = areas[ids == all_labels[1]]
        area_plus_y = areas[ids == all_labels[2]]
        area_minus_y = areas[ids == all_labels[3]]
        area_plus_z = areas[ids == all_labels[4]]
        area_minus_z = areas[ids == all_labels[5]]
        all_areas = [
            area_plus_x,
            area_minus_x,
            area_plus_y,
            area_minus_y,
            area_plus_z,
            area_minus_z,
        ]

        # Argmax across the counts
        if len(np.concatenate(all_areas)):
            biggest_seg = np.argmax(all_areas)
            swap = all_labels[biggest_seg]
        else:
            swap = 0
        for coord in coords:
            labeled_segments[coord[0], coord[1], coord[2]] = swap

    return labeled_segments, reassign, keep_ids


def load_segments(segment_name, key=None):
    """Wrapper for loading brain data."""
    if '.npy' in segment_name:
        segments = np.load(segment_name)  # In x/y/z
    elif '.nii' in segment_name:
        import nibabel as nib
        segments = nib.load(
            segment_name).get_fdata().astype(np.uint16)  # In x/y/z
    elif '.h5' in segment_name:
        segments = h5py.File(segment_name, 'r')[key].value
    elif '.npz' in segment_name:
        segments = np.load(segment_name)['segmentation']
    else:
        raise NotImplementedError
    return segments


# General settings
threshold = 1000  # 4096  min segment size
med_filt = 3
split_threshold = threshold * 2
erosion = 1
transpose = True
mode = 'reassign'  # 'remove' or 'reassign'
# volume = np.load('cube_sem_data_uint8.npy')
version = 41
run_reassign = False

# Iterative grouping settings
iterations = 10
extent = 1
connectivity = 1

# File names
fov_type = 'wide_fov'
hdf_root = os.path.join(
    '/media/data_cifs/connectomics/datasets/third_party/',
    fov_type)
gt_path = os.path.join(hdf_root, 'isbi2013/val')
gt_path = os.path.join(hdf_root, 'berson/val')
gt_label_name = os.path.join(gt_path, 'groundtruth.h5')
gt_image_name = os.path.join(gt_path, 'grayscale_maps.h5')
segment_name = '/media/data_cifs/connectomics/ffn_inferred/wide_fov/feedback_hgru_v5_3l_notemp_f_allbutcremi_r0/berson/val/1405136_2019_03_29_19_20_14_043294//0/0/seg-0_0_0.npz'
# segment_name = os.path.join(
#     '/media/data_cifs/connectomics/ffn_inferred',
#     fov_type,
#     'feedback_hgru_v5_3l_notemp_f_allbutfib_r0',
#     'berson',
#     'val',
#     'berson_cv_09_05_delta_14_14_2/982686/0/0/seg-0_0_0.npz')  # berson_cv_09_05_delta_22_22_4:
# segment_name = os.path.join(
#     '/media/data_cifs/connectomics/ffn_inferred',
#     fov_type,
#     'feedback_hgru_v5_3l_notemp_f_allbutfib_r0',
#     'isbi2013',
#     'val',
#     '982686/0/0/seg-0_0_0.npz')  # berson_cv_09_05_delta_22_22_4:
# segment_name = os.path.join(
#     '/media/data_cifs/connectomics/ffn_inferred',
#     fov_type,
#     'feedback_hgru_v5_3l_notemp_f_allbutfib_r0',
#     'berson',
#     'val',
#     'berson_cv_09_05_delta_14_14_2/982686/0/0/seg-0_0_0.npz')

# Start script
segments = load_segments(segment_name)
gt_label = load_segments(gt_label_name, key='stack')
gt_image = load_segments(gt_image_name, key='raw')

if mode == 'remove':
    raise NotImplementedError
    labeled_segments = morphology.remove_small_objects(
        segments,
        min_size=threshold)
    raise NotImplementedError('Do not use remove small objects routine')
else:
    below_thresh = np.inf
    labeled_segments = np.copy(segments)
    stop = False

    # Continue running until there are no subthreshold segments
    while below_thresh > 0 and iterations > 0:  # or stop:
        labeled_segments, reassign, keep_ids = clean_segments(
            labeled_segments, extent=extent, connectivity=connectivity)
        new_thresh = len(reassign)
        print 'Iteration %s, %s below threshold' % (iterations, new_thresh)
        below_thresh = new_thresh
        iterations -= 1

    if run_reassign:
        # Reassign new IDs the ID from the original volume that modally overlap
        unique_labels = np.unique(labeled_segments)
        fixed_segs = np.zeros_like(segments)
        for k in tqdm(
                unique_labels,
                desc='Reassigning labels',
                total=len(unique_labels)):
            mask = labeled_segments == k
            m = (mask).astype(float)
            seg_vec = segments * m
            seg_vec = seg_vec[seg_vec > 0]
            modal = stats.mode(seg_vec)[0][0]
            labeled_segments[mask] = modal

    # Iterate the IDs of discontinuous suprathreshold segments
    new_segs = np.unique(labeled_segments)
    counter = np.max(labeled_segments) + 1
    copy_labels = deepcopy(labeled_segments)
    for s in tqdm(
            new_segs,
            total=len(new_segs),
            desc='Fixing discontinuous'):
        selected_seg = measure.label(
            copy_labels == s, connectivity=connectivity, background=0)
        props = np.asarray(measure.regionprops(
            selected_seg, coordinates='rc'))
        if len(props) > 1:

            # Sort in ascending order
            areas = np.asarray([x.area for x in props])
            area_idx = np.argsort(areas)
            props = props[area_idx]
            props = props[:-1]  # Ignore biggest volume + bg
            for p in props:
                if p.area > split_threshold:  # Fix suprathreshold segments
                    coords = p.coords
                    for coor in coords:
                        labeled_segments[
                            coor[0], coor[1], coor[2]] = counter
                    counter += 1

    # Print out info on volumes
    old_segs = np.unique(segments)
    new_segs = np.unique(labeled_segments)
    print 'Original num labels %s' % len(old_segs)
    print 'New num labels %s' % len(new_segs)

# new_props = np.asarray(measure.regionprops(segments))

# Reassign segments to their human annotated IDs.
out_npy_name = 'filt_cleaned_predicted_segs_CURATED_v%s_max_%s.npy' % (
    version, threshold)
labeled_segments = labeled_segments.astype(np.uint16)

# Make sure background ID is correct then filter
bg = stats.mode(labeled_segments.ravel()[segments.ravel() == 0])[0][0]
labeled_segments[labeled_segments == bg] = 0
filt_labeled_segments = ndimage.median_filter(
    labeled_segments.astype(np.uint16), med_filt)
np.save(
    out_npy_name,
    filt_labeled_segments)
print len(np.unique(labeled_segments))
# segments = segmentation.relabel_from_one(filt_labeled_segments)[0]

# Throw away GT segments below threshold
new_segs = np.unique(gt_label)
thresh_gt_label = deepcopy(gt_label)
for s in tqdm(
        new_segs,
        total=len(new_segs),
        desc='Removing GT segs below thresh'):
    selected_seg = measure.label(
        thresh_gt_label == s, connectivity=connectivity, background=0)
    props = np.asarray(measure.regionprops(
        selected_seg, coordinates='rc'))
    for p in props:
        if p.area < threshold:  # Fix suprathreshold segments
            coords = p.coords
            for coor in coords:
                thresh_gt_label[
                    coor[0], coor[1], coor[2]] = 0

# Evaluate
thresh_gt_label = thresh_gt_label.astype(np.uint32)
segments = segments.astype(np.uint32)
neuron_ids_evaluation = NeuronIds(
    thresh_gt_label,
    border_threshold=10)
(voi_split, voi_merge) = neuron_ids_evaluation.voi(segments.astype(np.uint32))
adapted_rand, precision, recall = neuron_ids_evaluation.adapted_rand(
    segments)

print 'ARAND: %s, P: %s, R: %s' % (adapted_rand, precision, recall)
sel = 5
plt.subplot(121)
plt.title('Prediction')
plt.imshow(segmentation.relabel_from_one(segments[sel])[0])
plt.subplot(122)
plt.title('GT')
plt.imshow(segmentation.relabel_from_one(gt_label[sel])[0])
plt.show()
