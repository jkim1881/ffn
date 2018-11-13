import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    hdf_root = '/media/data_cifs/connectomics/datasets/third_party/'
    output_root = '/media/data_cifs/connectomics/ffn_inferred'
    fov = 'wide_fov'
    test_dataset_name = 'berson'
    dataset_type = 'val'
    idx = [50, 100, 150, 200, 342]
    cond_name = 'feedback_hgru_v5_3l_notemp_allbutberson_r0_55125' #'feedback_hgru_v5_3l_linfb_allbutberson_r0_363'

    # dataset_shape_list = [[250, 250, 250],
    #                       [384, 384, 300],
    #                       [1024, 1024, 75], 
    #                       [1250, 1250, 100], 
    #                       [1250, 1250, 100],
    #                       [1250, 1250, 100]]


    # LOAD VOLUME
    img_fullpath = os.path.join(hdf_root, fov, test_dataset_name, dataset_type, 'grayscale_maps.h5')
    data = h5py.File(img_fullpath, 'r')
    volume = np.array(data['raw'])

    # LOAD GT
    gt_fullpath = os.path.join(hdf_root, fov, test_dataset_name, dataset_type, 'groundtruth.h5')
    data = h5py.File(gt_fullpath, 'r')
    instances = np.array(data['stack'])

    # LOAD INFERRED MAP
    output_fullpath = os.path.join(output_root, fov, cond_name, test_dataset_name, dataset_type, '0/0/seg-0_0_0.npz')
    inference = np.load(output_fullpath)
    # ['overlaps', 'segmentation', 'request', 'origins', 'counters']
    inference = inference['segmentation']

    import skimage.feature
    import skimage.measure
    import skimage.morphology 

    print('mean ='+ str(np.mean(volume.flatten())))
    print('std ='+str(np.std(volume.flatten())))

    i=0
    for iz in idx:
        base = i*5
        i+=1
        instances_slice, _, _ = skimage.segmentation.relabel_sequential(instances[iz,:,:], offset=1)
        inference_slice, _, _ = skimage.segmentation.relabel_sequential(inference[iz,:,:], offset=1)
        plt.subplot(len(idx),5,base+1);plt.hist(volume.flatten())
        plt.subplot(len(idx),5,base+2);plt.imshow(volume[iz,:,:], cmap='gray')
        plt.subplot(len(idx),5,base+3);plt.imshow(instances_slice)
        plt.subplot(len(idx),5,base+4);plt.imshow(inference_slice)
        plt.subplot(len(idx),5,base+5);plt.hist(inference_slice.flatten())
    plt.show()
