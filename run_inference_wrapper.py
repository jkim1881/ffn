import sys
import os
import subprocess
import numpy as np
import metrics
import h5py

def write_custom_request(request_txt_fullpath, hdf_fullpath, ckpt_fullpath, output_fullpath,
                         net_name, seed_policy,
                         fov_size, deltas, move_threshold,
                         image_mean=128, image_stddev=33):
    # 'validate_request_' + str(validate_jobid) + '.txt'
    directory = os.path.split(request_txt_fullpath)[0]
    if not os.path.isdir(directory):
        os.makedirs(directory)
    file = open(request_txt_fullpath,"w+")
    file.write('image { \n')
    file.write('  hdf5: "' + hdf_fullpath + ':raw" \n')
    file.write('}')
    file.write('image_mean: ' + str(image_mean) + '\n')
    file.write('image_stddev: ' + str(image_stddev) + '\n')
    file.write('seed_policy: "'+seed_policy+'" \n')
    file.write('model_checkpoint_path: "' + ckpt_fullpath + '" \n')
    file.write('model_name: "'+net_name+'.ConvStack3DFFNModel" \n')
    file.write('model_args: "{\\"depth\\": 12, \\"fov_size\\": ' + str(fov_size) + ', \\"deltas\\": ' + str(deltas) + '}" \n')
    file.write('segmentation_output_dir: "' + output_fullpath + '" \n')
    file.write('inference_options { \n')
    file.write('  init_activation: 0.95 \n')
    file.write('  pad_value: 0.05 \n')
    file.write('  move_threshold: ' + str(move_threshold) + ' \n')
    file.write('  min_boundary_dist { x: 1 y: 1 z: 1} \n')
    file.write('  segment_threshold: 0.6 \n')
    file.write('  min_segment_size: 1000 \n')
    file.write('} \n')
    file.close()


def find_all_ckpts(ckpt_root, net_cond_name, ckpt_cap):
    raw_items = os.listdir(os.path.join(ckpt_root, net_cond_name))
    items = []
    for item in raw_items:
        if (item.split('.')[0]=='model') & (item.split('.')[-1]=='meta'):
            ckpt = int(item.split('.')[1].split('-')[1])
            if ckpt < ckpt_cap:
                items.append(ckpt)
    items.sort()
    return items


def find_checkpoint(checkpoint_num, ckpt_root, fov_type, net_cond_name, factor):
    raw_items = os.listdir(os.path.join(ckpt_root, fov_type, net_cond_name))
    items = []
    for item in raw_items:
        if (item.split('.')[0]=='model') & (item.split('.')[-1]=='meta'):
            items.append(int(item.split('.')[1].split('-')[1]))
    items.sort()
    interval = len(items)/factor

    checkpoint_num = items[checkpoint_num*interval]
    return checkpoint_num


if __name__ == '__main__':

    num_machines = int(sys.argv[1])
    i_machine = int(sys.argv[2]) ####### currently useless. later implement multi-ckpt or multi-vol pipeline

    script_root = '/home/drew/ffn/'

    net_name_obj = 'htd_cnn_3l_in' #'convstack_3d_bn' #'feedback_hgru_v5_3l_notemp' #'feedback_hgru_generic_longfb_3l_long'#'feedback_hgru_generic_longfb_3l' #'feedback_hgru_3l_dualch' #'feedback_hgru_2l'  # 'convstack_3d'
    net_name = net_name_obj
    train_tfrecords_name = 'berson3x_w_inf_memb'
    with_membrane = True
    seed_policy = 'PolicyMembranePeaks' #'PolicyPeaks'

    infer_volume_name = 'berson_w_inf_memb'
    infer_volume_type = 'train'

    # fov_type = 'traditional_fov'
    # fov_size = [33, 33, 33]
    # deltas = [8, 8, 8]
    fov_type = 'wide_fov'
    fov_size = [57, 57, 13]
    deltas = [8, 8, 3]

    hdf_root = os.path.join('/media/data_cifs/connectomics/datasets/third_party/', fov_type)
    ckpt_root = os.path.join('/media/data_cifs/connectomics/ffn_ckpts', fov_type)
    output_root = os.path.join('/media/data_cifs/connectomics/ffn_inferred', fov_type)
    request_txt_root = os.path.join(script_root, 'configs', fov_type)

    ckpt_ticks = 10
    ckpt_cap = 650000 # max number of iters from which to load ckpts
    single_ckpt = 1190936
    move_threshold = 0.9

    image_mean = 128
    image_stddev = 33

    kth_job = 0

    ### DEFINE PATHS
    net_cond_name = net_name + '_' + train_tfrecords_name + '_r0'
    request_txt_fullpath = os.path.join(request_txt_root, net_cond_name + '_inferon_' + infer_volume_name + '_' + infer_volume_type + '.pbtxt')
    hdf_fullpath = os.path.join(hdf_root, infer_volume_name, infer_volume_type, 'grayscale_maps.h5')
    gt_fullpath = os.path.join(hdf_root, infer_volume_name, infer_volume_type, 'groundtruth.h5')
    import h5py
    data = h5py.File(hdf_fullpath, 'r')
    vol_shape = data['raw'].shape
    zdim = vol_shape[0]
    xdim = vol_shape[1]
    ydim = vol_shape[2]
    data.close()

    ## COLLECT CKPTS
    print('>>>>> TRIMMING CKPS')
    print('>>>>>>>>>>>>>>>>>>>>> Collecting CKPTs....')
    if single_ckpt is None:
        ckpt_list = find_all_ckpts(ckpt_root, net_cond_name, ckpt_cap)
        trimmed_list = ckpt_list[-1::-(len(ckpt_list) / (ckpt_ticks * 2))][:ckpt_ticks]
    else:
        trimmed_list = [single_ckpt]
    print('>>>>> DONE.')
    print('>>>>> CKPTS TO USE :: '+ str(trimmed_list))

    ## LOOP OVER CKPTS AND RUN INFERENCE
    current_best_arand = 99.
    current_best_ckpt = trimmed_list[0]
    for checkpoint_num in trimmed_list:

        kth_job += 1
        if np.mod(kth_job, num_machines) != i_machine and num_machines != i_machine:
            continue
        elif np.mod(kth_job, num_machines) != 0 and num_machines == i_machine:
            continue

        ### DEFINE NAMES
        ckpt_fullpath = os.path.join(ckpt_root, net_cond_name, 'model.ckpt-' + str(checkpoint_num))
        inference_fullpath = os.path.join(output_root, net_cond_name, infer_volume_name, infer_volume_type, str(checkpoint_num))

        ### OPEN TEXT
        infer_result_txt_fullpath = os.path.join(output_root, net_cond_name, infer_volume_name, infer_volume_type, 'score.txt')
        if not os.path.exists(os.path.join(output_root, net_cond_name, infer_volume_name, infer_volume_type)):
            os.makedirs(os.path.join(output_root, net_cond_name, infer_volume_name, infer_volume_type))
        infer_result_txt = open(infer_result_txt_fullpath, "w+")
        infer_result_txt.write('>>>>>>>>>>>>>> Infer on: ' + infer_volume_name)
        infer_result_txt.write('>>>>>>>>>>>>>> Ckpt: ' + str(checkpoint_num))
        infer_result_txt.write("\n")
        infer_result_txt.close()

        ### RUN INFERENCE
        print('>>>>>>>>>>>>>> Model: ' + net_cond_name)
        print('>>>>>>>>>>>>>> Infer on: ' + infer_volume_name + ' with shape (xyz): ' + str([xdim, ydim, zdim]))
        print('>>>>>>>>>>>>>> Ckpt: ' + str(checkpoint_num))
        print('Output at: ' + inference_fullpath)
        write_custom_request(request_txt_fullpath, hdf_fullpath, ckpt_fullpath, inference_fullpath,
                             net_name, seed_policy,
                             fov_size, deltas, move_threshold,
                             image_mean, image_stddev)
        command = 'python ' + os.path.join(script_root, 'run_inference.py') + \
                  ' --inference_request="$(cat ' + request_txt_fullpath + ')"' +\
                  ' --bounding_box "start { x:0 y:0 z:0 } size { x:' + str(xdim) + ' y:' + str(ydim) + ' z:' + str(zdim) + ' }"' + \
                  ' --with_membrane=' + str(with_membrane)
        subprocess.call(command, shell=True)
        print('>>>>>>>>>>>>>> Inference finished')

        #### EVALUATE INFERRENCE
        print('>>>>>>>>>>>>>> Starting evaluation on inferred volume')
        data = h5py.File(gt_fullpath, 'r')
        gt = np.array(data['stack'])
        gt_unique = np.unique(gt)
        inference_fullpath = os.path.join(inference_fullpath, '0/0/seg-0_0_0.npz')
        seg = np.load(inference_fullpath)['segmentation']
        seg_unique = np.unique(seg)
        arand, precision, recall = metrics.adapted_rand(seg, gt, all_stats=True)

        print('gt_unique_lbls: ' + str(gt_unique.shape) + ' seg_unique_lbls: ' + str(seg_unique.shape))
        print('arand: ' + str(arand) + ', precision: ' + str(precision) + ' recall: ' + str(recall))
        infer_result_txt = open(infer_result_txt_fullpath, "a")
        infer_result_txt.write('>>>> arand: ' + str(arand) + ', precision: ' + str(precision) + ', recall: ' + str(recall))
        infer_result_txt.write("\n")
        infer_result_txt.close()

        if arand < current_best_arand:
            current_best_arand = arand
            current_best_ckpt = checkpoint_num
    print('>>>>>>>>>>>>>> Job finished. Best ckpt out of ' + len(trimmed_list) + ' ckpts is: ' + str(current_best_ckpt))
# .npz
# #
