import sys
import os
import subprocess
import numpy as np

def write_custom_request(request_txt_fullpath, hdf_fullpath, ckpt_fullpath, output_fullpath,
                         net_name,
                         fov_size, deltas,
                         image_mean=128, image_stddev=33):
    # 'validate_request_' + str(validate_jobid) + '.txt'
    file = open(request_txt_fullpath,"w")
    file.write('image { \n')
    file.write('  hdf5: "' + hdf_fullpath + ':raw" \n')
    file.write('}')
    file.write('image_mean: ' + str(image_mean) + '\n')
    file.write('image_stddev: ' + str(image_stddev) + '\n')
    file.write('seed_policy: "PolicyPeaks" \n')
    file.write('model_checkpoint_path: "' + ckpt_fullpath + '" \n')
    file.write('model_name: "'+net_name+'.ConvStack3DFFNModel" \n')
    file.write('model_args: "{\\"depth\\": 12, \\"fov_size\\": ' + str(fov_size) + ', \\"deltas\\": ' + str(deltas) + '}" \n')
    file.write('segmentation_output_dir: "' + output_fullpath + '" \n')
    file.write('inference_options { \n')
    file.write('  init_activation: 0.95 \n')
    file.write('  pad_value: 0.05 \n')
    file.write('  move_threshold: 0.9 \n')
    file.write('  min_boundary_dist { x: 1 y: 1 z: 1} \n')
    file.write('  segment_threshold: 0.6 \n')
    file.write('  min_segment_size: 1000 \n')
    file.write('} \n')
    file.close()


if __name__ == '__main__':

    num_machines = int(sys.argv[1])
    i_machine = int(sys.argv[2])

    script_root = '/home/jk/PycharmProjects/ffn/'
    request_txt_root = os.path.join(script_root, 'configs')
    hdf_root = '/media/data_cifs/connectomics/datasets/third_party/'
    ckpt_root = '/media/data_cifs/connectomics/ffn_ckpts'
    output_root = '/media/data_cifs/connectomics/ffn_inferred'

    fov_type = 'traditional_fov'
    fov_size = [33, 33, 33] #[41, 41, 21] # [33,33,33]
    deltas = [8, 8, 8] #[10, 10, 5] #[8,8,8]

    net_name = 'feedback_hgru_generic_longfb_3l'
    checkpoint_num = 46480

    train_dataset_name = 'neuroproof'
    dataset_name_list = ['cremi_a','cremi_b','cremi_c','berson','isbi2013'] # 'neuroproof'
    dataset_type = 'val' #'train'
    test_dataset_shape_list = [[520, 520, 520], #[250, 250, 250],
                               [1250, 625, 125],
                               [1250, 625, 125],
                               [1250, 625, 125],
                               [384, 192, 384],
                               [1024, 512, 100]]
    # test_dataset_shape_list = [[520, 520, 520],
    #                            [1250, 1250, 100],
    #                            [1250, 1250, 100],
    #                            [1250, 1250, 100],
    #                            [384, 384, 300],
    #                            [1024, 1024, 75]]

    num_model_repeats = 1 # 5
    image_mean = 128
    image_stddev = 33

    kth_job = 0
    for test_dataset_name, test_dataset_shape in zip(dataset_name_list, test_dataset_shape_list):
        for irep in range(num_model_repeats):
            kth_job += 1
            if np.mod(kth_job, num_machines) != i_machine and num_machines != i_machine:
                continue
            elif np.mod(kth_job, num_machines) != 0 and num_machines == i_machine:
                continue

            net_cond_name = net_name + '_' + train_dataset_name + '_r' + str(irep)
            ckpt_fullpath = os.path.join(ckpt_root, fov_type, net_cond_name, 'model.ckpt-' + str(checkpoint_num))

            request_txt_fullpath = os.path.join(request_txt_root, fov_type, net_cond_name + '_on_' + test_dataset_name + '_'+dataset_type+'.pbtxt')
            hdf_fullpath = os.path.join(hdf_root, fov_type, test_dataset_name, dataset_type, 'grayscale_maps.h5')
            output_fullpath = os.path.join(output_root, fov_type, net_cond_name + '_' + str(checkpoint_num), test_dataset_name, dataset_type)

            #ckpt_fullpath = os.path.join(ckpt_root, fov_type, 'convstack_3d_pretrained/model.ckpt-27465036')

            write_custom_request(request_txt_fullpath, hdf_fullpath, ckpt_fullpath, output_fullpath,
                                 net_name,
                                 fov_size, deltas,
                                 image_mean, image_stddev)

            command = 'python ' + os.path.join(script_root,'run_inference.py') + \
                      ' --inference_request="$(cat ' + request_txt_fullpath + ')"' +\
                      ' --bounding_box "start { x:0 y:0 z:0 } size { x:' + str(test_dataset_shape[0]) +' y:' + str(test_dataset_shape[1]) + ' z:' + str(test_dataset_shape[2]) + ' }"'
            subprocess.call(command, shell=True)

# .npz
# #
