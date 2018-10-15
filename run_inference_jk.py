import sys
import os
import subprocess

def write_custom_request(request_txt_fullpath, hdf_fullpath, ckpt_fullpath, output_fullpath,
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
    file.write('model_name: "convstack_3d.ConvStack3DFFNModel" \n')
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

    script_root = '/home/jk/PycharmProjects/ffn/'
    request_txt_root = os.path.join(script_root, 'configs')
    hdf_root = '/media/data_cifs/connectomics/datasets/third_party/traditional/'
    ckpt_root = '/media/data_cifs/connectomics/ffn_ckpts'
    output_root = '/media/data_cifs/connectomics/ffn_inferred'

    net_name = 'ffn'
    fov_size = [33, 33, 33] #[41, 41, 21] # [33,33,33]
    deltas = [8, 8, 8] #[10, 10, 5] #[8,8,8]
    dataset_name_list = ['neuroproof','cremi_a','cremi_b','cremi_c','berson','isbi2013']
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

    for trian_dataset_name in dataset_name_list:
        for test_dataset_name, test_dataset_shape in zip(dataset_name_list, test_dataset_shape_list):
            for irep in range(num_model_repeats):
                # cond_name = net_name + '_' + trian_dataset_name + '_r' + str(irep)
                cond_name = net_name + '_' + 'pretrained' ############################################################### SWITCH
                request_txt_fullpath = os.path.join(request_txt_root, cond_name + '_on_' + test_dataset_name + '.pbtxt')
                hdf_fullpath = os.path.join(hdf_root, test_dataset_name, dataset_type, 'grayscale_maps.h5')
                ckpt_fullpath = os.path.join(ckpt_root, cond_name, 'model.ckpt-27465036')
                output_fullpath = os.path.join(output_root, cond_name, test_dataset_name)

                write_custom_request(request_txt_fullpath, hdf_fullpath, ckpt_fullpath, output_fullpath,
                                     fov_size, deltas,
                                     image_mean, image_stddev)

                command = 'python ' + os.path.join(script_root,'run_inference.py') + \
                          ' --inference_request="$(cat ' + request_txt_fullpath + ')"' +\
                          ' --bounding_box "start { x:0 y:0 z:0 } size { x:' + str(test_dataset_shape[0]) +' y:' + str(test_dataset_shape[1]) + ' z:' + str(test_dataset_shape[2]) + ' }"'
                subprocess.call(command, shell=True)

# .npz
# #
