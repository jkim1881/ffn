import sys
import os
import subprocess

def write_custom_request(request_txt_fullpath, hdf_fullpath, ckpt_fullpath, output_fullpath,
                         fov_size, deltas,
                         image_mean=128, image_stddev=33):
    # 'validate_request_' + str(validate_jobid) + '.txt'
    file = open(request_txt_fullpath,"w")
    file.write('image {')
    file.write('  hdf5: "' + hdf_fullpath + ':raw"')
    file.write('}')
    file.write('image_mean: ' + image_mean)
    file.write('image_stddev: ' + image_stddev)
    file.write('seed_policy: PolicyPeaks')
    file.write('model_checkpoint_path: "' + ckpt_fullpath + '"')
    file.write('model_name: "convstack_3d.ConvStack3DFFNModel"')
    file.write('model_args: "{\\"depth\\": 12, \\"fov_size\\": ' + str(fov_size) + ', \\"deltas\\": ' + str(deltas) + '}"')
    file.write('segmentation_output_dir: "' + output_fullpath + '"')
    file.write('inference_options {')
    file.write('  init_activation: 0.95')
    file.write('  pad_value: 0.05')
    file.write('  move_threshold: 0.9')
    file.write('  min_boundary_dist { x: 1 y: 1 z: 1}')
    file.write('  segment_threshold: 0.6')
    file.write('  min_segment_size: 1000')
    file.write('}')
    file.close()


if __name__ == '__main__':

    script_root = '/home/jk/PycharmProjects/ffn/'
    request_txt_root = os.path.join(script_root, 'configs')
    hdf_root = '/media/data_cifs/connectomics/datasets/third_party/public/'
    ckpt_root = '/media/data_cifs/connectomics/ffn_ckpts'
    output_root = '/media/data_cifs/connectomics/ffn_inferred'

    net_name = 'ffn'
    fov_size = [41, 41, 21] # [33,33,33]
    deltas = [10, 10, 5] #[8,8,8]
    dataset_name_list = ['berson','isbi2013','cremi_a','cremi_b','cremi_c']
    dataset_shape_list = [[384,384,84], [1024, 1024, 25], [1250, 1250, 25], [1250, 1250, 25], [1250, 1250, 25]] # x, y, z. DONT KNOW HOW ITS USED

    num_model_repeats = 5
    image_mean = 128
    image_stddev = 33

    for trian_dataset_name in dataset_name_list:
        for test_dataset_name in dataset_name_list:
            for irep in range(num_model_repeats):
                cond_name = net_name + '_' + trian_dataset_name + '_r' + str(irep)
                request_txt_fullpath = os.path.join(request_txt_root, cond_name + '_on_' + test_dataset_name + '.pbtxt')
                hdf_fullpath = os.path.join(hdf_root, test_dataset_name, 'val/grayscale_maps.h5')
                ckpt_fullpath = os.path.join(ckpt_root, cond_name, 'ckpt ???????????????????')
                output_fullpath = os.path.join(output_root, cond_name, test_dataset_name)

                write_custom_request(request_txt_fullpath, hdf_fullpath, ckpt_fullpath, output_fullpath,
                                    image_mean, image_stddev)
                command = 'python ' + os.path.join(script_root,'run_inference.py') + \
                          ' --inference_request="$(cat ' + request_txt_fullpath + ')"' +\
                          ' --bounding_box "start { x:0 y:0 z:0 } size { x:250 y:250 z:250 }"'
                subprocess.call(command)


