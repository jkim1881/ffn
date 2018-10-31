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


def find_checkpoint(checkpoint_num, ckpt_root, fov_type, net_cond_name):
    if checkpoint_num >0:
        return checkpoint_num
    else:
        raw_items = os.listdir(os.path.join(ckpt_root, fov_type, net_cond_name))
        items = []
        for item in raw_items:
            if (item.split('.')[0]=='model') & (item.split('.')[-1]=='meta'):
                items.append(int(item.split('.')[1].split('-')[1]))
        items.sort()
        interval = len(items)/10

        checkpoint_num = items[checkpoint_num*interval - 1]
        return checkpoint_num


if __name__ == '__main__':

    script_root = '/home/jk/PycharmProjects/ffn/'
    request_txt_root = os.path.join(script_root, 'configs')
    hdf_root = '/media/data_cifs/connectomics/datasets/third_party/'
    ckpt_root = '/media/data_cifs/connectomics/ffn_ckpts'
    output_root = '/media/data_cifs/connectomics/ffn_inferred'

    fov_type = 'traditional_fov'
    fov_size = [33, 33, 33] #[41, 41, 21] # [33,33,33]
    deltas = [8, 8, 8] #[10, 10, 5] #[8,8,8]

    net_name = 'convstack_3d_provided'
    checkpoint_num = -1 # 0 if last -k for the last (k/10) ckpts

    train_dataset_name = 'neuroproof'
    dataset_name_list = ['berson','isbi2013', 'cremi_a','cremi_b','cremi_c']
    dataset_type = 'val' #'train'
    test_dataset_shape_list = [[384, 192, 384],
                               [1024, 512, 100],
                               [1250, 625, 125],
                               [1250, 625, 125],
                               [1250, 625, 125]]
    num_model_repeats = 1 # 5
    image_mean = 128
    image_stddev = 33

    for test_dataset_name, test_dataset_shape in zip(dataset_name_list, test_dataset_shape_list):
        for irep in range(num_model_repeats):


            net_cond_name = net_name + '_' + train_dataset_name + '_r' + str(irep)
            request_txt_fullpath = os.path.join(request_txt_root, fov_type, net_cond_name + '_on_' + test_dataset_name + '_' + dataset_type + '.pbtxt')
            hdf_fullpath = os.path.join(hdf_root, fov_type, test_dataset_name, dataset_type, 'grayscale_maps.h5')

            eval_result_txt = open(os.path.join(ckpt_root, fov_type, net_cond_name, 'eval.txt'), "w")
            eval_result_txt.write('TEST DATASET: ' + test_dataset_name + ', FOV: ' + fov_type)
            eval_result_txt.write("\n")
            eval_result_txt.close()

            current_best_arand = 99.
            ickpt = 0
            while ickpt < 4:

                checkpoint_num = find_checkpoint(-ickpt, ckpt_root, fov_type, net_cond_name)
                # ckpt_fullpath = os.path.join(ckpt_root, fov_type, 'convstack_3d_pretrained/model.ckpt-27465036')
                ckpt_fullpath = os.path.join(ckpt_root, fov_type, net_cond_name, 'model.ckpt-' + str(checkpoint_num))

                output_fullpath = os.path.join(output_root, fov_type, net_cond_name + '_' + str(checkpoint_num),test_dataset_name, dataset_type)
                write_custom_request(request_txt_fullpath, hdf_fullpath, ckpt_fullpath, output_fullpath,
                                     net_name,
                                     fov_size, deltas,
                                     image_mean, image_stddev)
                command = 'python ' + os.path.join(script_root,'run_inference.py') + \
                          ' --inference_request="$(cat ' + request_txt_fullpath + ')"' +\
                          ' --bounding_box "start { x:0 y:0 z:0 } size { x:' + str(test_dataset_shape[0]) +' y:' + str(test_dataset_shape[1]) + ' z:' + str(test_dataset_shape[2]) + ' }"'
                subprocess.call(command, shell=True)


                #### EVALUATE INFERRENCE
                import metrics
                import h5py
                gt_fullpath = os.path.join(hdf_root, fov_type, test_dataset_name, dataset_type, 'groundtruth.h5')
                data = h5py.File(gt_fullpath, 'r')
                gt = np.array(data['stack'])
                gt_unique = np.unique(gt)

                inference_fullpath = os.path.join(output_fullpath, '0/0', 'seg-0_0_0.npz')

                seg = np.load(inference_fullpath)['segmentation']
                seg_unique = np.unique(seg)

                arand, precision, recall = metrics.adapted_rand(seg, gt, all_stats=True)
                print('>>>>>>>>>>>>>> Model: ' + net_cond_name)
                print('>>>>>>>>>>>>>> Tested on: ' + test_dataset_name, ' fov: ' + fov_type)
                print('gt_unique_lbls: ' + str(gt_unique.shape) + ' seg_unique_lbls: ' + str(seg_unique.shape))
                print('arand: ' + str(arand) + ', precision: ' + str(precision) + ' recall: ' + str(recall))
                eval_result_txt = open(os.path.join(ckpt_root, fov_type, net_cond_name, 'eval.txt'), "a")
                eval_result_txt.write('>> CKPT: ' + str(checkpoint_num))
                eval_result_txt.write("\n")
                eval_result_txt.write('>>>> arand: ' + str(arand) + ', precision: ' + str(precision) + ', recall: ' + str(recall))
                eval_result_txt.write("\n")
                eval_result_txt.close()

                if arand < current_best_arand:
                    current_best_arand = arand
                    ickpt += 1
                else:
                    print('Accuracy stopped improving. (new=' + str(arand) + ' vs old=' + str(current_best_arand) + ') Terminating eval.')
                    ickpt = 99

# .npz
# #
