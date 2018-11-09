import sys
import os
import subprocess
import sys
import numpy as np


if __name__ == '__main__':

    num_machines = int(sys.argv[1])
    i_machine = int(sys.argv[2])
    batch_size = int(sys.argv[3])

    script_root = '/home/jk/PycharmProjects/ffn'
    net_name_obj = 'feedback_hgru_v5_3l_notemp'#'feedback_hgru_generic_longfb_3l_long'#'feedback_hgru_generic_longfb_3l' #'feedback_hgru_3l_dualch' #'feedback_hgru_2l'  # 'convstack_3d'
    net_name = net_name_obj
    dataset_name_list = ['neuroproof',
                         'berson',
                         'isbi2013',
                         'cremi_a',
                         'cremi_b',
                         'cremi_c']
    dataset_type = 'train' #'val' #'train'

    # fov_type = 'traditional_fov'
    # fov_size = [33, 33, 33]
    # deltas = [8, 8, 8]
    # fov_type = 'flat_fov'
    # fov_size = [41, 41, 21]
    # deltas = [10, 10, 5]
    fov_type = 'wide_fov'
    fov_size = [57, 57, 13]
    deltas = [8, 8, 3]

    hdf_root = os.path.join('/media/data_cifs/connectomics/datasets/third_party/', fov_type)
    ckpt_root = os.path.join('/media/data_cifs/connectomics/ffn_ckpts', fov_type)

    load_from_ckpt='None'
    #load_from_ckpt = os.path.join(ckpt_root, net_name_obj+ '_cremi_a_cremi_c_r0', 'model.ckpt-79999')
    net_name += '_cremi_a'
    #load_from_ckpt = os.path.join(script_root, 'models/fib25/model.ckpt-27465036') # THIS FEATURE DOESNT WORK
    #load_from_ckpt = os.path.join(ckpt_root, 'ffn_berson_r0/model.ckpt-0')

    num_model_repeats = 1
    max_steps = 16*100000 #64*100000/batch_size 
    optimizer = 'adam' #'adam' #'sgd'
    image_mean = 128
    image_stddev = 33

    kth_job=0

    for dataset_name in dataset_name_list:
        for irep in range(num_model_repeats):

            kth_job += 1
            if np.mod(kth_job, num_machines) != i_machine and num_machines != i_machine:
                continue
            elif np.mod(kth_job, num_machines) != 0 and num_machines == i_machine:
                continue

            print('>>>>>>>>>>>>>>>>>>>>> Dataset = ' + dataset_name + ' Rep = ' + str(irep))
            cond_name = net_name + '_' + dataset_name + '_r' + str(irep)
            coords_fullpath = os.path.join(hdf_root, dataset_name, dataset_type, 'tf_record_file')
            groundtruth_fullpath = os.path.join(hdf_root, dataset_name, dataset_type, 'groundtruth.h5')
            volume_fullpath = os.path.join(hdf_root, dataset_name, dataset_type, 'grayscale_maps.h5')

            command = 'python ' + os.path.join(script_root, 'train.py') + \
                      ' --train_coords ' + coords_fullpath + \
                      ' --data_volumes jk:' + volume_fullpath + ':raw' + \
                      ' --label_volumes jk:' + groundtruth_fullpath + ':stack' + \
                      ' --train_dir ' + os.path.join(ckpt_root, cond_name) + \
                      ' --model_name '+net_name_obj+'.ConvStack3DFFNModel' + \
                      ' --model_args "{\\"depth\\": 12, \\"fov_size\\": ' + str(fov_size) + ', \\"deltas\\": ' + str(deltas) + '}"' + \
                      ' --image_mean ' + str(image_mean) + \
                      ' --image_stddev ' + str(image_stddev) + \
                      ' --max_steps=' + str(max_steps) + \
                      ' --optimizer ' + optimizer + \
                      ' --load_from_ckpt ' + load_from_ckpt + \
                      ' --batch_size=' + str(batch_size)


            ############# TODO(jk): USE DATA VOLUMES FOR MULTI VOLUME TRAINING????


            subprocess.call(command, shell=True)




### DATA PREPATATION


  # python compute_partitions.py \
  #   --input_volume /media/data_cifs/connectomics/datasets/third_party/public/isbi2013/train/groundtruth.h5:stack \
  #   --output_volume /media/data_cifs/connectomics/datasets/third_party/public/isbi2013/train/af.h5:af \
  #   --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  #   --lom_radius 30,30,15 \
  #   --min_size 1000

        # lom used to be 24 24 24 (xyz), min_size=10000


  # python build_coordinates.py \
  #    --partition_volumes jk:third_party/media/data_cifs/connectomics/datasets/third_party/public/XXXXXXX/train/af.h5:af \
  #    --coordinate_output /media/data_cifs/connectomics/datasets/third_party/public/XXXXXXX/train/tf_record_file \
  #    --margin 14,24,24

        # margin used to be 24 24 24 (zyx)
