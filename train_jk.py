import sys
import os
import subprocess


if __name__ == '__main__':

    script_root = '/home/jk/PycharmProjects/ffn'
    ckpt_root = '/media/data_cifs/connectomics/ffn_ckpts'
    net_name = 'hgru' #'ffn'
    net_name_obj = 'feedback_hgru_2l' #'convstack_3d'
    dataset_name_list = ['neuroproof',
                         'berson',
                         'isbi2013',
                         'cremi_a',
                         'cremi_b',
                         'cremi_c']
    dataset_type = 'val' #'train'

    hdf_root = '/media/data_cifs/connectomics/datasets/third_party/traditional/'
    fov_size = [33, 33, 33]
    deltas = [8, 8, 8]

    # hdf_root = '/media/data_cifs/connectomics/datasets/third_party/flat_fov/'
    # fov_size = [41, 41, 21]
    # deltas = [10, 10, 5]

    import sys
    import numpy as np
    num_machines = int(sys.argv[1])
    i_machine = int(sys.argv[2])
    batch_size = int(sys.argv[3])

    #load_from_ckpt = 'None'
    load_from_ckpt = os.path.join(ckpt_root, 'ffn_pretrained/model.ckpt-27465036') # THIS FEATURE DOESNT WORK
    #load_from_ckpt = os.path.join(ckpt_root, 'ffn_berson_r0/model.ckpt-0')
    num_model_repeats = 1
    max_steps = 64*100000/batch_size # 100K for ffn
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
                      ' --model_name convstack_3d.ConvStack3DFFNModel' + \
                      ' --model_args "{\\"depth\\": 12, \\"fov_size\\": ' + str(fov_size) + ', \\"deltas\\": ' + str(deltas) + '}"' + \
                      ' --image_mean ' + str(image_mean) + \
                      ' --image_stddev ' + str(image_stddev) + \
                      ' --max_steps=' + str(max_steps) + \
                      ' --optimizer ' + optimizer + \
                      ' --load_from_ckpt ' + load_from_ckpt + \
                      ' --batch_size=' + str(batch_size)

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