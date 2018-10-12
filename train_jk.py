import sys
import os
import subprocess


if __name__ == '__main__':

    script_root = '/home/jk/PycharmProjects/ffn'
    hdf_root = '/media/data_cifs/connectomics/datasets/third_party/public/'
    ckpt_root = '/media/data_cifs/connectomics/ffn_ckpts'

    net_name = 'ffn'
    fov_size = [41, 41, 21] # [33,33,33] XYZ
    deltas = [10, 10, 5] #[8,8,8] XYZ
    dataset_name_list = ['berson',
                         'isbi2013',
                         'cremi_a',
                         'cremi_b',
                         'cremi_c']
    dataset_shape_list = [[384, 384, 300],
                          [1024, 1024, 75],
                          [1250, 1250, 100],
                          [1250, 1250, 100],
                          [1250, 1250, 100]] # x, y, z. DONT KNOW HOW ITS USED
    load_from_ckpt = None
    #load_from_ckpt = os.path.join(script_root, 'models/fib25/model.ckpt-27465036') # None
    #load_from_ckpt = os.path.join(ckpt_root, 'ffn_berson_r0/model.ckpt-0')
    num_model_repeats = 5
    max_steps = 100000
    optimizer = 'adam' #'sgd'
    batch_size = 64
    image_mean = 128
    image_stddev = 33

    for dataset_name in dataset_name_list:
        for irep in range(num_model_repeats):
            print('>>>>>>>>>>>>>>>>>>>>> Dataset = ' + dataset_name + ' Rep = ' + str(irep))
            cond_name = net_name + '_' + dataset_name + '_r' + str(irep)
            coords_fullpath = os.path.join(hdf_root, dataset_name, 'train/tf_record_file')
            groundtruth_fullpath = os.path.join(hdf_root, dataset_name, 'train/groundtruth.h5')
            volume_fullpath = os.path.join(hdf_root, dataset_name, 'train/grayscale_maps.h5')

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
                      ' --load_from_ckpt ' + str(load_from_ckpt) + \
                      ' --batch_size=' + str(batch_size)

            subprocess.call(command, shell=True)




### DATA PREPATATION


  # python compute_partitions.py \
  #   --input_volume /media/data_cifs/connectomics/datasets/third_party/public/isbi2013/train/groundtruth.h5:stack \
  #   --output_volume /media/data_cifs/connectomics/datasets/third_party/public/isbi2013/train/af.h5:af \
  #   --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  #   --lom_radius 30,30,15 \
  #   --min_size 1000

        # lom used to be 24 24 24 (xyz)


  # python build_coordinates.py \
  #    --partition_volumes jk:third_party/media/data_cifs/connectomics/datasets/third_party/public/XXXXXXX/train/af.h5:af \
  #    --coordinate_output /media/data_cifs/connectomics/datasets/third_party/public/XXXXXXX/train/tf_record_file \
  #    --margin 14,24,24

        # margin used to be 24 24 24 (zyx)