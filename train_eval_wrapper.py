import sys
import os
import subprocess
import sys
import numpy as np

def find_all_ckpts(ckpt_root, net_cond_name):
    raw_items = os.listdir(os.path.join(ckpt_root, net_cond_name))
    items = []
    for item in raw_items:
        if (item.split('.')[0]=='model') & (item.split('.')[-1]=='meta'):
            items.append(int(item.split('.')[1].split('-')[1]))
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

    batch_size = int(sys.argv[1])

    script_root = '/home/drew/ffn'
    net_name_obj = 'convstack_3d_in' #'convstack_3d_bn' #'feedback_hgru_v5_3l_notemp' #'feedback_hgru_generic_longfb_3l_long'#'feedback_hgru_generic_longfb_3l' #'feedback_hgru_3l_dualch' #'feedback_hgru_2l'  # 'convstack_3d'
    net_name = net_name_obj
    # volumes_name_list = ['isbi2013']
    # volumes_name_list = ['neuroproof']
    # volumes_name_list = ['isbi2013',
    #                      'cremi_a',
    #                      'cremi_b',
    #                      'cremi_c',
    #                      'berson']
    # volumes_name_list = ['neuroproof',
    #                       'isbi2013',
    #                      'cremi_a',
    #                      'cremi_b',
    #                      'cremi_c']
    # volumes_name_list = ['neuroproof',
    #                      'cremi_a',
    #                      'cremi_b',
    #                      'cremi_c',
    #                      'berson']
    volumes_name_list = ['neuroproof',
                         'isbi2013',
                         'berson']
    # volumes_name_list = ['cremi_a',
    #                      'cremi_b',
    #                      'cremi_c']
    # volumes_name_list = ['berson_w_memb']
    tfrecords_name = 'allbutcremi'
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

    ckpt_ticks = 10
    with_membrane = False
    adabn = True
    eval_steps = 6000/batch_size
    move_threshold = 0.9

    print('>>>>>>>>>>>>>>>>>>>>> Dataset = ' + tfrecords_name)
    cond_name = net_name + '_' + tfrecords_name + '_r0' #+ str(irep)
    coords_fullpath = os.path.join(hdf_root, tfrecords_name, dataset_type, 'tf_record_file')

    data_string = ''
    label_string = ''
    for i, vol in enumerate(volumes_name_list):
        volume_fullpath = os.path.join(hdf_root, vol, dataset_type, 'grayscale_maps.h5')
        groundtruth_fullpath = os.path.join(hdf_root, vol, dataset_type, 'groundtruth.h5')
        if len(volumes_name_list)==1:
            partition_prefix='jk'
        else:
            partition_prefix=str(i)
        data_string += partition_prefix + ':' + volume_fullpath + ':raw'
        label_string += partition_prefix + ':' + groundtruth_fullpath + ':stack'
        if i < len(volumes_name_list)-1:
            data_string += ','
            label_string += ','

    print('>>>>>>>>>>>>>>>>>>>>> Collecting CKPTs....')
    load_from_ckpt = 'None'
    ckpt_list = find_all_ckpts(ckpt_root, cond_name)
    trimmed_list = ckpt_list[-1::-(len(ckpt_list) / (ckpt_ticks*2))][:ckpt_ticks]

    for ckpt_idx in trimmed_list:
        print('>>>>>>>>>>>>>>>>>>>>> Running....(CKPT='+str(ckpt_idx)+')')
        ckpt_path = os.path.join(cond_name, 'model.ckpt-' + str(ckpt_idx))
        train_dir = os.path.join(ckpt_root, cond_name) + '_eval'
        ## COPY CKPT AND MOVE
        if not os.path.exists(train_dir):
            os.path.makedirs(train_dir)
        from shutil import copyfile
        copyfile(ckpt_path, os.path.join(train_dir,'model.ckpt-' + str(ckpt_idx)))

        command = 'python ' + os.path.join(script_root, 'train_old_eval.py') + \
                  ' --train_coords ' + coords_fullpath + \
                  ' --data_volumes ' + data_string + \
                  ' --label_volumes ' + label_string + \
                  ' --train_dir ' + train_dir + \
                  ' --model_name '+net_name_obj+'.ConvStack3DFFNModel' + \
                  ' --model_args "{\\"depth\\": 12, \\"fov_size\\": ' + str(fov_size) + ', \\"deltas\\": ' + str(deltas) + '}"' + \
                  ' --image_mean ' + str(128) + \
                  ' --image_stddev ' + str(33) + \
                  ' --eval_steps=' + str(eval_steps) + \
                  ' --optimizer ' + 'adam' + \
                  ' --ckpt_idx ' + str(ckpt_idx) + \
                  ' --batch_size=' + str(batch_size) + \
                  ' --with_membrane=' + str(with_membrane) + \
                  ' --validation_mode=' + str(True) + \
                  ' --adabn=' + str(adabn) + \
                  ' --threshold=' + str(move_threshold)

        ############# TODO(jk): USE DATA VOLUMES FOR MULTI VOLUME TRAINING????
        subprocess.call(command, shell=True)
