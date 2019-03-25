import sys
import os
import subprocess
import sys
import numpy as np

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

    batch_size = int(sys.argv[1])

    script_root = '/home/drew/ffn'
    net_name_obj = 'convstack_3d_bn_f' #'convstack_3d_bn' #'feedback_hgru_v5_3l_notemp' #'feedback_hgru_generic_longfb_3l_long'#'feedback_hgru_generic_longfb_3l' #'feedback_hgru_3l_dualch' #'feedback_hgru_2l'  # 'convstack_3d'
    net_name = net_name_obj

    train_tfrecords_name_list = ['allbutberson', 'allbutfib', 'allbutisbi', 'allbutcremi']
    topup_volumes_name_list_list = [['berson'],
                                   ['neuroproof'],
                                   ['isbi2013'],
                                   ['cremi_a','cremi_b','cremi_c']]
    topup_tfrecords_name_list = ['berson','neuroproof','isbi2013','cremi_abc']


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
    ckpt_cap = 680000 # max number of iters from which to load ckpts
    single_ckpt = 1190936
    use_latest = True

    verbose = False
    with_membrane = False
    validation_mode = False
    cap_gradient = 0.0
    topup_mode=True

    adabn = True
    topup_steps = 15000/batch_size # five full rounds
    move_threshold = 0.9

    for topup_vol_list, topup_tfr, train_tfr in zip(topup_volumes_name_list_list, topup_tfrecords_name_list, train_tfrecords_name_list):
        print('>>>>>>>>>>>>>>>>>>>>> Train Dataset = ' + train_tfr)
        print('>>>>>>>>>>>>>>>>>>>>> Eval Dataset = ' + topup_tfr)

        cond_name = net_name + '_' + train_tfr + '_r0'
        train_dir = os.path.join(ckpt_root, cond_name) + '_topup'
        if adabn:
            train_dir += '_ada'
        coords_fullpath = os.path.join(hdf_root, topup_tfr, dataset_type, 'tf_record_file')

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        data_string = ''
        label_string = ''
        for i, tvol in enumerate(topup_vol_list):
            volume_fullpath = os.path.join(hdf_root, tvol, dataset_type, 'grayscale_maps.h5')
            groundtruth_fullpath = os.path.join(hdf_root, tvol, dataset_type, 'groundtruth.h5')
            if len(topup_vol_list)==1:
                partition_prefix='jk'
            else:
                partition_prefix=str(i)
            data_string += partition_prefix + ':' + volume_fullpath + ':raw'
            label_string += partition_prefix + ':' + groundtruth_fullpath + ':stack'
            if i < len(topup_vol_list)-1:
                data_string += ','
                label_string += ','

        print('>>>>>>>>>>>>>>>>>>>>> Collecting CKPTs....')
        if use_latest:
            ckpt_list = find_all_ckpts(ckpt_root, cond_name, ckpt_cap)
            trimmed_list = [ckpt_list[-1]]
        else:
            if single_ckpt is None:
                trimmed_list = ckpt_list[-1::-(len(ckpt_list) / (ckpt_ticks * 2))][:ckpt_ticks]
            else:
                trimmed_list = [single_ckpt]

        for ckpt_idx in trimmed_list:
            print('>>>>>>>>>>>>>>>>>>>>> Running....(CKPT='+str(ckpt_idx)+')')
            ckpt_path = os.path.join(ckpt_root, cond_name, 'model.ckpt-' + str(ckpt_idx) + '*')
            ## COPY CKPT AND MOVE
            import glob
            import shutil
            for file in glob.glob(ckpt_path):
                print(file)
                shutil.copy(file, train_dir)

            command = 'python ' + os.path.join(script_root, 'train_old_eval.py') + \
                      ' --train_coords ' + coords_fullpath + \
                      ' --data_volumes ' + data_string + \
                      ' --label_volumes ' + label_string + \
                      ' --train_dir ' + train_dir + \
                      ' --model_name '+net_name_obj+'.ConvStack3DFFNModel' + \
                      ' --model_args "{\\"depth\\": 12, \\"fov_size\\": ' + str(fov_size) + ', \\"deltas\\": ' + str(deltas) + '}"' + \
                      ' --image_mean ' + str(128) + \
                      ' --image_stddev ' + str(33) + \
                      ' --eval_steps=' + str(topup_steps) + \
                      ' --optimizer ' + 'adam' + \
                      ' --ckpt_idx ' + str(ckpt_idx) + \
                      ' --batch_size=' + str(batch_size) + \
                      ' --with_membrane=' + str(with_membrane) + \
                      ' --validation_mode=' + str(validation_mode) + \
                      ' --cap_gradient=' + str(cap_gradient) + \
                      ' --topup_mode=' + str(topup_mode) + \
                      ' --progress_verbose=' + str(verbose) + \
                      ' --adabn=' + str(adabn) + \
                      ' --threshold=' + str(move_threshold)

            ############# TODO(jk): USE DATA VOLUMES FOR MULTI VOLUME TRAINING????
            subprocess.call(command, shell=True)
