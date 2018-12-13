import sys
import os
import subprocess
import sys
import numpy as np
import train_functional

if __name__ == '__main__':

    batch_size = int(sys.argv[1])

    script_root = '/home/jk/PycharmProjects/ffn' #'/home/drew/ffn'
    net_name_obj = 'convstack_3d' #'feedback_hgru_v5_3l_notemp' #'feedback_hgru_v5_3l_linfb' #'feedback_hgru_generic_longfb_3l_long'#'feedback_hgru_generic_longfb_3l' #'feedback_hgru_3l_dualch' #'feedback_hgru_2l'  # 'convstack_3d'
    net_name = net_name_obj
    # volumes_name_list = ['neuroproof',
    #                      'isbi2013',
    #                      'cremi_a',
    #                      'cremi_b',
    #                      'cremi_c']
    volumes_name_list = ['berson_w_memb']
    tfrecords_name = 'berson_w_memb'
    dataset_type = 'train' #'val' #'train'
    with_membrane = True

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

    load_from_ckpt = 'None'
    #load_from_ckpt = os.path.join(script_root, 'models/fib25/model.ckpt-27465036') # THIS FEATURE DOESNT WORK
    #load_from_ckpt = os.path.join(ckpt_root, 'ffn_berson_r0/model.ckpt-0')

    num_model_repeats = 1
    max_steps = 16*400000/batch_size
    optimizer = 'adam' #'adam' #'sgd'
    image_mean = 128
    image_stddev = 33

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


    # command = 'python ' + os.path.join(script_root, 'train.py') + \
    #           ' --train_coords ' + coords_fullpath + \
    #           data_string + \
    #           label_string + \
    #           ' --train_dir ' + os.path.join(ckpt_root, cond_name) + \
    #           ' --model_name '+net_name_obj+'.ConvStack3DFFNModel' + \
    #           ' --model_args "{\\"depth\\": 12, \\"fov_size\\": ' + str(fov_size) + ', \\"deltas\\": ' + str(deltas) + '}"' + \
    #           ' --image_mean ' + str(image_mean) + \
    #           ' --image_stddev ' + str(image_stddev) + \
    #           ' --max_steps=' + str(max_steps) + \
    #           ' --optimizer ' + optimizer + \
    #           ' --load_from_ckpt ' + load_from_ckpt + \
    #           ' --batch_size=' + str(batch_size) + \
    #           ' --with_membrane=' + str(with_membrane)
    # ############# TODO(jk): USE DATA VOLUMES FOR MULTI VOLUME TRAINING????
    # subprocess.call(command, shell=True)

    import tensorflow as tf
    with tf.Graph().as_default():
        with tf.device(tf.train.replica_device_setter(0, merge_devices=True)):
        # with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks, merge_devices=True)):
            # SET UP TRAIN MODEL
            print('>>>>>>>>>>>>>>>>>>>>>>SET UP TRAIN MODEL')
            import logging
            from ffn.training.import_util import import_symbol
            import time
            import random
            import json
            TA = train_functional.TrainArgs(train_coords=coords_fullpath,
                                            data_volumes=data_string,
                                            label_volumes=label_string,
                                            train_dir=os.path.join(ckpt_root, cond_name),
                                            model_name=net_name_obj+'.ConvStack3DFFNModel',
                                            model_args='{"depth": 12, "fov_size": ' + str(fov_size) + ', "deltas": ' + str(deltas) + '}',
                                            image_mean=image_mean,
                                            image_stddev=image_stddev,
                                            max_steps=max_steps,
                                            optimizer=optimizer,
                                            load_from_ckpt=load_from_ckpt,
                                            batch_size=batch_size,
                                            with_membrane=with_membrane)
            model_class = import_symbol(TA.model_name)
            seed = int(time.time() + TA.task * 3600 * 24)
            logging.info('Random seed: %r', seed)
            random.seed(seed)
            eval_tracker, model, secs, load_data_ops, summary_writer, merge_summaries_op = \
                train_functional.build_train_graph(model_class, TA, save_ckpt=False,
                                                   **json.loads(TA.model_args))

            # START SESSION
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.25)
            scaffold = tf.train.Scaffold(saver=saver)
            sess = tf.train.MonitoredTrainingSession(
                master=TA.master,
                is_chief=(TA.task == 0),
                save_summaries_steps=30,
                save_checkpoint_secs=secs,  # 10000/FLAGS.batch_size, #save_checkpoint_steps=10000/FLAGS.batch_size,
                config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True),
                checkpoint_dir=TA.train_dir,
                scaffold=scaffold)

            # START TRAINING
            print('>>>>>>>>>>>>>>>>>>>>>>START TRAINING')
            sess = train_functional.train_ffn(
                TA, eval_tracker, model, sess, load_data_ops, summary_writer, merge_summaries_op)

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
