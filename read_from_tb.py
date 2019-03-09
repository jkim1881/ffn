import os
import numpy as np
import tensorflow as tf

gs_folder = '/media/data_cifs/connectomics/ffn_ckpts'
# tbs = ['wide_fov/htd_cnn_3l_berson3x_w_inf_memb_r0/events.out.tfevents.1551181357.serrep2',
#        'wide_fov/htd_cnn_3l_in_berson3x_w_inf_memb_r0/events.out.tfevents.1551181470.serrep2',
#        'wide_fov/htd_cnn_3l_inplace_berson3x_w_inf_memb_r0/events.out.tfevents.1551181417.serrep2',
#        'wide_fov/htd_cnn_3l_trainablestat_berson3x_w_inf_memb_r0/events.out.tfevents.1551181544.serrep2',
#        'wide_fov/htd_cnn_3l_newrf_berson3x_w_inf_memb_r0/events.out.tfevents.1551181593.serrep2',
#        'ultrawide_fov/htd_cnn_3l_in_berson3x_w_inf_memb_r0/events.out.tfevents.1551312783.serrep6']
tbs = ['wide_fov/htd_cnn_3l_allbutberson_r0/events.out.tfevents.1551312791.p1',
       'wide_fov/htd_cnn_3l_in_allbutberson_r0/events.out.tfevents.1551312815.p1',
       'wide_fov/htd_cnn_3l_inplace_allbutberson_r0/events.out.tfevents.1551312862.p1',
       'wide_fov/htd_cnn_3l_trainablestat_allbutberson_r0/events.out.tfevents.1551313197.p1',
       'wide_fov/htd_cnn_3l_newrf_allbutberson_r0/events.out.tfevents.1551312982.p1',
       'ultrawide_fov/htd_cnn_3l_in_allbutberson_r0/events.out.tfevents.1551453166.serrep6']
names = ['3l',
         'in',
         'inplace',
         'trainablestat',
         'newrf',
         'ultrawide']
batches = [8,
           7,
           8,
           7,
           2,
           7]

path = [os.path.join(gs_folder, x) for x in tbs]
step, step_sc, accuracy, precision, recall = [], [], [], [], []
if isinstance(path, list):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,14))
    for imodel, p in enumerate(path):
        step.append([])
        accuracy.append([])
        precision.append([])
        recall.append([])
        step_sc.append([])
        for e in tf.train.summary_iterator(p):
            step[imodel].append(int(e.step))
            step_sc[imodel].append(int(e.step)*batches[imodel])
            for v in e.summary.value:
                if v.tag == 'eval/accuracy':
                    accuracy[imodel].append(v.simple_value)
                elif v.tag == 'eval/precision':
                    precision[imodel].append(v.simple_value)
                elif v.tag == 'eval/recall':
                    recall[imodel].append(v.simple_value)
        print('Model: ' + names[imodel] + ', Missing recall: ' + str(len(step_sc[imodel]) - len(recall[imodel])))
        print('Model: ' + names[imodel] + ', Missing accuracy: ' + str(len(step_sc[imodel]) - len(accuracy[imodel])))
        plt.subplot(221);
        plt.plot(step_sc[imodel][:len(recall[imodel])], recall[imodel]);plt.title('recall')
        plt.subplot(222);
        plt.plot(step_sc[imodel][:len(accuracy[imodel])], accuracy[imodel]);plt.title('accuracy')
        plt.subplot(223);
        plt.plot(step[imodel][:len(recall[imodel])], recall[imodel]);plt.title('recall')
        plt.subplot(224);
        plt.plot(step[imodel][:len(accuracy[imodel])], accuracy[imodel], label=names[imodel]);plt.title('accuracy')
    plt.legend()
    plt.savefig(os.path.join(gs_folder,'curves.pdf'))
    #plt.show()

else:
    for e in tf.train.summary_iterator(path):
        step += [int(e.step)]
        for v in e.summary.value:
            if v.tag == 'eval/accuracy':
                accuracy += [v.simple_value]
            elif v.tag == 'eval/precision':
                precision += [v.simple_value]
            elif v.tag == 'eval/recall':
                recall += [v.simple_value]

import ipdb;ipdb.set_trace()

local_file = 'tb_scrape.npz'
np.savez(
    local_file,
    path=path,
    ex_sec=ex_sec,
    loss_1=loss_1,
    loss_2=loss_2,
    top_1_accuracy=t1,
    top_5_accuracy=t5)
os.system('gsutil cp %s %s' % (local_file, gs_folder))
print('Saved to %s' % os.path.join(gs_folder, local_file))




###########

# list your tensorboards here `tbs = ['events.out.tfevents.1537103269.resnet-vm']`
# put an ipdb above this line                 `if v.tag == 'examples/sec':` to see what the v.tags are
# for you it's gonna be like pixel_acc, etc
# then change the np.savez at the bottom of course, but this at least shows how to read from tensorboards