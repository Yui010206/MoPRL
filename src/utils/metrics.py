import torch
import numpy as np
import os

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from utils.normalize import normalize_anomaly_score
from datasets.dataset_path import *
from utils.load_save import load_json

def recover_poses(local_pose,global_box,dataset):
    # local_pose: (B,17*T,2)
    # global_box: (B,2*T,2)
    t = int(global_box.shape[1]/2)
    scale_factor = 100
    if dataset=='ShanghaiTech':
        frame_width = 856
        frame_height = 480

    local_pose_split = torch.chunk(local_pose,t,dim=1)
    global_box_split = torch.chunk(global_box,t,dim=1)

    assert len(local_pose_split) == len(global_box_split)

    recovered_poses = []

    for i in range(len(local_pose_split)):
        pose = local_pose_split[i] #(B,17,2)
        #print(pose.shape)
        b_min,b_max = global_box_split[i][:,0,:],global_box_split[i][:,1,:] #(B,2)

        x_min ,y_min = b_min[:,0]/scale_factor*frame_width, b_min[:,1]/scale_factor*frame_height
        x_max ,y_max = b_max[:,0]/scale_factor*frame_width, b_max[:,1]/scale_factor*frame_height

        w, h = x_max-x_min, y_max-y_min
        x_c , y_c = (x_max+x_min)/2, (y_max+y_min)/2

        x = ((pose[:,:,0]/scale_factor)*w[0] + x_c[0]).unsqueeze(-1)
        y = ((pose[:,:,1]/scale_factor)*h[0] + y_c[0]).unsqueeze(-1)
        recovered_pose = torch.cat([x,y],dim=-1)
        #print(recovered_pose.shape)
        recovered_poses.append(recovered_pose)

    recovered_poses = torch.cat(recovered_poses,dim=1)

    return recovered_poses

def L1_err(gt,pred,weight=None):

    err = torch.norm((gt - pred), p=1, dim=-1)

    if weight is not None:
        err = err*weight

    err = err.mean(dim=-1)

    return err.tolist()

def L2_err(gt,pred,weight=None):

    err = torch.norm(gt - pred, p=2, dim=-1)

    if weight is not None:
        err = err*weight

    err = err.mean(dim=-1)
    
    return err.tolist()

def compute_auc(rec_errs,pred_errs,meta,duration,k,dataset,cal_type='sum'):

    if dataset == 'UCF_crime':
        fps_dic = load_json(os.path.join(UCF_crime_Dir, 'fps_stats.json'))
        frames_dic = load_json(os.path.join(UCF_crime_Dir, 'frame_stats.json'))
        label_dic = {}
        with open(os.path.join(UCF_crime_Dir, 'Temporal_Anomaly_Annotation_for_Testing_Videos.txt'), 'r') as fin:
            for line in fin.readlines():
                spl = line.strip().split('  ')
                label_dic[spl[0]] = [spl[2], spl[3], spl[4], spl[5]]

        compute_dict = {}

        for rec_err, pred_err, name in zip(rec_errs,pred_errs,meta):

            scene ,frame = name.split('.mp4_')
            scene = scene + '.mp4'

            if cal_type=='sum':
                err = (1-k/10)*rec_err+(k/10)*pred_err
            elif cal_type=='max':
                err = max(rec_err,pred_err)

            if scene not in compute_dict:
                compute_dict[scene] = {}
            if int(frame) not in compute_dict[scene]:
                compute_dict[scene][int(frame)] = [err]
            else:
                compute_dict[scene][int(frame)].append(err)

        max_err_dict = {}
        all_label = []
        all_score = []
        all_nor_score = []

        for scene in compute_dict:
            max_err_dict[scene] = []
            frames = compute_dict[scene].keys()
            sorted_frames = list(sorted(frames))

            label = np.zeros(int(frames_dic[scene]*5/fps_dic[scene]))

            if int(label_dic[scene][0]) != -1 and int(label_dic[scene][1]) != -1:
                s1 = int(float(label_dic[scene][0])*5/float(fps_dic[scene]))
                f1 = int(float(label_dic[scene][1])*5/float(fps_dic[scene]))
                label[s1: f1] = 1

            if int(label_dic[scene][2]) != -1 and int(label_dic[scene][3]) != -1:
                s2 = int(float(label_dic[scene][2])*5/float(fps_dic[scene]))
                f2 = int(float(label_dic[scene][3])*5/float(fps_dic[scene]))
                label[s2: f2] = 1

            label = label.tolist()

            num_frame = len(label)
            anchor = 0
            for i in range(num_frame):
                if i > sorted_frames[-1]:
                    max_err_dict[scene].append(0)
                elif int(sorted_frames[anchor]) == i:
                    max_rec = max(compute_dict[scene][sorted_frames[anchor]])
                    max_err_dict[scene].append(max_rec)
                    anchor += 1
                else:
                    max_err_dict[scene].append(0)

            ano_score = max_err_dict[scene]
            all_label.extend(label[duration:])
            all_score.extend(ano_score[duration:])
            all_nor_score.extend(normalize_anomaly_score(ano_score)[duration:])

    else:
        compute_dict = {}

        for rec_err, pred_err, name in zip(rec_errs,pred_errs,meta):
                # main scene/ sub scene
            if dataset.split('_')[0] == 'ShanghaiTech' or dataset == 'Avenue':
                main, sub ,frame = name.split('_')
                scene = main + '_' + sub
            else:
                scene ,frame = name.split('_')

            if cal_type=='sum':
                err = (1-k/10)*rec_err+(k/10)*pred_err
            elif cal_type=='max':
                err = max(rec_err,pred_err)

            if scene not in compute_dict:
                compute_dict[scene] = {}
            if int(frame) not in compute_dict[scene]:
                compute_dict[scene][int(frame)] = [err]
            else:
                compute_dict[scene][int(frame)].append(err)

        max_err_dict = {}
        all_label = []
        all_score = []
        all_nor_score = []

        for scene in compute_dict:
            max_err_dict[scene] = []
            frames = compute_dict[scene].keys()
            sorted_frames = list(sorted(frames))
            if dataset.split('_')[0] == 'ShanghaiTech':
                Label_Dir = ShanghaiTech_Lable_Dir + scene

            elif dataset == 'Corridor':

                Label_Dir = Corridor_Label_Dir + scene + '/' +scene

            label = np.load(Label_Dir+'.npy').tolist()
            num_frame = len(label)
            anchor = 0
            for i in range(num_frame):
                if i > sorted_frames[-1]:
                    max_err_dict[scene].append(0)
                elif int(sorted_frames[anchor]) == i:
                    max_rec = max(compute_dict[scene][sorted_frames[anchor]])
                    max_err_dict[scene].append(max_rec)
                    anchor += 1
                else:
                    max_err_dict[scene].append(0)

            ano_score = max_err_dict[scene]
            all_label.extend(label[duration:])
            all_score.extend(ano_score[duration:])
            all_nor_score.extend(normalize_anomaly_score(ano_score)[duration:])

    all_score = gaussian_filter1d(all_score, 20)
    all_nor_score = gaussian_filter1d(all_nor_score, 20)
    AUC = roc_auc_score(all_label, all_score)
    AUC_norm = roc_auc_score(all_label, all_nor_score)

    return AUC,AUC_norm