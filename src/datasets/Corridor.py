import os
import logging
import torch
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import glob

import sys 
sys.path.append("../")
from utils.load_save import load_json, write_json
from utils.normalize import *

POSE_META_FILE = 'pose_meta_{}_length{}_stride{}.json'
POSE_DATA_FILE = 'pose_data_{}_length{}_stride{}.json'

class Corridor(Dataset):
    def __init__(self, pose_dir, split='train', tracklet_len=8 , stride=2, pre_len=1,
        normalize_tracklet=True, normalize_score=True, normalize_pose=True,
        embed_dim=128, 
        mask_rec=True,fusion_type='div',motion_type='rayleigh',mask_pro=0.15):

        self.pose_dir = pose_dir
        self.split = split
        self.tracklet_len = tracklet_len
        self.stride = stride
        self.frame_width = 1920
        self.frame_height = 1080
        self.scale_factor = 100
        self.mask_rec = mask_rec
        self.fusion_type = fusion_type #'none' #fusion_type
        self.motion_type = motion_type
        self.mask_pro = mask_pro
        self.pre_len = pre_len

        self.joints_num = 25
        self.type_token, self.spatial_token, self.temporal_token = self._gen_fixed_token_seq()
        self.meta_path = pose_dir + '/' + POSE_META_FILE.format(self.split,str(self.tracklet_len),str(self.stride))
        self.tracklet_path = pose_dir + '/' + POSE_DATA_FILE.format(self.split,str(self.tracklet_len),str(self.stride))

        self.normalize_tracklet = True # normalize_boxes
        self.normalize_score = normalize_score
        self.normalize_pose = True # False # normalize_pose

        self._load_tracklets()

    def __len__(self):
        return len(self.meta_data)

    def _gen_fixed_token_seq(self):

        type_token = []
        spatial_token = []
        temporal_token = []
        single_type_tok = [0,0] + [1 for n in range(self.joints_num)]

        for i in range(self.tracklet_len):
            type_token.extend(single_type_tok)
            for j in range(self.joints_num):
                spatial_token.append(j)
                temporal_token.append(i)

        return torch.tensor(type_token), torch.tensor(spatial_token), torch.tensor(temporal_token)

    def _load_tracklets(self):

        if os.path.exists(self.tracklet_path) and os.path.exists(self.meta_path):
            print('Load {} Traclets from saved files, Traclet Length {}, Stride {}'.format(self.split, self.tracklet_len, self.stride))
            self.meta_data, self.tracklet_data = self._lazy_load_tracklets()
        else:
            print('Load {} Traclets from scratch, Traclet Length {}, Stride {}'.format(self.split, self.tracklet_len, self.stride))
            self.meta_data, self.tracklet_data = self._scratch_load_tracklets()

    def _lazy_load_tracklets(self):

        return load_json(self.meta_path), load_json(self.tracklet_path)

    def _scratch_load_tracklets(self):

        meta_data = []
        tracklet_data = []
        base_dir = self.pose_dir+'/'+self.split+'/'
        all_npy = glob.glob(os.path.join(base_dir, "*.npy"))
        logging.info('Processing raw traclets')
        filter_less_than = self.tracklet_len * self.stride

        for file in tqdm(all_npy):

            track = np.load(file)
            video_id, pid = file.split('/')[-1].split('_')
            if len(track)<filter_less_than:
                continue

            for i in range(len(track)-self.tracklet_len*self.stride):
                window = track[i : i+self.tracklet_len*self.stride : self.stride]
                start_frame = int(window[0][2])
                end_frame = int(window[-1][2])
                
                simple_pose = [ np.around(np.array(w[3:-2]),4).tolist() for w in window ]

                meta_data.append(video_id + '_' + str(end_frame))
                tracklet_data.append(simple_pose)

        print('Process Done. Sample amount: ', len(meta_data))
        write_json(meta_data,self.meta_path)
        print('Save meta data Done')
        write_json(tracklet_data,self.tracklet_path)
        print('Save data Done')

        return meta_data,tracklet_data

    # tracklet[8,17*3] (x,y,c)

    def _extract_boxes(self,tracklet,normalize=True):

        if normalize:
            box_xy_max = [[max(pose[:25])/self.frame_width,max(pose[25:50])/self.frame_height] for pose in tracklet]
            box_xy_min = [[min(pose[:25])/self.frame_width,min(pose[25:50])/self.frame_height] for pose in tracklet]
        else:
            box_xy_max = [[max(pose[:25]),max(pose[25:50])] for pose in tracklet]
            box_xy_min = [[min(pose[:25]),min(pose[25:50])] for pose in tracklet]

        return box_xy_max , box_xy_min

    def _extract_conf_score(self,tracklet,normalize=True):

        scores = []
        for pose in tracklet:
            pose_score = np.array(pose[50:75])
            if normalize:
                pose_score = normalize_score(pose_score)
            scores.append(pose_score.tolist())

        return scores

    def _extract_poses(self,tracklet,normalize=True):

        if isinstance(tracklet,list):
            tracklet = np.array(tracklet)

        x = tracklet[:, :25]
        y = tracklet[:, 25:50]

        valid = np.logical_or(x>1e-10, y>1e-10)
        no_valid = np.logical_and(x<1e-10, y<1e-10)
        valid_sum = valid.sum(axis=1)

        valid_sum[valid_sum < 1] = 1


        x_mean = (x.sum(axis=1)/valid_sum)[:, np.newaxis]
        y_mean = (y.sum(axis=1)/valid_sum)[:, np.newaxis]

        tmp = np.zeros_like(x)
        tmp[no_valid] = 1

        x += tmp*x_mean
        y += tmp*y_mean

        if normalize:
            x, y = normalize_pose(x,y)

        if isinstance(x,list):
            x, y = np.array(x), np.array(y)

        x[no_valid] = 0
        y[no_valid] = 0

        x = np.expand_dims(x,-1)
        y = np.expand_dims(y,-1)
        pose = np.concatenate((x,y),axis=-1).tolist()

        # (T,17,2)

        return pose

    def _extract_poses_boxes(self,tracklet,normalize_boxes=True,normalize_poses=False):

        if isinstance(tracklet,list):
            tracklet = np.array(tracklet)
        x = tracklet[:, :25]
        y = tracklet[:, 25:50]

        valid = np.logical_or(x>1e-10, y>1e-10)
        no_valid = np.logical_and(x<1e-10, y<1e-10)
        valid_sum = valid.sum(axis=1)

        valid_sum[valid_sum < 1] = 1


        x_mean = (x.sum(axis=1)/valid_sum)[:, np.newaxis]
        y_mean = (y.sum(axis=1)/valid_sum)[:, np.newaxis]

        tmp = np.zeros_like(x)
        tmp[no_valid] = 1

        x += tmp*x_mean
        y += tmp*y_mean

        if normalize_boxes:
            box_xy_max = [[x[i, :].max()/self.frame_width,y[i, :].max()/self.frame_height] for i in range(len(x))]
            box_xy_min = [[x[i, :].min()/self.frame_width,y[i, :].min()/self.frame_height] for i in range(len(y))]
        else:
            box_xy_max = [[x[i, :].max(),y[i, :].max()] for i in range(len(x))]
            box_xy_min = [[x[i, :].min(),y[i, :].min()] for i in range(len(y))]

        if normalize_poses:
            x, y = normalize_pose(x,y)
        else:
            x, y = center_pose(x, y)

        if isinstance(x,list):
            x, y = np.array(x), np.array(y)

        x[no_valid] = 0
        y[no_valid] = 0

        # w = np.abs(np.array(box_xy_max)[:,0] - np.array(box_xy_min)[:,0]).max()
        # h = np.abs(np.array(box_xy_max)[:,1] - np.array(box_xy_min)[:,1]).max()
        # x = x*w
        # y = y*h

        x = np.expand_dims(x,-1)
        y = np.expand_dims(y,-1)
        pose = np.concatenate((x,y),axis=-1).tolist()

        # (T,17,2)

        return pose, box_xy_max, box_xy_min

    def _inters_factor(self, v):

        if self.motion_type == 'gaussian':

            sigma = 0.18917838310469845
            mu = 0.09870275102403338
            factor = np.exp(-(np.linalg.norm(v-mu))**2/(2*(sigma**2)))

        if self.motion_type == 'rayleigh':
            
            sigma = 0.0202
            con = 0.0142
            factor = v * np.exp(-(v**2)/(2*(sigma**2))) / con
        
        if self.motion_type == 'uniform':

            factor = 0.5

        if self.motion_type == 'none':

            factor = 1

        return factor*0.7 + 0.3 # avoid zero

    def merge(self,pose,factor):

        if self.fusion_type == 'div':
            return pose / factor
        if self.fusion_type == 'add':
            return pose + factor
        if self.fusion_type == 'mul':
            return pose * factor

    def _gen_rec_mask(self,mask,prob=0.15):

        ref = torch.ones_like(torch.tensor(mask))
        masked_indices = torch.bernoulli(torch.full(ref.shape, prob)).bool()
        ref[masked_indices] = 0

        return ref.tolist()

    def _flat_input(self,poses, boxes_max, boxes_min, scores):

        assert len(poses) == len(boxes_max)
        assert len(boxes_max) == len(boxes_min)
        assert len(poses) == len(scores)

        pose_fusion = []

        weights = []
        inters = []
        poses_np = np.array(poses)
        boxes_max_np = np.array(boxes_max)
        boxes_min_np = np.array(boxes_min)

        for i in range(len(poses_np)-1):
            v = np.linalg.norm((boxes_max_np[i] + boxes_min_np[i])/2 - (boxes_max_np[(i+1)] + boxes_min_np[(i+1)])/2)
            v_norm = v/((boxes_max_np[i] - boxes_min_np[i] + 1e-6).mean())
            inters.append(self._inters_factor(v_norm))

        inters.append(inters[len(poses_np)-2])

        # inters = [max(inters)] * len(poses_np)
        
        pose_fusion = self.merge(poses_np[0],inters[0])[np.newaxis, :, :]
        weights.extend(scores[0])
        ## begin
        for i in range(len(poses)-1):
            pose_fusion = np.concatenate((pose_fusion, (self.merge(poses_np[i+1],inters[i+1]))[np.newaxis, :, :]), axis = 0)
            weights.extend(scores[i+1])

        return weights, pose_fusion.tolist()

    def __getitem__(self, idx):

        meta = self.meta_data[idx]
        tracklet = self.tracklet_data[idx]
        # boxes_max, boxes_min = self._extract_boxes(tracklet,self.normalize_tracklet)
        scores = self._extract_conf_score(tracklet,self.normalize_score)
        # poses = self._extract_poses(tracklet,self.normalize_pose)
        poses, boxes_max, boxes_min = self._extract_poses_boxes(tracklet,self.normalize_tracklet,self.normalize_pose)

        poses_gt = copy.deepcopy(poses)
        weights, pose_fusion = self._flat_input(poses, boxes_max, boxes_min,scores)
        pose_fusion = (torch.tensor(pose_fusion)*self.scale_factor).to(torch.int32)

        weights = torch.tensor(weights)
        poses_gt = (torch.tensor(poses_gt)*self.scale_factor).to(torch.int32)
        gt = poses_gt.reshape(-1,2)
        #poses_gt = torch.chunk(poses_gt,self.tracklet_len,0)
        weights = torch.chunk(weights,self.tracklet_len,0)

        weights = torch.cat([weights[i] for i in range(len(weights))],dim=0)
        #gt = torch.cat([poses_gt[i] for i in range(len(poses_gt))],dim=0)

        spatial_token = self.spatial_token.reshape(self.tracklet_len,-1)
        temporal_token = self.temporal_token.reshape(self.tracklet_len,-1)

        if self.mask_rec:
            mask = [1 for i in range((self.tracklet_len)*(self.joints_num))]
            mask = self._gen_rec_mask(mask)
            mask = torch.tensor(mask)
            #rint(mask)
            mask_ = torch.cat((mask.unsqueeze(0),mask.unsqueeze(0)),dim=0).permute(1,0)
            mask_index = mask_==0
            mask_index = mask_index.reshape(self.tracklet_len,self.joints_num,-1)
            pose_fusion[mask_index] = 0

        input_dict = {
            'meta': meta,
            'pose': pose_fusion,
            'gt': gt,
            'weigths': weights,
            'spatial_token':spatial_token,
            'temporal_token':temporal_token,
            'frame_width':self.frame_width,
            'frame_height':self.frame_height,
            'scale_factor': self.scale_factor,
            'joints_num':self.joints_num
            }

        return input_dict

if __name__ == '__main__':

    from dataset_path import *
    import cv2
    from torch.utils.data import DataLoader
    sys.path.append(".") 
    from utils.visualization import visualize_local_tracklets
    #from utils.metrics import recover_poses
    
    debug_Dataset = Corridor(pose_dir=Corridor_Pose_Dir,split='train',tracklet_len=8 , stride=2, pre_len=4)
    dataloader = DataLoader(debug_Dataset, batch_size=2, shuffle=True, num_workers=0)
    VIS = False

    for i, input_dict in enumerate(tqdm(dataloader)):

        print(input_dict['pose'].size())
        print(input_dict['weigths'].size())
        print(input_dict['gt'].size())
        print(input_dict['spatial_token'].size())
        print(input_dict['temporal_token'].size())
        print(input_dict['meta'])

        print("----------",i,"-------------")

        if i>10:
            break
