import os
import logging
from sklearn.utils import shuffle
import torch
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import random

import sys 
sys.path.append("../../src")
from utils.load_save import load_json, write_json
from utils.normalize import normalize_score,normalize_pose

POSE_META_FILE = 'processed/pose_meta_{}_length{}_stride{}.json'
POSE_DATA_FILE = 'processed/pose_data_{}_length{}_stride{}.json'

class UCF_crime(Dataset):
    def __init__(self, pose_dir, split='train', tracklet_len=8, stride=1, pre_len=0, head_less=False,
        normalize_tracklet=True, normalize_score=True,
        normalize_pose=True, embed_dim=128, 
        mask_rec=True,fusion_type='div',motion_type='rayleigh', mask_pro=0.15):

        self.pose_dir = pose_dir
        self.split = split
        self.head_less = head_less
        self.tracklet_len = tracklet_len
        self.stride = stride
        ## TO DO
        self.frame_width = 320
        self.frame_height = 240
        ## TO DO END
        self.scale_factor = 100
        self.mask_rec = mask_rec
        self.fusion_type = fusion_type
        self.motion_type = motion_type
        self.mask_pro = mask_pro

        if self.head_less:
            self.joints_num =14
        else:
            self.joints_num =17

        self.pre_len = pre_len

        self.type_token, self.spatial_token, self.temporal_token = self._gen_fixed_token_seq()
        self.meta_path = os.path.join(pose_dir, POSE_META_FILE.format(self.split,str(self.tracklet_len),str(self.stride)))
        self.tracklet_path = os.path.join(pose_dir, POSE_DATA_FILE.format(self.split,str(self.tracklet_len),str(self.stride)))

        self.propossed_path = os.path.dirname(self.meta_path)
        os.makedirs(self.propossed_path, exist_ok=True)

        self.normalize_tracklet = normalize_tracklet
        self.normalize_score = normalize_score
        self.normalize_pose = normalize_pose

        self._load_tracklets()
        print('dataset length: {}'.format(self.__len__()))

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

    ## TO DO

    def _scratch_load_tracklets(self):

        meta_data = []
        tracklet_data = []

        split_info = load_json(os.path.join(self.pose_dir, 'train_test_split.json'))
        name_list = split_info[self.split]

        base_dir = os.path.join(self.pose_dir, 'ucf_samples')
        logging.info('Processing raw traclets')
        filter_less_than = self.tracklet_len * self.stride

        for name in tqdm(name_list):
            filepath = os.path.join(base_dir, name[:-4], 'alphapose-results.json')
            if not os.path.exists(filepath):
                continue
            origin_tracks = load_json(filepath)
            person_tracks_frame = {}
            person_tracks_pose = {}
            person_tracks_frame_exist = {}
            person_frame_for_search = {}
            for detected in origin_tracks:
                if detected['idx'] not in person_tracks_frame.keys():
                    person_frame_for_search[detected['idx']] = []
                    person_tracks_frame[detected['idx']] = [None]*1000000
                    person_tracks_pose[detected['idx']] = [None]*1000000
                    person_tracks_frame_exist[detected['idx']] = np.zeros(1000000, dtype=bool)
                else:
                    person_frame_for_search[detected['idx']].append(int(detected['image_id'][:-4]))
                    person_tracks_frame[detected['idx']][int(detected['image_id'][:-4])] = detected['image_id'][:-4].rjust(4, '0')
                    person_tracks_pose[detected['idx']][int(detected['image_id'][:-4])] = detected['keypoints']
                    person_tracks_frame_exist[detected['idx']][int(detected['image_id'][:-4])] = True

            # person_num = len(person_tracks.keys())
            for p in person_frame_for_search.keys():
                frame_num = len(person_frame_for_search[p])
                if frame_num < filter_less_than:
                    continue

                if self.split == 'train':
                    if frame_num < filter_less_than*2:
                        continue
                
                ### version1
                # for i in range(frame_num-self.tracklet_len*self.stride):
                #     simple_pose = person_tracks_pose[p][i : i+self.tracklet_len*self.stride : self.stride]
                #     meta_data.append(name+'_'+person_tracks_frame[p][i+(self.tracklet_len-1)*self.stride])
                #     tracklet_data.append(simple_pose)

                ### version2
                for j in range(frame_num):
                    i = int(person_frame_for_search[p][j])
                    if np.all(person_tracks_frame_exist[p][i : i+self.tracklet_len*self.stride : self.stride]):
                        simple_pose = person_tracks_pose[p][i : i+self.tracklet_len*self.stride : self.stride]
                        meta_data.append(name+'_'+person_tracks_frame[p][i+(self.tracklet_len-1)*self.stride])
                        tracklet_data.append(simple_pose)

        print('Process Done. Sample amount: ', len(meta_data))
        write_json(meta_data,self.meta_path)
        print('Save meta data Done')
        write_json(tracklet_data,self.tracklet_path)
        print('Save data Done')

        return meta_data,tracklet_data
    
    ## TO DO END

    # tracklet[8,17*3] (x,y,c)

    def _extract_boxes(self,tracklet,normalize=True):

        if normalize:
            box_xy_max = [[max(pose[::3])/self.frame_width,max(pose[1::3])/self.frame_height] for pose in tracklet]
            box_xy_min = [[min(pose[::3])/self.frame_width,min(pose[1::3])/self.frame_height] for pose in tracklet]
        else:
            box_xy_max = [[max(pose[::3]),max(pose[1::3])] for pose in tracklet]
            box_xy_min = [[min(pose[::3]),min(pose[1::3])] for pose in tracklet]

        return box_xy_max , box_xy_min

    def _extract_conf_score(self,tracklet,normalize=True):

        scores = []
        for pose in tracklet:
            pose_score = np.array(pose[2::3])
            if normalize:
                pose_score = normalize_score(pose_score)
            scores.append(pose_score.tolist())

        return scores

    def _extract_poses(self,tracklet,normalize=True):

        if isinstance(tracklet,list):
            tracklet = np.array(tracklet)
        x = tracklet[:, ::3]
        y = tracklet[:, 1::3]

        if normalize:
            x, y = normalize_pose(x,y)

        if isinstance(x,list):
            x, y = np.array(x), np.array(y)

        x = np.expand_dims(x,-1)
        y = np.expand_dims(y,-1)
        pose = np.concatenate((x,y),axis=-1).tolist()

        # (T,17,2)

        return pose
    
    ## TO DO

    def _inters_factor(self, v):

        if self.motion_type == 'gaussian':

            # sigma = 0.18917838310469845
            # mu = 0.09870275102403338
            # factor = np.exp(-(np.linalg.norm(v-mu))**2/(2*(sigma**2)))
            pass

        if self.motion_type == 'rayleigh':

            if self.stride == 1:
    
                sigma = 0.008
                con = 0.0048

            if self.stride == 2:

                sigma = 0.009
                con = 0.0055

            factor = v * np.exp(-(v**2)/(2*(sigma**2))) / con
        
        if self.motion_type == 'uniform':

            factor = 0.5

        if self.motion_type == 'none':

            factor = 1

        if self.motion_type == 'random':

            factor = 1 + 0.5*np.random.rand()

        return factor*0.7 + 0.3
    
    ## TO DO END

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
            v_norm = v/((boxes_max_np[i] - boxes_min_np[i]).mean())
            inters.append(self._inters_factor(v_norm))

        inters.append(inters[len(poses_np)-2])
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
        boxes_max, boxes_min = self._extract_boxes(tracklet,self.normalize_tracklet)
        scores = self._extract_conf_score(tracklet,self.normalize_score)
        poses = self._extract_poses(tracklet,self.normalize_pose)

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

        if self.mask_rec and self.split=='train':
            mask = [1 for i in range((self.tracklet_len)*(self.joints_num))]
            mask = self._gen_rec_mask(mask,self.mask_pro)
            mask = torch.tensor(mask)
            #rint(mask)
            mask_ = torch.cat((mask.unsqueeze(0),mask.unsqueeze(0)),dim=0).permute(1,0)
            mask_index = mask_==0
            mask_index = mask_index.reshape(self.tracklet_len,self.joints_num,-1)
            pose_fusion[mask_index] = 0

        if self.pre_len>0 :
            mask = torch.tensor([1 for i in range((self.tracklet_len-self.pre_len)*(self.joints_num))] + [0 for i in range(self.joints_num*self.pre_len)])
            mask_ = torch.cat((mask.unsqueeze(0),mask.unsqueeze(0)),dim=0).permute(1,0)
            mask_ = mask_.reshape(self.tracklet_len, self.joints_num, 2)
            mask_index = mask_==0
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
    # import cv2
    from torch.utils.data import DataLoader
    sys.path.append(".") 
    from utils.visualization import visualize_local_tracklets
    #from utils.metrics import recover_poses
    
    debug_Dataset = UCF_crime(pose_dir=UCF_crime_Dir,split='test',tracklet_len=8 , stride=2, head_less=False,pre_len=4)
    dataloader = DataLoader(debug_Dataset, batch_size=2, shuffle=True, num_workers=0)
    VIS = False

    for i, input_dict in enumerate(tqdm(dataloader)):

        # print(input_dict['MPP_GT'].size())
        # print(input_dict['MPR_GT'].size())
        #print(input_dict['pose'])
        print(input_dict['spatial_token'].size())
        print(input_dict['temporal_token'].size())
        print(input_dict['meta'])
        #recovered_poses = recover_poses(input_dict['MPP_GT'],input_dict['MTP_GT'],'ShanghaiTech')
        #print('recovered_poses',recovered_poses.shape)

        print("----------",i,"-------------")

        if i>10:
            break


