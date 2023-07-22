import os, time, sys, cv2
import torch
import random

import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW,get_linear_schedule_with_warmup

from utils.logger import get_logger
from utils.load_save import save_parameters,write_json
from utils.losses import OKS_Loss
from utils.visualization import visualize_local_tracklets
from utils.metrics import recover_poses,L1_err,L2_err,compute_auc
from opts import parse_opts
from datasets.datasets import get_training_set, get_test_set
from models.moprl import MoPRL
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Train_Eval_Inference(object):

    def __init__(self, opt):

        self.opt = opt
        self.dataset_name = opt.dataset
        self.exp_name = opt.exp_name

        self.workspace = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../exps/')
        self.jobname = opt.dataset
        self.exp_dir = os.path.join(self.workspace, self.exp_name)
        self.model_save_dir = os.path.join(self.exp_dir, 'models')
        self.vis_sample_dir = os.path.join(self.exp_dir, 'vis_samples')
        self.test_result_dir = os.path.join(self.exp_dir, 'result')

        self.train_tasks = 'rec'
        self.test_tasks = 'rec'
        self.scale_factor = 100

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        if not os.path.exists(self.vis_sample_dir):
            os.makedirs(self.vis_sample_dir)
        if not os.path.exists(self.test_result_dir):
            os.makedirs(self.test_result_dir)

        # whether to start training from an existing snapshot
        self.load_pretrain_model = opt.load_pretrain_model
        if self.load_pretrain_model:
            self.iter_to_load = opt.iter_to_load

        save_parameters(self.exp_dir,opt)

        train_Dataset = get_training_set(opt)
        self.train_loader = DataLoader(train_Dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                                      pin_memory=True, drop_last=True)
        test_Dataset = get_test_set(opt)
        self.test_loader = DataLoader(test_Dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers,
                                     pin_memory=True, drop_last=False)

        self.logger = get_logger(self.exp_dir + '/log.txt')

        self.oks_loss = OKS_Loss()

        if self.opt.dataset == 'ShanghaiTech_AlphaPose' or self.opt.dataset == 'UCF_crime':
            self.num_joints = 17
        else:
            self.num_joints = 25

    def train_batch(self,model,optimizer,epoch,iteration,scheduler=None):

        for input_dict in iter(self.train_loader):

            pose = input_dict['pose'].float().cuda()
            weigths = input_dict['weigths'].float().cuda()
            gt = input_dict['gt'].float().cuda()
            spatial_token = input_dict['spatial_token'].long().cuda()
            temporal_token = input_dict['temporal_token'].long().cuda()

            model.zero_grad()
            rec_pose = model(pose,spatial_token,temporal_token)

            loss = self.oks_loss(rec_pose,gt,weigths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            iteration += 1

            if iteration % self.opt.log_interval == 0:

                self.logger.info("iter {} (epoch {}), loss = {:.6f}".format(iteration, epoch, loss.item()))

            if iteration % self.opt.vis_interval == 0:

                pred_pose = rec_pose[0].cpu().detach()
                gt_pose = gt[0].cpu().detach()
                pred_local_img = visualize_local_tracklets(pred_pose, self.num_joints)
                gt_local_img = visualize_local_tracklets(gt_pose, self.num_joints)
                local_imgs = np.vstack([gt_local_img,pred_local_img])
                cv2.imwrite(self.vis_sample_dir+'/{}_normal.jpg'.format(str(iteration)),local_imgs)

            if iteration % self.opt.eval_interval == 0 or iteration == 1:

                self.logger.info('Start evaluation!')
                model.eval()
                l1, l2, all_meta, vis_pose, vis_meta, vis_gt = self.eval_batch(model)
                write_json(l1,self.test_result_dir+'/iteration{}_L1.json'.format(str(iteration)))
                write_json(l2,self.test_result_dir+'/iteration{}_L2.json'.format(str(iteration)))
                write_json(all_meta,self.test_result_dir+'/iteration{}_meta.json'.format(str(iteration)))

                model.train()
                torch.save(model.state_dict(), self.model_save_dir+'/{:06d}_model.pth.tar'.format(iteration))  

        return iteration

    def eval_batch(self,model):
        # Set to evaluation mode (randomly sample z from the whole distribution)
        all_err_l1 = []
        all_err_l2 = []
        all_meta = []
        vis_pose = []
        vis_meta = []
        vis_gt = []

        with torch.no_grad():
            for i,input_dict in enumerate(tqdm(self.test_loader)):
                    #input = input_dict['input_sequence'].float().cuda()
                weigths = input_dict['weigths'].float().cuda()
                pose = input_dict['pose'].float().cuda()
                gt = input_dict['gt'].float()
                spatial_token = input_dict['spatial_token'].long().cuda()
                temporal_token = input_dict['temporal_token'].long().cuda()
                meta = input_dict['meta']
                output = model(pose,spatial_token,temporal_token)
                err_l1 = L1_err(output.cpu(),gt)
                err_l2 = L2_err(output.cpu(),gt)

                all_err_l1.extend(err_l1)
                all_err_l2.extend(err_l2)
                all_meta.extend(meta)

            L1_auc, L1_norm_auc = compute_auc(all_err_l1,all_err_l1,all_meta,0, 0,self.dataset_name)
            L2_auc, L2_norm_auc = compute_auc(all_err_l2,all_err_l2,all_meta,0, 0,self.dataset_name)
            self.logger.info('Best AUC under L1 Err: {}'.format(str(round(L1_auc,4)*100)))
            self.logger.info('Best AUC under L2 Err: {}'.format(str(round(L2_auc,4)*100)))
            self.logger.info('Best AUC under Norm L1 Err: {}'.format(str(round(L1_norm_auc,4)*100)))
            self.logger.info('Best AUC under Norm L2 Err: {}'.format(str(round(L2_norm_auc,4)*100)))

        vis_pose_np = np.array(vis_pose)
        vis_gt_np = np.array(vis_gt)

        return all_err_l1,all_err_l2,all_meta, vis_pose_np, vis_meta, vis_gt_np


    def inference(self):
        
        model = MoPRL(tracklet_len=self.opt.tracklet_len,headless=self.opt.headless,pre_len=opt.pre_len,embed_dim=self.opt.embed_dim,
            spatial_depth=self.opt.spatial_depth, temporal_depth=self.opt.temporal_depth, num_joints=self.num_joints).cuda()
        model.load_state_dict(torch.load(self.opt.model_path))

        # ToDo
        self.logger.info('Start evaluation!')
        model.eval()

        all_err_l1 = []
        all_err_l2 = []
        all_err_score = []
        all_meta = []
        out_pose = []

        with torch.no_grad():
            for i,input_dict in enumerate(tqdm(self.test_loader)):

                weigths = input_dict['weigths'].float().cuda()
                pose = input_dict['pose'].float().cuda()
                gt = input_dict['gt'].float()
                spatial_token = input_dict['spatial_token'].long().cuda()
                temporal_token = input_dict['temporal_token'].long().cuda()
                meta = input_dict['meta']
                output = model(pose,spatial_token,temporal_token)

                err_l1 = L1_err(output.cpu(),gt)
                err_l2 = L2_err(output.cpu(),gt)
                all_err_l1.extend(err_l1)
                all_err_l2.extend(err_l2)
                all_meta.extend(meta)
                out_pose.extend(output.cpu().numpy())

            L1_auc, L1_norm_auc = compute_auc(all_err_l1,all_err_l1,all_meta,0, 0,self.dataset_name)
            L2_auc, L2_norm_auc = compute_auc(all_err_l2,all_err_l2,all_meta,0, 0,self.dataset_name)
            self.logger.info('Best AUC under L1 Err: {}'.format(str(round(L1_auc,4)*100)))
            self.logger.info('Best AUC under L2 Err: {}'.format(str(round(L2_auc,4)*100)))
            self.logger.info('Best AUC under Norm L1 Err: {}'.format(str(round(L1_norm_auc,4)*100)))
            self.logger.info('Best AUC under Norm L2 Err: {}'.format(str(round(L2_norm_auc,4)*100)))


    def train_eval(self):

        gpu_ids = range(torch.cuda.device_count())
        self.logger.info('Number of GPUs in use {}'.format(gpu_ids))
        
        
        model = MoPRL(tracklet_len=self.opt.tracklet_len,headless=self.opt.headless,pre_len=opt.pre_len,embed_dim=self.opt.embed_dim,
            spatial_depth=self.opt.spatial_depth, temporal_depth=self.opt.temporal_depth, num_joints=self.num_joints).cuda()

        total_steps = len(self.train_loader)*self.opt.epochs
        optimizer = AdamW(model.parameters(), lr=opt.lr_rate, eps = 1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 1000, num_training_steps = total_steps)

        self.logger.info(self.jobname)

        iteration = 0
        if self.load_pretrain_model:
            model_name = self.model_save_dir + '/{:06d}_model.pth.tar'.format(self.iter_to_load)
            self.logger.info("loading model from {}".format(model_name))
            state_dict = torch.load(model_name)
            model.load_state_dict(state_dict['model'])
            optimizer.load_state_dict(state_dict['optimizer'])
            iteration = self.iter_to_load + 1

        tmp = sum(p.numel() for p in model.parameters())

        self.logger.info('model paras sum: {}'.format(tmp))

        self.logger.info('Start Training!')

        for epoch in range(self.opt.epochs):

            model.train()
            iteration = self.train_batch(model,optimizer,epoch,iteration,scheduler=scheduler)
            
        self.logger.info('End Training!')

if __name__ == '__main__':

    opt = parse_opts()
    print (opt)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

    torch.manual_seed(opt.seed) 
    torch.cuda.manual_seed(opt.seed) 
    torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(opt.seed)

    pipeline = Train_Eval_Inference(opt)

    if opt.inference:
        pipeline.inference()
    else:
        pipeline.train_eval()