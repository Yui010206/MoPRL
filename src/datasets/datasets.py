from .dataset_path import *
import os

def get_training_set(opt):
    assert opt.dataset in ['ShanghaiTech_AlphaPose', 'Corridor', 'UCF_crime']

    if opt.dataset == 'ShanghaiTech_AlphaPose':
    
        from .ShanghaiTech_AlphaPose import ShanghaiTech_AlphaPose

        train_Dataset = ShanghaiTech_AlphaPose(pose_dir=ShanghaiTech_AlphaPose_Dir, split='train', mask_pro=opt.mask_ratio,
            tracklet_len=opt.tracklet_len ,stride=opt.stride,head_less=opt.headless,pre_len=opt.pre_len,embed_dim=opt.embed_dim, fusion_type=opt.fusion_type, motion_type=opt.motion_type, noise_factor = opt.noise_factor)

    elif opt.dataset == 'Corridor':
        from .Corridor import Corridor

        train_Dataset = Corridor(pose_dir=Corridor_Pose_Dir, split='train', mask_pro=opt.mask_ratio,
            tracklet_len=opt.tracklet_len ,stride=opt.stride,pre_len=opt.pre_len,embed_dim=opt.embed_dim, fusion_type=opt.fusion_type, motion_type=opt.motion_type)

    elif opt.dataset == 'UCF_crime':
        
        from .UCF_crime import UCF_crime

        train_Dataset = UCF_crime(pose_dir=UCF_crime_Dir, split='train', mask_pro=opt.mask_ratio,
            tracklet_len=opt.tracklet_len ,stride=opt.stride,head_less=opt.headless,pre_len=opt.pre_len,embed_dim=opt.embed_dim, fusion_type=opt.fusion_type, motion_type=opt.motion_type)


    return train_Dataset


def get_test_set(opt):
    assert opt.dataset in ['ShanghaiTech_AlphaPose', 'Corridor', 'UCF_crime']

    if opt.dataset == 'ShanghaiTech_AlphaPose':

        from .ShanghaiTech_AlphaPose import ShanghaiTech_AlphaPose

        test_Dataset = ShanghaiTech_AlphaPose(pose_dir=ShanghaiTech_AlphaPose_Dir, split='test', 
            tracklet_len=opt.tracklet_len ,stride=opt.stride,head_less=opt.headless,pre_len=opt.pre_len,embed_dim=opt.embed_dim, fusion_type=opt.fusion_type, motion_type=opt.motion_type, noise_factor = opt.noise_factor)

    elif opt.dataset == 'Corridor':
        from .Corridor import Corridor

        test_Dataset = Corridor(pose_dir=Corridor_Pose_Dir, split='test', 
            tracklet_len=opt.tracklet_len ,stride=opt.stride,pre_len=opt.pre_len,embed_dim=opt.embed_dim, fusion_type=opt.fusion_type, motion_type=opt.motion_type)
    
    elif opt.dataset == 'UCF_crime':
        
        from .UCF_crime import UCF_crime

        test_Dataset = UCF_crime(pose_dir=UCF_crime_Dir, split='test', 
            tracklet_len=opt.tracklet_len ,stride=opt.stride,head_less=opt.headless,pre_len=opt.pre_len,embed_dim=opt.embed_dim, fusion_type=opt.fusion_type, motion_type=opt.motion_type)

    else:
        raise ValueError ("Dataset Name Invalid!")


    return test_Dataset
    