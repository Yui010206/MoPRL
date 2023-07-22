import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--seed',default=2021,type=int)
    parser.add_argument('--workers',default=2,type=int)
    parser.add_argument('--exp_name',default='debug',type=str)
    parser.add_argument('--inference',default=False,type=bool,help='turn on inference mode')
    parser.add_argument('--ano_score',default='max',type=str)
    parser.add_argument('--gpu',default=0,type=int)

    # show config
    parser.add_argument('--log_interval',default=100,type=int)
    parser.add_argument('--vis_interval',default=200,type=int)
    parser.add_argument('--eval_interval',default=2000,type=int)

    # test config
    parser.add_argument('--model_path',default='',type=str)

    # training config
    parser.add_argument('--batch_size',default=256,type=int,help='batch size')
    parser.add_argument('--lr_rate',default=5e-5,type=float)
    parser.add_argument('--epochs',default=50,type=int)
    parser.add_argument('--load_pretrain_model',default=False,type=bool)
    parser.add_argument('--iter_to_load',default=5000,type=int,help='load checkpoints')

    # dataset config
    parser.add_argument('--dataset',default='UCF_crime',type=str)
    parser.add_argument('--tracklet_len',default=8,type=int)
    parser.add_argument('--stride',default=1,type=int)
    parser.add_argument('--headless',default=False,type=bool)
    parser.add_argument('--mask_ratio',default=0.15,type=float)
    parser.add_argument('--motion_type',default='rayleigh',type=str) #rayleigh #gaussian #uniform #none
    parser.add_argument('--fusion_type',default='div',type=str) #div #add #mul
    parser.add_argument('--noise_factor',default=0,type=float)
    parser.add_argument('--pre_len',default=0,type=int)

    # model config
    parser.add_argument('--embed_dim',default=128,type=int)
    parser.add_argument('--spatial_depth',default=2,type=int)
    parser.add_argument('--temporal_depth',default=2,type=int)

    args = parser.parse_args()

    return args