import argparse


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='sam2', help='net type')
    parser.add_argument('-encoder', type=str, default='vit_b', help='encoder type')
    parser.add_argument('-exp_name', default='samba_train_test', type=str, help='experiment name')
    parser.add_argument('-vis', type=bool, default=False, help='Generate visualisation during validation')
    parser.add_argument('-train_vis', type=bool, default=False, help='Generate visualisation during training')
    parser.add_argument('-prompt', type=str, default='bbox', help='type of prompt, bbox or click') # 原始bbox
    parser.add_argument('-prompt_freq', type=int, default=2, help='frequency of giving prompt in 3D images')
    parser.add_argument('-pretrain', type=str, default=None, help='path of pretrain weights')
    parser.add_argument('-val_freq',type=int,default=5,help='interval between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu') # 1
    parser.add_argument('-image_size', type=int, default=1024, help='image_size')
    parser.add_argument('-out_size', type=int, default=1024, help='output_size')
    parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use')
    parser.add_argument('-dataset', default='vos' ,type=str,help='dataset name')  # btcv, vcod, vos
    parser.add_argument('-sam_ckpt', type=str, default=None , help='sam checkpoint address')
    parser.add_argument('-sam_config', type=str, default=None , help='sam checkpoint address')
    parser.add_argument('-video_length', type=int, default=3, help='sam checkpoint address') # moca是3,和视频加载匹配；图片是1
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-weights', type=str, default = 0, help='the weights file you want to test')
    parser.add_argument('-multimask_output', type=int, default=1 , help='the number of masks output for multi-class segmentation')
    parser.add_argument('-memory_bank_size', type=int, default=16, help='sam 2d memory bank size')
    parser.add_argument('-data_path', type=str, default='/home/user0/BRL/fabian/BRL/zhangxin/Datasets/Video_object_segmentation', help='The path of segmentation data')  # './dataset/btcv' , './dataset/vcod'
    parser.add_argument("--weight_decay", default=5e-4, type=float)


    # gmflow
    parser.add_argument("--attn_splits_list", default=[2], type=list)
    parser.add_argument("--corr_radius_list", default=[-1], type=list)
    parser.add_argument("--prop_radius_list", default=[-1], type=list)
    parser.add_argument("--pred_bidir_flow", default=True, type=bool)

    opt = parser.parse_args()

    return opt
