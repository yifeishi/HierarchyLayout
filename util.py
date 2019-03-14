import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='grass_pytorch')
    parser.add_argument('--obj_code_size', type=int, default=256+8)
    parser.add_argument('--box_code_size', type=int, default=8)
    parser.add_argument('--box_size', type=int, default=8)
    parser.add_argument('--feature_size', type=int, default=1000)
    parser.add_argument('--pc_feature_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=1000)
    parser.add_argument('--layout_input_size', type=int, default=8)
    parser.add_argument('--layout_code_size', type=int, default=7)
    parser.add_argument('--layout_feature_size', type=int, default=100)
    parser.add_argument('--symmetry_size', type=int, default=8)
    parser.add_argument('--max_box_num', type=int, default=50)
    parser.add_argument('--max_sym_num', type=int, default=10)

    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--show_log_every', type=int, default=1)
    parser.add_argument('--save_log', action='store_true', default=False)
    parser.add_argument('--save_log_every', type=int, default=20)
    parser.add_argument('--save_snapshot', action='store_true', default=True)
    parser.add_argument('--save_snapshot_every', type=int, default=30)
    parser.add_argument('--no_plot', action='store_true', default=True)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--lr_decay_by', type=float, default=1)
    parser.add_argument('--lr_decay_every', type=float, default=20)
    parser.add_argument('--validate', action='store_true', default=True)
    parser.add_argument('--validate_every', type=float, default=20)

    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--g_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='/home/net663/Downloads/yifeis/S3DIS/region_feature')
    parser.add_argument('--save_path', type=str, default='models')
    parser.add_argument('--nick_name', type=str, default='')
    parser.add_argument('--resume_snapshot', type=str, default='')
    parser.add_argument('--pretrained_model', type=str, default='')

    parser.add_argument('--room_type', type=str, default='Area_1')
    parser.add_argument('--ap_category', type=int, default=6)
    parser.add_argument('--IOU', type=float, default=0.5)

    args = parser.parse_args()
    return args
