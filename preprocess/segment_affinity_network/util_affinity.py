import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='plygen')
    parser.add_argument('--seg_json_name', type=str, default='region0.0.010000.segs.json')
    parser.add_argument('--ply_name', type=str, default='region0.ply')
    parser.add_argument('--g_path', type=str, default='/home/net663/Downloads/yifeis/Matterport/region_segmentations/1LXtFkjw3qL/region_segmentations/region0')
    parser.add_argument('--g_box_path', type=str, default='/home/net663/Downloads/yifeis/Matterport/obbs/1LXtFkjw3qL/house_features/region_0')
    parser.add_argument('--out_path', type=str, default='./')
    
    args = parser.parse_args()
    return args
