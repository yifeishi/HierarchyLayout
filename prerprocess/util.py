import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='preprocess')
    parser.add_argument('--g_path', type=str, default='')
    parser.add_argument('--g_gt_path', type=str, default='')
    parser.add_argument('--pickle_path', type=str, default='')
    parser.add_argument('--scene_list_path', type=str, default='')
    args = parser.parse_args()
    return args
