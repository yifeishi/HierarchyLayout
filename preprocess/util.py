import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='preprocess')
    parser.add_argument('--g_path', type=str, default='/home/net663/Downloads/yifeis/S3DIS/data_release')
    parser.add_argument('--pickle_path', type=str, default='/home/net663/Downloads/yifeis/S3DIS/data_release')
    args = parser.parse_args()
    return args
