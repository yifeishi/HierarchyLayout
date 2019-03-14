import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='pointcnn')
    parser.add_argument('--g_path', type=str, default='')
    args = parser.parse_args()
    return args
