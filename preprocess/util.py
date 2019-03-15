import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='preprocess')
    parser.add_argument('--g_path', type=str, default='../data')
    parser.add_argument('--data_path', type=str, default='../processed_data')
    parser.add_argument('--pretrained_model', type=str, default='pretrained_model/model')
    args = parser.parse_args()
    return args
