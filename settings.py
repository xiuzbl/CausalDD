import os
import argparse
import torch

def myparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file",type=str)
    parser.add_argument("--add_mlm_pretrain", type=bool)
    parser.add_argument("--add_lm_entire_response_generation", type=bool)
    args = parser.parse_args()
    return args
