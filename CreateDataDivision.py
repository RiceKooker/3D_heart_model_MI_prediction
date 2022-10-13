"""
This script generates the data division indices for different test scenarios.
"""

import os
import argparse
import dataset.my_dataset as my_dataset
from colorama import Fore, Back, Style
from utils.utils2 import get_data_dir1 as get_data_dir
from utils.utils2 import k_fold_idx
from utils.utils1 import get_both_dir
import pickle


if __name__ == "__main__":
    # data_dir, dis_dir = get_data_dir('p')
    data_dir, dis_dir = get_both_dir('i', 'ES', 'lv')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpr', type=float, default=0.675)
    parser.add_argument('--nfold', type=int, default=4)
    parser.add_argument('--reduce_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--val_prop', type=float, default=0.2)
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--disease_dir', type=str, default=dis_dir)
    args1 = parser.parse_args()

    nfold = args1.nfold
    ds = my_dataset.Dataset193(args1)
    total_CV_idx = k_fold_idx(ds, nfold, args1.val_prop)

    file_name = 'Concatenated_prevalence_idx.pkl'
    file_name = 'DataDivision/' + file_name
    if os.path.exists(file_name):
        print(Fore.RED + 'File already exist! New idx file generation failed!' + Style.RESET_ALL)
    else:
        with open(file_name, "wb") as fp:
            pickle.dump(total_CV_idx, fp)