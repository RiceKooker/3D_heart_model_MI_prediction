"""
This script is to find the AUC.
The dataset used is for concatenated pointclouds.
Class balance is ensured.
"""
import argparse
import utils.utils1 as utils
import dataset.my_dataset as my_dataset
from matplotlib import pyplot as plt
from colorama import Fore, Back, Style
# from utils.dataset_utils import visualize_ds
from utils.utils2 import get_data_dir1 as get_data_dir
from utils.training_utils import find_ave_AUC_info, train_AUC, save_AUC_history, load_CV_idx_file
from utils.training_utils import cross_validation, cross_validation2


def load_dataset(args):
    # latent_ds = my_dataset.Dataset193(args)
    latent_ds = my_dataset.Dataset194(args)  # Overall dataset
    return latent_ds


if __name__ == "__main__":
    data_dir, dis_dir = get_data_dir('i')
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=int, default=625)
    parser.add_argument('--dpr', type=float, default=0.675)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--base_lr', type=float, default=1e-6)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--nfold', type=int, default=4)
    parser.add_argument('--reduce_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--val_prop', type=float, default=0.2)
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--disease_dir', type=str, default=dis_dir)
    args1 = parser.parse_args()

    # Cross validation
    ds = load_dataset(args1)
    total_CV_idx = load_CV_idx_file('DataDivision/Concatenated_incidence_idx.pkl')

    fold_history = cross_validation2(train_AUC, ds, total_CV_idx, args1)

    # fold_dict = np.load('AUC_history/fold_history_ROC_temp.npy', allow_pickle=True).item()

    fold_dict = fold_history
    Ave_TPR, Ave_FPR = find_ave_AUC_info(fold_dict)
    save_AUC_history(fold_history, args1.code)
    AUC = utils.find_AUC(Ave_FPR, Ave_TPR)

    print(Fore.MAGENTA + 'Dropout rate: {}'.format(args1.dpr) + Style.RESET_ALL)
    print(Fore.MAGENTA + 'AUC: {}'.format(AUC) + Style.RESET_ALL)
    f1 = utils.ROC_plot(Ave_FPR, Ave_TPR, 'FPR', 'TPR')
    plt.show()
