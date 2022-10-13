"""
Exp21.np
"""
import argparse
import utils.utils1 as utils
import dataset.my_dataset as my_dataset
from matplotlib import pyplot as plt
from colorama import Fore, Back, Style
# from utils.dataset_utils import visualize_ds
from utils.utils2 import get_data_dir1 as get_data_dir
from utils.training_utils import save_fold_history, train, calculate_fold_ave, load_CV_idx_file, cross_validation


def load_dataset(args):
    latent_ds = my_dataset.Dataset1901(args)  # Overall dataset
    return latent_ds


if __name__ == "__main__":
    data_dir, dis_dir = get_data_dir('p')
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=int, default=619)
    parser.add_argument('--dpr', type=float, default=0.675)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--base_lr', type=float, default=1e-6)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--nfold', type=int, default=4)  # This cannot be changed since a pre-defined CV index file
    # is used
    parser.add_argument('--reduce_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--val_prop', type=float, default=0.2)
    parser.add_argument('--fold_count', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--disease_dir', type=str, default=dis_dir)
    args1 = parser.parse_args()

    ds = load_dataset(args1)
    # visualize_ds(ds)
    total_CV_idx = load_CV_idx_file('DataDivision/Concatenated_prevalence_idx.pkl')

    fold_history = cross_validation(train, ds, total_CV_idx, args1)
    save_fold_history(fold_history, args1.code)

    fold_dict = fold_history
    Ave_history, test_acc = calculate_fold_ave(fold_dict)
    print(Fore.MAGENTA + 'Dropout rate: {}'.format(args1.dpr) + Style.RESET_ALL)
    print(Fore.MAGENTA + 'Final test accuracy: {}'.format(test_acc) + Style.RESET_ALL)

    f1, f2 = utils.loss_plot(Ave_history)
    plt.show()
