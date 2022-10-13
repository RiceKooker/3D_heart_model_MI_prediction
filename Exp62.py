"""
This script uses the additional data Marcel provided. Dataset has been separated based on incidence and prevalence.
Gender is not used as a conditional input.
It uses concatenated data of ES and ED point clouds.
It ensures class balance in the overall dataset and all data division.
It uses all the data available and duplicates class 1 data to ensure class balance in the training set.
"""
import argparse
import utils.utils1 as utils
import dataset.my_dataset as my_dataset
from matplotlib import pyplot as plt
from colorama import Fore, Back, Style
# from utils.dataset_utils import visualize_ds
from utils.utils2 import get_data_dir1 as get_data_dir
from utils.training_utils import save_fold_history, train, calculate_fold_ave, load_CV_idx_file, cross_validation2


def load_dataset(args):
    latent_ds = my_dataset.Dataset194(args)  # Overall dataset
    return latent_ds


if __name__ == "__main__":
    data_dir, dis_dir = get_data_dir('i')
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=int, default=73181)
    parser.add_argument('--dpr', type=float, default=0.725)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--base_lr', type=float, default=1e-6)
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--nfold', type=int, default=4)  # This cannot be changed since a pre-defined CV index file
    # is used
    parser.add_argument('--reduce_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--val_prop', type=float, default=0.2)
    parser.add_argument('--fold_count', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--disease_dir', type=str, default=dis_dir)
    parser.add_argument('--stage_spec', type=str, default='Incidence')
    parser.add_argument('--phase_spec', type=str, default='Concatenated')
    parser.add_argument('--ventricle', type=str, default='Full')

    args1 = parser.parse_args()

    # Load the dataset and load the pre-determined data division configuration file.
    ds = load_dataset(args1)
    # visualize_ds(ds)
    total_CV_idx = load_CV_idx_file('DataDivision/Concatenated_incidence_idx.pkl')

    # Cross validation
    fold_history = cross_validation2(train, ds, total_CV_idx, args1)

    # Save the source data
    save_fold_history(fold_history, args1.code)

    # Evaluate the average performance
    fold_dict = fold_history
    Ave_history, test_acc = calculate_fold_ave(fold_dict)
    print(Fore.MAGENTA + 'Dropout rate: {}'.format(args1.dpr) + Style.RESET_ALL)
    print(Fore.MAGENTA + 'Final test accuracy: {}'.format(test_acc) + Style.RESET_ALL)
    ave_performance = utils.print_final_performance(fold_dict)

    # Plot the average performance
    f1, f2 = utils.loss_plot(Ave_history)
    plt.show()

