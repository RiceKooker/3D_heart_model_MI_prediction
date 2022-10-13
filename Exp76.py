"""
Variation of Exp62.py
It tests all 9 scenarios for prevalence.
"""
import argparse
import utils.utils1 as utils
import dataset.my_dataset as my_dataset
from matplotlib import pyplot as plt
from colorama import Fore, Back, Style
# from utils.dataset_utils import visualize_ds
from utils.training_utils import save_fold_history, train, calculate_fold_ave, load_CV_idx_file, cross_validation2


def load_dataset(args):
    latent_ds = my_dataset.Dataset1941(args)  # Use 194 if concatenated
    # latent_ds = my_dataset.Dataset194(args)
    return latent_ds


if __name__ == "__main__":
    data_dir, dis_dir = utils.get_both_dir('p', 'ES', 'lv')
    parser = argparse.ArgumentParser()
    parser.add_argument('--code', type=int, default=7681)
    parser.add_argument('--dpr', type=float, default=0.7)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--base_lr', type=float, default=1e-6)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--nfold', type=int, default=4)  # This cannot be changed since a pre-defined CV index file
    # is used
    parser.add_argument('--reduce_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--val_prop', type=float, default=0.2)
    parser.add_argument('--fold_count', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default=data_dir)
    parser.add_argument('--disease_dir', type=str, default=dis_dir)
    parser.add_argument('--stage_spec', type=str, default='Prevalence')
    parser.add_argument('--phase_spec', type=str, default='ES')
    parser.add_argument('--ventricle', type=str, default='Left')

    args1 = parser.parse_args()

    # Load the dataset and load the pre-determined data division configuration file.
    ds = load_dataset(args1)
    # visualize_ds(ds)
    total_CV_idx = load_CV_idx_file('DataDivision/Concatenated_prevalence_idx.pkl')

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
