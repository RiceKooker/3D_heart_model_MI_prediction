"""
This script is to tune the hyperparameters in the incident, concatenated, biventricular test.
"""
import argparse
import utils.utils1 as utils
import dataset.my_dataset as my_dataset
import utils.training_utils as utils_train
import pandas as pd
from matplotlib import pyplot as plt
import torch
import random
import numpy as np


def load_dataset(args, is_Conc=False):
    if is_Conc:
        latent_ds = my_dataset.Dataset194(args)
    else:
        latent_ds = my_dataset.Dataset1941(args)
    return latent_ds


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pd.set_option('display.max_columns', None)
    df = pd.read_pickle('Results_dataframe/Full_with_AUC_test_4.pkl')
    for i, row in df.iterrows():
        if i != 0:
            continue
        hyper_dict = utils.read_hyperparameters(row, i)
        # hyper_dict['n_epoch'] = 1
        stage_spec, phase_spec, ventricle, is_conc = utils.fit_name(hyper_dict['stage'], hyper_dict['phase'],
                                                                    hyper_dict['ventricle'])
        data_dir, dis_dir = utils.get_both_dir(stage_spec, phase_spec, ventricle)
        dpr_range = list(range(750, 800, 25))
        dpr_range = [x / 1000 for x in dpr_range]
        print(dpr_range)
        code = 8205
        for dpr in dpr_range:
            parser = argparse.ArgumentParser()
            parser.add_argument('--code', type=int, default=code)
            parser.add_argument('--dpr', type=float, default=dpr)
            parser.add_argument('--batch_size', type=int, default=20)
            parser.add_argument('--base_lr', type=float, default=1e-6)
            parser.add_argument('--num_epochs', type=int, default=100)
            parser.add_argument('--nfold', type=int, default=4)  # This cannot be changed.
            parser.add_argument('--reduce_size', type=int, default=3)
            parser.add_argument('--num_workers', type=int, default=1)
            parser.add_argument('--val_prop', type=float, default=0.2)
            parser.add_argument('--fold_count', type=int, default=0)
            parser.add_argument('--data_dir', type=str, default=data_dir)
            parser.add_argument('--disease_dir', type=str, default=dis_dir)
            parser.add_argument('--model_path', type=str, default='Model_history/New2')
            parser.add_argument('--AUC_path', type=str, default='AUC_history/New2')
            parser.add_argument('--epoch_interval', type=int, default=10)
            args1 = parser.parse_args()

            ds = load_dataset(args1, is_conc)
            total_CV_idx = utils_train.pick_cv_index(stage_key=hyper_dict['stage'])
            fold_history = utils_train.cross_validation2(utils_train.train_AUC_tune, ds, total_CV_idx, args1)

            fold_history['code'] = args1.code
            fold_history['dpr'] = args1.dpr
            fold_history['Num_epochs'] = args1.num_epochs
            utils_train.save_AUC_history(fold_history, args1)

            code += 1
        # ave_performance, f1 = utils.print_final_performance(fold_history, if_plot=True)
        # f2 = utils.history_plot(fold_history, data_name='Validation AUC', is_fold=True)
        # AUC = utils_train.find_AUC(fold_history)
        # print(f'Final test AUC: {AUC}')
        # plt.show()



