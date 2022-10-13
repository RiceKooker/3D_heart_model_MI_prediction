"""
This script is to find the AUC.
It reads off the test conditions from a dataframe, trains the network and report AUC values.
"""
import argparse
import utils.utils1 as utils
import dataset.my_dataset as my_dataset
import utils.training_utils as utils_train
import pandas as pd
import random
import numpy as np
import torch


def load_dataset(args, is_Conc=False):
    if is_Conc:
        latent_ds = my_dataset.Dataset194(args)
    else:
        latent_ds = my_dataset.Dataset1941(args)
    return latent_ds


if __name__ == "__main__":
    # pd.set_option('display.max_columns', None)
    # df = pd.read_pickle('Results_dataframe/Final_version.pkl')
    # for i, row in df.iterrows():
    #     if i != 20:
    #         continue
    #     print(row)
    #     hyper_dict = utils.read_hyperparameters(row, i)
    #
    #     hyper_dict['exp_id'] = 123456
    #
    #     stage_spec, phase_spec, ventricle, is_conc = utils.fit_name(hyper_dict['stage'], hyper_dict['phase'],
    #                                                                 hyper_dict['ventricle'])
    #     print(hyper_dict['stage'])
    #     data_dir, dis_dir = utils.get_both_dir(stage_spec, phase_spec, ventricle)
    #
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--code', type=int, default=hyper_dict['exp_id'])
    #     parser.add_argument('--dpr', type=float, default=hyper_dict['dpr'])
    #     parser.add_argument('--batch_size', type=int, default=20)
    #     parser.add_argument('--base_lr', type=float, default=1e-6)
    #     parser.add_argument('--num_epochs', type=int, default=hyper_dict['n_epoch'])
    #     parser.add_argument('--nfold', type=int, default=4)  # This cannot be changed.
    #     parser.add_argument('--reduce_size', type=int, default=3)
    #     parser.add_argument('--num_workers', type=int, default=1)
    #     parser.add_argument('--val_prop', type=float, default=0.2)
    #     parser.add_argument('--fold_count', type=int, default=0)
    #     parser.add_argument('--data_dir', type=str, default=data_dir)
    #     parser.add_argument('--disease_dir', type=str, default=dis_dir)
    #     args1 = parser.parse_args()
    #
    #     ds = load_dataset(args1, is_conc)
    #     total_CV_idx = pick_cv_index(stage_key=hyper_dict['stage'])
    #     fold_history = cross_validation2(train_AUC, ds, total_CV_idx, args1)
    #     fold_dict = fold_history
    #     ave_performance = utils.print_final_performance(fold_dict)
    #     Ave_TPR, Ave_FPR = find_ave_AUC_info(fold_dict)
    #     AUC = utils.find_AUC(Ave_FPR, Ave_TPR)
    #     print('AUC value: ', AUC)  # 0.6219

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pd.set_option('display.max_columns', None)
    df = pd.read_pickle('Results_dataframe/Full_with_AUC_test_5.pkl')
    for i, row in df.iterrows():
        hyper_dict = utils.read_hyperparameters(row, i)

        # Only redo the tests with incident cases
        if hyper_dict['stage'] == 'Incidence':
            hyper_dict['dpr'] = 0.725
            hyper_dict['n_epoch'] = 45
        else:
            continue

        # hyper_dict['exp_id'] += 1000

        stage_spec, phase_spec, ventricle, is_conc = utils.fit_name(hyper_dict['stage'], hyper_dict['phase'],
                                                                    hyper_dict['ventricle'])
        data_dir, dis_dir = utils.get_both_dir(stage_spec, phase_spec, ventricle)

        parser = argparse.ArgumentParser()
        parser.add_argument('--code', type=int, default=hyper_dict['exp_id'])
        parser.add_argument('--dpr', type=float, default=hyper_dict['dpr'])
        parser.add_argument('--batch_size', type=int, default=20)
        parser.add_argument('--base_lr', type=float, default=1e-6)
        parser.add_argument('--num_epochs', type=int, default=hyper_dict['n_epoch'])
        parser.add_argument('--nfold', type=int, default=4)  # This cannot be changed.
        parser.add_argument('--reduce_size', type=int, default=3)
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--val_prop', type=float, default=0.2)
        parser.add_argument('--fold_count', type=int, default=0)
        parser.add_argument('--data_dir', type=str, default=data_dir)
        parser.add_argument('--disease_dir', type=str, default=dis_dir)
        parser.add_argument('--model_path', type=str, default='Model_history/New')
        parser.add_argument('--AUC_path', type=str, default='AUC_history/New')
        args1 = parser.parse_args()

        ds = load_dataset(args1, is_conc)
        total_CV_idx = utils_train.pick_cv_index(stage_key=hyper_dict['stage'])
        # fold_history = utils_train.cross_validation2(utils_train.train_AUC, ds, total_CV_idx, args1)
        # ave_performance = utils.print_final_performance(fold_history)
        # utils_train.save_AUC_history(fold_history, args1)

        fold_history = utils_train.load_fold_history(args1.AUC_path + '/fold_history_ROC_{}.npy'.format(args1.code))

        AUC = utils_train.find_AUC(fold_history)
        # model_ok = utils_train.check_model_loading(AUC, args1, ds, total_CV_idx)
        val_AUC = utils_train.check_val_AUC(args1, ds, total_CV_idx)

        model_ok = True
        if model_ok:
            # df = utils_train.update_df2(df, i, ave_performance)
            # df = utils_train.update_df_any(df, i, Test_AUC=AUC, Droppout_rate=hyper_dict['dpr'],
            #                                Num_of_epochs=hyper_dict['n_epoch'], Exp_id=hyper_dict['exp_id'])
            df = utils_train.update_df_any(df, i, Validation_AUC=val_AUC)
            print(df.loc[i, :])
            df.to_pickle('Results_dataframe/Full_with_AUC_test_5.pkl')
        else:
            break
