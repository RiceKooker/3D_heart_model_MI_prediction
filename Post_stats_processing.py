"""
This script is used to calculate some of the statistics from the data generated from past training.
"""
import argparse
import utils.utils1 as utils
import dataset.my_dataset as my_dataset
import utils.training_utils as utils_train
import pandas as pd


def load_dataset(args, is_Conc=False):
    if is_Conc:
        latent_ds = my_dataset.Dataset194(args)
    else:
        latent_ds = my_dataset.Dataset1941(args)
    return latent_ds


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    df = pd.read_pickle('Results_dataframe/Full_with_AUC_test_4.pkl')
    print(df.head())
    for i, row in df.iterrows():

        hyper_dict = utils.read_hyperparameters(row, i)
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

        fold_path = args1.AUC_path + '/fold_history_ROC_{}.npy'.format(args1.code)
        """
        First layer key: Fold1, Fold2...
        Second layer key: ROC_data_TPR, ROC_data_FPR, Confusion_info
        Confusion_info key: TP, TN...
        """
        fold_history = utils_train.load_fold_history(fold_path)
        std1 = utils.find_AUC_std(fold_history)
        df.loc[i, 'Std'] = std1
        df.to_pickle('Results_dataframe/Full_with_AUC_test_4.pkl')
