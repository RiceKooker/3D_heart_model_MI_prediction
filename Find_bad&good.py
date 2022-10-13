"""
This script is to find the AUC.
The dataset used is for concatenated pointclouds.
Class balance is ensured.
This script uses trained models. No training is done in this script.
"""
import argparse
import utils.utils1 as utils
import dataset.my_dataset as my_dataset
from utils.training_utils import pick_extreme_cases_cv, get_total_extreme_cases, save_AUC_history, pick_cv_index
from utils.training_utils import check_model_loading
import pandas as pd
import numpy as np


def load_dataset(args, is_Conc=False):
    if is_Conc:
        ds1_ = my_dataset.Dataset194GetId(args)
        ds2_ = my_dataset.Dataset194(args)
    else:
        ds1_ = my_dataset.Dataset1941GetId(args)
        ds2_ = my_dataset.Dataset1941(args)

    return ds1_, ds2_


if __name__ == "__main__":
    total_dict = {}
    df = pd.read_pickle('Results_dataframe/Full_with_AUC_test_4.pkl')
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

        ds1, ds2 = load_dataset(args1, is_conc)
        total_CV_idx = pick_cv_index(stage_key=hyper_dict['stage'])
        model_is_right = check_model_loading(row['AUC'], args1, ds2, total_CV_idx)
        fold_history = pick_extreme_cases_cv(args1, ds1, total_CV_idx)
        case_dict = get_total_extreme_cases(fold_history)
        total_dict[args1.code] = case_dict
        np.save('Extreme_cases/12_each.npy', total_dict)
        keyword_list1 = ['tp', 'fp', 'fn', 'tn']
        keyword_list2 = ['id', 'score']
        # for k1 in keyword_list1:
        #     print(k1.upper())
        #     print('Case id' + ' ' + 'Score')
        #     for ids, scores in zip(case_dict[k1]['id'], case_dict[k1]['score']):
        #         print(ids, round(scores, 3))
        #     print('---------------------------------------------------------' + '\n')



