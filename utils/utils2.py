"""
SummerResearch.utils11
"""

import numpy as np
import utils.utils1 as utils
from torch.utils.data import DataLoader, random_split
from colorama import Fore, Back, Style
import random
from utils.data_utils2 import use_root_dir
import pickle
import os
from utils.utils1 import to_numpy
from math import ceil
from utils.dataset_utils import index_list_list


def look_ds_distribution(ds):
    count0, count1, count2 = 0, 0, 0
    is_1st = True
    for x, y in ds:
        y = int(to_numpy(y))
        if y == 0 and is_1st:
            count0 += 1
        elif y == 1:
            is_1st = False
            count1 += 1
            is_3rd = True
        elif y == 0 and is_3rd:
            count2 += 1
    return count0, count1, count2


def balance_train_idx(ds, train_idx):
    """
    This function adds the extra data of class 0 into the training set.
    But it still ensures class balance by duplicating the class 1 data.
    :param ds:
    :param train_idx:
    :return:
    """
    n0 = ds.get_class_n(0)
    n1 = ds.get_class_n(1)
    n_extra = n0 - n1
    list0, list1 = pick_0_and_1(ds, train_idx)
    list0_extra = list(range(n1*2, len(ds)))
    if n_extra <= len(list1):
        list1_extra = random.sample(list1, n_extra)
    else:
        list1_extra = random.choices(list1, k=n_extra)
    list_final = list0 + list0_extra + list1 + list1_extra
    random.shuffle(list_final)
    return list_final


def check_larger(train_idx, n_reference):
    is_larger = True
    for i in train_idx:
        if i <= n_reference:
            is_larger = False
    return is_larger


def pick_0_and_1(ds, train_idx):
    list0 = []
    list1 = []
    for i in train_idx:
        x, y = ds[i]
        y = int(to_numpy(y))
        if y == 0:
            list0.append(i)
        elif y == 1:
            list1.append(i)
    return list0, list1


def get_idx(train_idx_, proportion_):
    n_val = int(len(train_idx_)*proportion_)
    n_train = len(train_idx_) - n_val
    random.shuffle(train_idx_)
    train_idx_out = train_idx_[:n_train]
    val_idx_out = train_idx_[n_train:]

    return train_idx_out, val_idx_out


def check_balance(index_list, dataset_in):
    count0, count1 = 0, 0
    for idx in index_list:
        x, y = dataset_in[idx]
        y = int(to_numpy(y))
        if y == 1:
            count1 += 1
        elif y == 0:
            count0 += 1
    if count0 == count1:
        return True
    else:
        return False


def k_fold_idx(dataset_in, k, validation_prop):
    """
    This function returns the indices required to do K fold cross validation.
    output[1] = [train indices, validation indices, test indices] for the 2nd fold.
    :param dataset_in:
    :param k:
    :param validation_prop:
    :return:
    """
    total_idx = []
    idx0, idx1 = check_0_and_1(dataset_in)
    dev_test_list = k_fold_idx_dev(k, len(idx0))
    for fold in range(k):
        dev_idx_list, test_idx_list = dev_test_list[fold]
        train_idx_list, val_idx_list = get_idx(dev_idx_list, validation_prop)
        train_list = index_list_list(idx0, train_idx_list) + index_list_list(idx1, train_idx_list)
        val_list = index_list_list(idx0, val_idx_list) + index_list_list(idx1, val_idx_list)
        test_list = index_list_list(idx0, test_idx_list) + index_list_list(idx1, test_idx_list)
        random.shuffle(train_list)
        random.shuffle(val_list)
        random.shuffle(test_list)
        total_idx.append([train_list, val_list, test_list])
    return total_idx


def k_fold_idx_dev(k, num_total):
    """
    This function returns a list where list[i] contains the lists of indices for the development and test set for the
    i+1 th fold.
    i.e list[i] = [list_dev, list_test]
    :param k: number of folds
    :param num_total: total number of data in the dataset to be cross-validated.
    :return:
    """
    num_test = int(ceil(num_total/k))
    dev_test_list = []
    for fold in range(k):
        test_list = []
        total_list = list(range(num_total))
        start_idx = fold * num_test
        end_idx = start_idx + num_test
        if end_idx >= num_total:
            end_idx = num_total
        idx_check = 0
        for i in range(start_idx, end_idx):
            if idx_check == 0:
                pop_index = i
            test_elem = total_list.pop(pop_index)
            test_list.append(test_elem)
            idx_check += 1
        dev_list = total_list
        dev_test_list.append([dev_list, test_list])
    return dev_test_list


def check_0_and_1(dataset_in):
    """
    This function checks the number of positive and negative cases in the dataset.
    It returns lists of indices of positive and negative cases.
    :param dataset_in: The overall dataset.
    :return: lists of indices for the positive and negative cases in the dataset
    """
    idx_list0, idx_list1 = [], []
    for i, (x, y) in enumerate(dataset_in):
        y = int(to_numpy(y))
        if y == 0:
            idx_list0.append(i)
        elif y == 1:
            idx_list1.append(i)
    if not len(idx_list0) == len(idx_list1):
        print(Fore.RED + 'Error: the dataset is not balanced.' + Style.RESET_ALL)
    random.shuffle(idx_list0)
    random.shuffle(idx_list1)
    return idx_list0, idx_list1


def duplicate_train_idx(train_idx_old, dataset_in):
    list0 = []
    list1 = []
    # Count the number of positive and negative cases and record them.
    for idx in train_idx_old:
        temp, y = dataset_in[idx]
        y = utils.to_numpy(y)
        if y == 0:
            list0.append(idx)
        elif y == 1:
            list1.append(idx)
    num0 = len(list0)
    num1 = len(list1)
    num_extra = np.abs(num1 - num0)
    if num0 > num1:
        if num_extra > num1:
            train_idx_extra = random.choices(train_idx_old, k=num_extra)
        else:
            train_idx_extra = random.sample(train_idx_old, num_extra)
        train_idx_new = [*train_idx_old, *train_idx_extra]
        return train_idx_new
    elif num0 < num1:
        print(Fore.RED + 'Error: number of negative cases is smaller than that of positive cases.' + Style.RESET_ALL)


def load_dataset(args, ds):
    # This function returns the overall dataset, training dataset, validation dataset and test dataset based on the
    # proportions given .
    train_prop = args.train_prop
    val_prop = args.val_prop
    latent_ds = ds  # Overall dataset
    train_size = int(train_prop * len(latent_ds))
    train_ds, dev_ds = random_split(latent_ds, [train_size, len(latent_ds) - train_size])  # Split dataset
    val_size = int(val_prop * len(dev_ds))
    val_ds, test_ds = random_split(dev_ds, [val_size, len(dev_ds) - val_size])
    return train_ds, val_ds, test_ds


def get_data_loaders(args, ds):
    train_ds, val_ds, test_ds = load_dataset(args, ds)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                         pin_memory=True)

    return train_dl, val_dl, test_dl


def model_path(code, path):
    parent_dir = os.getcwd() + '/' + path + '/model'
    dir_out = parent_dir + str(code) + '.pt'
    return dir_out


def get_data_dir1(stage_spec):
    """
    This function returns the absolute directories of the healthy data and diseased data.
    :param stage_spec: 'i' indicates incidence and 'p' indicates prevalence.
    :return:
    """
    dir0 = 'data/Summer Research/healthy_dataset'
    dir1 = 'data/Summer Research/prevalent_mi_cases'
    if stage_spec == 'i':
        dir1 = 'data/Summer Research/incident_mi_cases'
    return use_root_dir(dir0), use_root_dir(dir1)


def is_picklable(obj):
    try:
        pickle.dumps(obj)
    except Exception:
        return False
    return True

