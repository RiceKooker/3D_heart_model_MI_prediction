"""This is the old New.utils."""
import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import random_split
import utils.data_utils as datatools
# from pyntcloud import PyntCloud
# import open3d as o3d
import os
from colorama import Fore, Back, Style
import random


def get_fold_keys(fold_history):
    fold_keys = []
    for key1 in fold_history:
        if 'Fold' not in key1:
            continue
        fold_keys.append(key1)
    return fold_keys


def delete_last_elements(train_history, num):
    if num == 0:
        return train_history
    for key1 in train_history:
        if len(train_history[key1]) > num:
            del train_history[key1][-num:]
    return train_history


def find_max_min(list_in, num, if_max=True):
    """
    This function finds the largest/smallest numbers in a list.
    :param list_in:
    :param num: number of largest/smallest numbers
    :param if_max: 'True' indicates finding maximum
    :return: the largest/smallest numbers and their indices
    """
    fun1 = max
    if not if_max:
        fun1 = min
    temp = list_in[:]
    extreme_list, extreme_index = [], []
    for i in range(num):
        extreme_list.append(fun1(temp))
        temp.remove(fun1(temp))

    for extreme in extreme_list:
        for i, value in enumerate(list_in):
            if value == extreme:
                extreme_index.append(i)
                break
    return extreme_list, extreme_index


def read_hyp_from_code(code, df):
    row = df.loc[df['Exp id'] == code]
    i = row.index.values.astype(int)[0]
    dict_out = read_hyperparameters(row, i)
    return dict_out


def read_hyperparameters(df_row, i):
    print(Fore.RED + 'Processing row {}...'.format(i) + Style.RESET_ALL)
    stage_spec_ = df_row['Stage specification']
    phase_spec_ = df_row['Phase specification']
    ventricle_ = df_row['Ventricle']
    exp_id = df_row['Exp id']
    n_epoch = df_row['Num of epochs']
    dpr = df_row['Droppout rate']
    return {'stage': stage_spec_, 'phase': phase_spec_, 'ventricle': ventricle_, 'exp_id': exp_id, 'n_epoch': n_epoch,
            'dpr': dpr}


def fit_name(stage_spec_, phase_spec_, ventricle_):
    is_conc = False
    if stage_spec_ == 'Prevalence':
        stage_spec = 'p'
    elif stage_spec_ == 'Incidence':
        stage_spec = 'i'
    if phase_spec_ == 'Concatenated':
        phase_spec = 'Conc'
        is_conc = True
    else:
        phase_spec = phase_spec_
    if ventricle_ == 'Right':
        ventricle = 'rv'
    elif ventricle_ == 'Left':
        ventricle = 'lv'
    elif ventricle_ == 'Full':
        ventricle = 'full'
    return stage_spec, phase_spec, ventricle, is_conc


def get_both_dir(stage_spec, phase_spec, ventricle_spec):
    """
    This function returns the target data directories for both healthy and mi cases.
    :param stage_spec: 'i' (stands for incident) or 'p' (stands for prevalent).
    :param ventricle_spec: 'lv', 'rv' or 'full.
    :param phase_spec: 'ES', 'ED', or 'Conc'.
    :return: the absolute directories of the data.
    """
    healthy_dir = get_ventricle_dir('healthy', stage_spec, phase_spec, ventricle_spec)
    disease_dir = get_ventricle_dir('mi', stage_spec, phase_spec, ventricle_spec)
    return healthy_dir, disease_dir


def get_ventricle_dir(healthy_status, stage_spec, phase_spec, ventricle_spec):
    """
    This function returns the target data directory.
    :param healthy_status: 'healthy' or 'mi'.
    :param stage_spec: 'i' (stands for incident) or 'p' (stands for prevalent).
    :param ventricle_spec: 'lv', 'rv' or 'full.
    :param phase_spec: 'ES', 'ED', or 'Conc'.
    :return: the absolute directory of the data.
    """
    base_dir = 'data/Summer Research/'
    health_key = ['healthy', 'incident', 'prevalent']

    sub_dir1 = ''
    filler = '_'
    filler1 = '_cases'
    filler2 = '_mi'
    if healthy_status == 'healthy':
        if ventricle_spec == 'lv' or ventricle_spec == 'rv':
            sub_dir1 = sub_dir1 + health_key[0] + filler1 + filler + ventricle_spec
        elif ventricle_spec == 'full':
            sub_dir1 = sub_dir1 + health_key[0] + filler + 'dataset'
    elif healthy_status == 'mi':
        if stage_spec == 'i':
            sub_dir1 = sub_dir1 + health_key[1] + filler2 + filler1
        elif stage_spec == 'p':
            sub_dir1 = sub_dir1 + health_key[2] + filler2 + filler1
        if ventricle_spec == 'lv' or ventricle_spec == 'rv':
            sub_dir1 = sub_dir1 + filler + ventricle_spec

    if phase_spec == 'ES' or phase_spec == 'ED':
        sub_dir2 = '/' + phase_spec
        sub_dir1 += sub_dir2

    return datatools.use_root_dir(base_dir + sub_dir1)


def add_list(a, b):
    """
    Add two lists a and b element-wise
    a and b should have the same number of elements
    :param a:
    :param b:
    :return:
    """
    c = [x + y for x, y in zip(a, b)]

    if len(a) == 0:
        c = b
    elif len(b) == 0:
        c = a
    return c


def divide_list(A, a):
    """
    Divide each element in A by a
    :param A:
    :param a:
    :return:
    """
    c = [x / a for x in A]
    return c


def print_final_performance(fold_dict, if_plot=False, if_print=True):
    Ave_history = init_train_history()
    testacc_count = 0
    fold_count = 0
    plot_key_list = ['training_acc_epoch', 'validation_acc_epoch', 'training_loss_epoch', 'validation_loss_epoch']

    for key in fold_dict:
        if'Fold' not in key:
            continue
        train_history1 = fold_dict[key]
        for key1 in plot_key_list:
            Ave_history[key1] = add_list(Ave_history[key1], train_history1[key1])
        test_acc_ = train_history1['Test accuracy']
        testacc_count += test_acc_
        fold_count += 1

    for key in Ave_history:
        Ave_history[key] = divide_list(Ave_history[key], fold_count)
    final_train_acc = Ave_history['training_acc_epoch'][-1]
    final_val_acc = Ave_history['validation_acc_epoch'][-1]
    final_train_loss = Ave_history['training_loss_epoch'][-1]
    final_val_loss = Ave_history['validation_loss_epoch'][-1]

    AUC_std = find_AUC_std(fold_dict)

    if if_print:
        print(Fore.MAGENTA + '\n Final train accuracy: {0:.2f}'.format(final_train_acc * 100) + Style.RESET_ALL)
        print(Fore.MAGENTA + 'Final val accuracy: {0:.2f}'.format(final_val_acc * 100) + Style.RESET_ALL)
        print(Fore.MAGENTA + 'Final test accuracy: {0:.2f}'.format(testacc_count * 100 / fold_count) + Style.RESET_ALL)
        print(Fore.BLUE + 'Final train loss: {0:.3f}'.format(final_train_loss) + Style.RESET_ALL)
        print(Fore.BLUE + 'Final val loss: {0:.3f}'.format(final_val_loss) + Style.RESET_ALL)
    if if_plot:
        f1 = loss_plot(Ave_history)
        return {'Training acc': final_train_acc * 100, 'Validation acc: ': final_val_acc * 100,
                'Test acc': testacc_count * 100 / fold_count, 'Training loss': final_train_loss,
                'Validation loss': final_val_loss,
                'Std': AUC_std}, f1
    return {'Training acc': final_train_acc * 100, 'Validation acc': final_val_acc * 100,
            'Test acc': testacc_count * 100 / fold_count, 'Training loss': final_train_loss,
            'Validation loss': final_val_loss,
            'Std': AUC_std}


def read_train_history(train_history):
    final_train_acc = train_history['training_acc_epoch'][-1]
    final_val_acc = train_history['validation_acc_epoch'][-1]
    final_train_loss = train_history['training_loss_epoch'][-1]
    final_val_loss = train_history['validation_loss_epoch'][-1]
    return final_train_acc, final_val_acc, final_train_loss, final_val_loss


def print_final_performance2(fold_dict, if_print=True):
    Ave_history = init_train_history()
    testacc_count = 0
    fold_count = 0
    for key in fold_dict:
        train_history1 = fold_dict[key]
        for key1 in Ave_history:
            Ave_history[key1] = add_list(Ave_history[key1], train_history1[key1])
        test_acc_ = train_history1['Test accuracy']
        testacc_count += test_acc_
        fold_count += 1

    for key in Ave_history:
        Ave_history[key] = divide_list(Ave_history[key], fold_count)

    final_train_acc = Ave_history['training_acc_epoch'][-1]
    final_val_acc = Ave_history['validation_acc_epoch'][-1]
    final_train_loss = Ave_history['training_loss_epoch'][-1]
    final_val_loss = Ave_history['validation_loss_epoch'][-1]

    AUC_std = find_AUC_std(fold_dict)

    if if_print:
        print(Fore.MAGENTA + 'Final train accuracy: {0:.2f}'.format(final_train_acc * 100) + Style.RESET_ALL)
        print(Fore.MAGENTA + 'Final val accuracy: {0:.2f}'.format(final_val_acc * 100) + Style.RESET_ALL)
        print(Fore.MAGENTA + 'Final test accuracy: {0:.2f}'.format(testacc_count * 100 / fold_count) + Style.RESET_ALL)
        print(Fore.BLUE + 'Final train loss: {0:.3f}'.format(final_train_loss) + Style.RESET_ALL)
        print(Fore.BLUE + 'Final val loss: {0:.3f}'.format(final_val_loss) + Style.RESET_ALL)

    return {'Training acc': final_train_acc * 100, 'Validation acc: ': final_val_acc * 100,
            'Test acc': testacc_count * 100 / fold_count, 'Training loss': final_train_loss,
            'Validation loss': final_val_loss,
            'Std': AUC_std}


def ROC_plot(x1, y1, x2, y2, x3, y3):
    # Plot
    parameters = {'axes.labelsize': 13,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'legend.fontsize': 11}
    plt.rcParams.update(parameters)
    plot = plt.figure()
    plt.plot(x1, y1, label='ED')
    plt.plot(x2, y2, label='ES')
    plt.plot(x3, y3, label='Concatenated')
    plt.plot(np.linspace(0, 1, num=100), np.linspace(0, 1, num=100), color='k', label='TPR = FPR')
    plt.grid(color='0.95')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.xlim((0, 1))
    plt.ylim((0, 1.005))
    # plt.title(title)
    return plot


# This
def sort_2_list(list1, list2):
    zipped_lists = zip(list1, list2)
    sorted_pairs = sorted(zipped_lists)

    tuples = zip(*sorted_pairs)
    list1, list2 = [list(tuple) for tuple in tuples]

    return list1, list2


# This
def find_AUC(Ave_FPR, Ave_TPR):
    Ave_FPR, Ave_TPR = sort_2_list(Ave_FPR, Ave_TPR)
    AUC = np.trapz(Ave_TPR, Ave_FPR)
    return AUC


def find_AUC_std(fold_history):
    AUC_list = []
    for key1 in fold_history:
        if 'Fold' not in key1:
            continue
        train_history = fold_history[key1]
        AUC_list.append(find_AUC(train_history['ROC_data_FPR'], train_history['ROC_data_TPR']))
    AUC_std = np.std(AUC_list)
    return AUC_std


def get_idx(train_idx, proportion):
    n_val = int(len(train_idx) * proportion)
    n_train = len(train_idx) - n_val
    random.shuffle(train_idx)
    train_idx_out = train_idx[:n_train]
    val_idx_out = train_idx[n_train:]
    return train_idx_out, val_idx_out


def loss_plot(train_history, no_acc=False):
    # f1 = performance_plot(train_history['training_loss'], train_history['validation_loss'],
    #                       title='Loss at each iteration/batch', x_label='Iteration', y_label='Loss', zero_start=False)
    # f2 = performance_plot(train_history['training_acc'], train_history['validation_acc'],
    #                       title='Accuracy at each iteration/batch', x_label='Iteration', y_label='Accuracy',
    #                       zero_start=False)
    f3 = performance_plot(train_history['training_loss_epoch'], train_history['validation_loss_epoch'],
                          title='Averaged loss at each epoch', x_label='Epoch', y_label='Loss')
    f4 = performance_plot(train_history['training_acc_epoch'], train_history['validation_acc_epoch'],
                          title='Averaged accuracy at each epoch', x_label='Epoch', y_label='Accuracy')
    # if no_acc:
    #     return f1, f3
    # else:
    #     return f1, f2, f3, f4
    return f3, f4


def fold_plot(train_history, fold_count):
    f1 = performance_plot_fold(train_history['training_loss_epoch'], train_history['validation_loss_epoch'],
                               title='Averaged loss at each epoch', x_label='Epoch', y_label='Loss',
                               fold_count=fold_count)
    f2 = performance_plot_fold(train_history['training_acc_epoch'], train_history['validation_acc_epoch'],
                               title='Averaged accuracy at each epoch', x_label='Epoch', y_label='Accuracy',
                               fold_count=fold_count)
    return f1, f2


def performance_plot(training_history, validation_history, title, x_label, y_label, zero_start=True):
    is_epoch = len(training_history) == len(validation_history)
    x_train = range(1, len(training_history) + 1)
    x_val = x_train
    if not is_epoch:
        x_val = range(1, len(validation_history) + 1)
    if zero_start:
        x_train = range(0, len(x_train))
        x_val = range(0, len(x_val))
    y_train = training_history
    y_val = validation_history

    # Plot
    parameters = {'axes.labelsize': 13,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'legend.fontsize': 11}
    plt.rcParams.update(parameters)
    plot = plt.figure()
    plt.plot(x_train, y_train, label='Training')
    plt.plot(x_val, y_val, label='Validation')
    plt.legend(frameon=False)
    plt.grid(color='0.95')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    return plot


def history_plot(train_history, data_name, is_fold=False, x_label='Epoch', y_label=None):
    if is_fold:
        ave_list = []
        fold_num = 0
        fold_list = []
        for key in train_history:
            if 'Fold' not in key:
                continue
            fold_list.append(key)
        for fold_count, fold_key in enumerate(fold_list):
            ave_list = add_list(ave_list, train_history[fold_key][data_name])
            fold_num += 1
        ave_list = divide_list(ave_list, fold_num)
        y = ave_list
    else:
        y = train_history[data_name]

    x = range(len(y))
    parameters = {'axes.labelsize': 13,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'legend.fontsize': 11}
    plt.rcParams.update(parameters)
    plot = plt.figure()
    plt.plot(x, y)
    plt.grid(color='0.95')
    plt.xlabel(x_label)
    if y_label is None:
        y_label = data_name
    plt.ylabel(y_label)
    plt.title(data_name)

    return plot


def performance_plot_fold(training_history, validation_history, title, x_label, y_label, fold_count, zero_start=True):
    is_epoch = len(training_history) == len(validation_history)
    x_train = range(1, len(training_history) + 1)
    x_val = x_train
    if not is_epoch:
        x_val = range(1, len(validation_history) + 1)
    if zero_start:
        x_train = range(0, len(x_train))
        x_val = range(0, len(x_val))
    y_train = training_history
    y_val = validation_history

    # Plot
    parameters = {'axes.labelsize': 13,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'legend.fontsize': 11}
    plt.rcParams.update(parameters)
    plot = plt.figure()
    plt.plot(x_train, y_train, label='Training')
    plt.plot(x_val, y_val, label='Validation')
    plt.legend(frameon=False)
    plt.grid(color='0.95')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title + ' - ' + 'Fold ' + str(fold_count))

    return plot


def ROC_plot(x, y, x_label, y_label):
    # Plot
    parameters = {'axes.labelsize': 13,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'legend.fontsize': 11}
    plt.rcParams.update(parameters)
    plot = plt.figure()
    plt.plot(x, y)
    plt.grid(color='0.95')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    # plt.title(title)
    return plot


def regression_plot(network, data_loader, title_name):
    '''
    This function plots the regression plot of a regression network given a data loader and a title
    :param network:
    :param data_loader:
    :param title_name:
    :return:
    '''
    # Training set performance
    pred, labels = get_labels_and_prediction(network, data_loader)
    z = np.polyfit(labels, pred, 1)

    parameters = {'axes.labelsize': 13,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'legend.fontsize': 11}
    plt.rcParams.update(parameters)
    plt.rcParams['axes.axisbelow'] = True
    plot = plt.figure()
    plt.plot(labels, labels, color='#2ca02c', label='True age')
    plt.scatter(labels, pred, s=10, label='Predictions')
    plt.plot(labels, labels * z[0] + z[1], color='#cf1d1d', label='Line of best fit')
    plt.legend(frameon=False)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.xlabel('True age')
    plt.ylabel('Predicted age')
    plt.grid(color='0.95')
    plt.title(title_name)
    return plot


def evaluate(model, criterion, data_loader, is_regression=False, is_multi_class=False):
    if is_regression:
        return evaluate_regression(model, criterion, data_loader)
    elif is_multi_class:
        return evaluate_multi_class(model, criterion, data_loader)
    else:
        return evaluate_binary1(model, criterion, data_loader)


def evaluate_regression(network, criterion, data_loader):
    """
    Evaluation function for PointNet regression
    """
    device = get_default_device()
    losses, total_seen, total_correct = [], 0, 0
    network.eval()
    for x, y in data_loader:
        y = y.to(device)
        with torch.no_grad():
            yhat = network(x.to(device))
            loss = criterion(yhat, y)
            losses.append(loss.item())
            total_seen += x.shape[0]
    mean_loss = np.mean(losses)
    print(Fore.GREEN + 'Regression evaluation completed!' + Style.RESET_ALL)
    print('Loss: {:.4f}'.format(mean_loss))


def evaluate_multi_class(network, criterion, data_loader):
    device = get_default_device()
    losses, total_seen, total_correct = [], 0, 0
    network.eval()
    for x, y in data_loader:
        y = y.to(device)
        y = y.view(-1)
        with torch.no_grad():
            yhat = network(x.to(device))
            loss = criterion(yhat, y)
            yhat = torch.max(yhat, dim=-1)[1]
            total_correct += float((yhat == y).sum().item())
            losses.append(loss.item())
            total_seen += x.shape[0]
    print(Fore.GREEN + 'Multi-class evaluation completed!' + Style.RESET_ALL)
    print('Corr: {}, Seen: {}, Acc: {:.4f}'.format(total_correct, total_seen, total_correct / float(total_seen)))


def evaluate_binary1(network, criterion, val_dl):
    """
    Binary evaluation for PointNet
    :param network:
    :param criterion:
    :param val_dl:
    :return:
    """
    device = get_default_device()
    losses, total_seen, total_correct = [], 0, 0
    network.eval()
    for x, y in val_dl:
        y = y.to(device)
        with torch.no_grad():
            yhat = network(x.to(device))
            loss = criterion(yhat, y)
            yhat = get_gender(yhat)
            total_correct += float((yhat == y).sum().item())
            losses.append(loss.item())
            total_seen += x.shape[0]
    print(Fore.GREEN + 'PointNet: Binary evaluation completed!' + Style.RESET_ALL)
    print('Corr: {}, Seen: {}, Acc: {:.4f}'.format(total_correct, total_seen, total_correct / float(total_seen)))


def evaluate_binary2(network, criterion, val_dl):
    """
    Binary evaluation for PointNet2
    :param network:
    :param criterion:
    :param val_dl:
    :return:
    """
    device = get_default_device()
    losses, total_seen, total_correct = [], 0, 0
    network.eval()
    for x, y in val_dl:
        y = y.to(device)
        with torch.no_grad():
            yhat = network(x.to(device), x.to(device))  # x.shape = [batch_size, npoints, 3]
            loss = criterion(yhat, y)
            yhat = get_gender(yhat)
            total_correct += float((yhat == y).sum().item())
            losses.append(loss.item())
            total_seen += x.shape[0]
    print(Fore.GREEN + 'Binary evaluation completed!' + Style.RESET_ALL)
    print('Corr: {}, Seen: {}, Acc: {:.4f}'.format(total_correct, total_seen, total_correct / float(total_seen)))


def evaluate_binary2_output(network, criterion, val_dl):
    """
    Binary evaluation for PointNet2
    :param network:
    :param criterion:
    :param val_dl:
    :return:
    """
    device = get_default_device()
    losses, total_seen, total_correct = [], 0, 0
    network.eval()
    for x, y in val_dl:
        y = y.to(device)
        with torch.no_grad():
            yhat = network(x.to(device))  # x.shape = [batch_size, npoints, 3]
            loss = criterion(yhat, y)
            yhat = get_gender(yhat)
            total_correct += float((yhat == y).sum().item())
            losses.append(loss.item())
            total_seen += x.shape[0]
    return total_correct / float(total_seen)


def evaluate_ROC_with_k(network, criterion, val_dl, k):
    confusion_dict = {}
    device = get_default_device()
    TP, TN, FP, FN = 0, 0, 0, 0
    network.eval()
    for x, y in val_dl:
        y = y.to(device)
        with torch.no_grad():
            yhat = network(x.to(device))  # x.shape = [batch_size, npoints, 3]
            yhat = get_gender_ROC(yhat, k)
            tp, tn, fp, fn = find_confusion_values(y, yhat)
            TP += tp
            TN += tn
            FP += fp
            FN += fn
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    confusion_dict['TP'] = TP
    confusion_dict['TN'] = TN
    confusion_dict['FP'] = FP
    confusion_dict['FN'] = FN
    confusion_dict['TPR'] = TPR
    confusion_dict['FPR'] = FPR

    return confusion_dict


def evaluate_ROC(network, criterion, val_dl):
    list_dict = {'TPR': [], 'FPR': [], 'TP': [], 'TN': [], 'FP': [], 'FN': []}
    K = np.linspace(0, 1, num=100)
    for k in K:
        confusion_dict = evaluate_ROC_with_k(network, criterion, val_dl, k)
        list_dict['TPR'].append(confusion_dict['TPR'])
        list_dict['FPR'].append(confusion_dict['FPR'])
        list_dict['TP'].append(confusion_dict['TP'])
        list_dict['TN'].append(confusion_dict['TN'])
        list_dict['FP'].append(confusion_dict['FP'])
        list_dict['FN'].append(confusion_dict['FN'])
    return list_dict


def find_confusion_values(y, yhat):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i, elem in enumerate(y):
        y_here = int(elem.item())
        yhat_here = yhat[i]
        yhat_here = int(yhat_here.item())
        if yhat_here == 1:
            if y_here == 1:
                TP += 1
            else:
                FP += 1
        else:
            if y_here == 1:
                FN += 1
            else:
                TN += 1
    return TP, TN, FP, FN


def evaluate_binary21(network, criterion, val_dl):
    """
    Binary evaluation for PointNet2
    :param network:
    :param criterion:
    :param val_dl:
    :return:
    """
    device = get_default_device()
    losses, total_seen, total_correct = [], 0, 0
    network.eval()
    for x, y in val_dl:
        y = y.to(device)
        with torch.no_grad():
            yhat = network(x.to(device), x.to(device))  # x.shape = [batch_size, npoints, 3]
            loss = criterion(yhat, y)
            yhat = get_gender(yhat)
            total_correct += float((yhat == y).sum().item())
            losses.append(loss.item())
            total_seen += x.shape[0]
    print(Fore.GREEN + 'Binary evaluation completed!' + Style.RESET_ALL)
    print('Corr: {}, Seen: {}, Acc: {:.4f}'.format(total_correct, total_seen, total_correct / float(total_seen)))


def get_labels_and_prediction(network, data_loader):
    """
    This function returns all labels from a data loader and the corresponding predictions produced by a network
    :param network:
    :param data_loader:
    :return:
    """
    pred = get_prediction(network, data_loader)
    labels = get_labels(data_loader)
    labels = labels.view(-1)
    labels = to_numpy(labels)
    pred = pred.view(-1)
    pred = to_numpy(pred)
    return pred, labels


def get_gender(y):
    device_name = y.device
    y = cpu_and_detach(y)
    y = np.array(y)
    y = np.where(y <= 0.5, 0, y)
    y = np.where(y > 0.5, 1, y)
    y = torch.tensor(y, device=device_name)
    return y


def get_gender_ROC(y, k):
    device_name = y.device
    y = cpu_and_detach(y)
    y = np.array(y)
    y = np.where(y <= k, 0, y)
    y = np.where(y > k, 1, y)
    y = torch.tensor(y, device=device_name)
    return y


def find_average(dataset):
    N = len(dataset)
    total = 0
    for i in range(N):
        data_pair = dataset[i]
        label = data_pair[1].item()
        total += label
    ave = total / N
    return ave


def load_dataset(args):
    # This function returns the overall dataset, training dataset, validation dataset and test dataset based on the
    # proportions given .
    train_prop = args.train_prop
    val_prop = args.val_prop
    latent_ds = datatools.latent_ds(args)  # Overall dataset
    train_size = int(train_prop * len(latent_ds))
    train_ds, dev_ds = random_split(latent_ds, [train_size, len(latent_ds) - train_size])  # Split dataset
    val_size = int(val_prop * len(dev_ds))
    val_ds, test_ds = random_split(dev_ds, [val_size, len(dev_ds) - val_size])
    return latent_ds, train_ds, val_ds, test_ds


def eval_net(network, x):
    device = get_default_device()
    out, trans, trans_feat = network(x.to(device))
    # out = network(x.to(device))
    return out


def get_prediction(network, data_loader):
    """
    This function returns all predictions of a regression PointNet given a data loader.
    The results are stored in cpu and detached.
    :param network:
    :param data_loader:
    :return:
    """
    n = get_number_dl(data_loader)
    device = get_default_device()
    prediction = torch.empty((n, 1))
    prediction = prediction.to(device=device)
    print('Getting all predictions from PointNet')
    data_index = 0
    network.eval()
    for x, y in data_loader:
        y = y.to(device)
        with torch.no_grad():
            yhat = network(x.to(device))
            prediction[data_index:(data_index + len(y))] = yhat
            data_index += len(y)
    prediction = cpu_and_detach(prediction)
    return prediction


def get_labels(data_loader):
    """
    This function returns all labels in a data loader.
    The results are stored in cpu and detached.
    :param data_loader:
    :return:
    """
    n = get_number_dl(data_loader)
    device_name = get_default_device()
    labels = torch.empty((n, 1))
    labels = labels.to(device=device_name)
    data_index = 0
    for x, y in data_loader:
        labels[data_index:(data_index + len(y))] = y
        data_index += len(y)
    labels = cpu_and_detach(labels)
    return labels


def get_number_dl(data_loader):
    """
    This function returns the number of data in a data loader
    :param data_loader:
    :return:
    """
    n = 0
    for x, y in data_loader:
        n += len(y)
    return n


def cpu_and_detach(a):
    a = a.cpu()
    a = a.detach()
    return a


def to_numpy(a):
    # Convert tensors in any device to numpy arrays
    a = cpu_and_detach(a)
    a = a.numpy()
    return a


# def find_volume(tensor_PC):
#     dim = tensor_PC.shape
#     if not dim[1] == 3:
#         tensor_PC = tensor_PC.permute(1, 0)
#     np_PC = to_numpy(tensor_PC)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(np_PC)
#     o3d.io.write_point_cloud("temp.ply", pcd)
#     sample_heart = PyntCloud.from_file("temp.ply")
#     convex_hull_id = sample_heart.add_structure("convex_hull")
#     convex_hull = sample_heart.structures[convex_hull_id]
#     volume = convex_hull.volume
#     volume = volume * 1e-3  # Convert from mm^3 to ml
#     os.remove("temp.ply")
#     return volume


def append_train_history(train_history, acc, loss, is_epoch=True, is_val=False):
    key_term = 'training'
    if is_val:
        key_term = 'validation'
    key_term_acc = key_term + '_acc'
    key_term_loss = key_term + '_loss'
    if is_epoch:
        key_term_acc = key_term_acc + '_epoch'
        key_term_loss = key_term_loss + '_epoch'

    train_history[key_term_acc].append(acc)
    train_history[key_term_loss].append(loss)
    return train_history


def init_train_history():
    train_history = {'training_loss': [], 'validation_loss': [], 'training_acc': [], 'validation_acc': [],
                     'training_loss_epoch': [], 'validation_loss_epoch': [], 'training_acc_epoch': [],
                     'validation_acc_epoch': []}
    return train_history


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def oddNumbers(l, r):
    odd_list = list(range(l if l % 2 else l + 1, r + 1, 2))
    return odd_list


def reduce_pc_size_2(input_vec):
    # This function reduces the size of the input point clouds by taking out odd rows of the input
    wanted_list = oddNumbers(1, len(input_vec))
    wanted_list = np.array(wanted_list) - 1
    reduced_vec = input_vec[wanted_list]
    return reduced_vec


def reduce_pc_size(input_vec, n_times):
    # This function performs odd-number reduction of a matrix n_times times
    if n_times == 0:
        return input_vec
    else:
        for i in range(n_times):
            input_vec = reduce_pc_size_2(input_vec)
    return input_vec


if __name__ == "__main__":
    pass
