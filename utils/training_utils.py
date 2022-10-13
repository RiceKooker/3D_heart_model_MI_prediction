import utils.utils1 as utils
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from colorama import Fore, Back, Style
import sys
from utils.utils2 import model_path, balance_train_idx
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.utils2 import check_balance
import pickle
from utils.dataset_utils import divide_list, add_list
import utils.dataset_utils as datatools
from utils.file_operation import change_file_name
from models.PointNetModels import PointNetClsGender, PointNetClsDis
from sys import exit


class TrainHistory:
    def __init__(self):
        self.history_dict = utils.init_train_history()

    def __call__(self, item, name):
        try:
            self.history_dict[name].append(item)
        except KeyError:
            self.history_dict[name] = []
            self.history_dict[name].append(item)

    def add_single_item(self, item, name):
        self.history_dict[name] = item

    def get_key_list(self):
        return list(self.history_dict.keys())

    def record_train(self, acc, loss):
        self.history_dict['training_loss_epoch'].append(loss)
        self.history_dict['training_acc_epoch'].append(acc)

    def record_val(self, acc, loss):
        self.history_dict['validation_loss_epoch'].append(loss)
        self.history_dict['validation_acc_epoch'].append(acc)


class EarlyStopping:
    def __init__(self, epoch_interval=5):
        self.best_metric = 0
        self.best_epoch = 0
        self.epoch_interval = epoch_interval
        self.early_stop = False

    def __call__(self, metric, epoch):
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_epoch = epoch
        if epoch >= self.best_epoch + self.epoch_interval:
            self.early_stop = True


class SaveModel:
    def __init__(self, epoch_interval=5):
        self.models = {}
        self.epoch_interval = epoch_interval
        self.epoch = 0

    def __call__(self, new_model, epoch):
        self.models[epoch] = new_model
        self.epoch = epoch
        if len(self.models) > self.epoch_interval:
            del self.models[epoch - self.epoch_interval]

    def save_best_model(self, args, if_stop):
        if if_stop:
            save_model_at_fold(self.models[0], args)
            print(f'Model training stops at {self.epoch}')
            print(f'Model saved at {self.epoch-self.epoch_interval}')


def find_new_stats(fold_history):
    for key1 in fold_history:
        train_history = fold_history[key1]
        confusion_info = train_history['Confusion_info']


def verify_scores_and_id(network, extreme_dict, dataset):
    network.eval()
    device = utils.get_default_device()
    for key1 in extreme_dict:
        dict_here = extreme_dict[key1]
        ids = dict_here['id']
        scores = dict_here['score']
        for each_id, each_score in zip(ids, scores):
            x, y, case_id = dataset.get_data_by_id(each_id)
            # print(f'ID from list: {each_id}')
            # print(f'ID from dataset function: {case_id}')
            # print(f'Shape of x: {x.shape}')  # 20, 3, 2250
            # print(f'Shape of y: {y.shape}')
            score_from_model = utils.to_numpy(network(x.to(device))).item()
            if not 0.999*each_score < score_from_model < 1.001*each_score:
                print(Fore.RED + f'Case {case_id} is validated unsuccessfully!' + Style.RESET_ALL)
                exit("Case id and score do not match.")


def get_total_extreme_cases(fold_history):
    tp = {'id': [], 'score': []}
    fp = {'id': [], 'score': []}
    fn = {'id': [], 'score': []}
    tn = {'id': [], 'score': []}
    out_dict = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
    for key in fold_history:
        fold_history_here = fold_history[key]
        out_dict['tp']['id'] += fold_history_here['tp']['id']
        out_dict['tp']['score'] += fold_history_here['tp']['score']
        out_dict['fp']['id'] += fold_history_here['fp']['id']
        out_dict['fp']['score'] += fold_history_here['fp']['score']
        out_dict['fn']['id'] += fold_history_here['fn']['id']
        out_dict['fn']['score'] += fold_history_here['fn']['score']
        out_dict['tn']['id'] += fold_history_here['tn']['id']
        out_dict['tn']['score'] += fold_history_here['tn']['score']
    return out_dict


def check_model_loading(AUC_known, args, dataset, total_CV_idx):
    fold_history = {}
    print(Fore.YELLOW + '\n Model validation is in progress...' + Style.RESET_ALL)
    for fold, (train_idx, val_idx, test_idx) in enumerate(total_CV_idx):
        test_loader, test_sampler = create_CV_data_loaders21(dataset, train_idx, val_idx, test_idx, args)
        network = load_model(int(str(args.code) + str(fold + 1)), args.model_path)
        criterion = nn.BCELoss()
        train_history = find_AUC_model(network, test_loader, criterion=criterion)
        fold_history['Fold{}'.format(fold + 1)] = train_history

    Ave_TPR, Ave_FPR = find_ave_AUC_info(fold_history)
    AUC = utils.find_AUC(Ave_FPR, Ave_TPR)

    print('AUC from training: ', AUC_known)
    print('AUC from loading: ', AUC)
    if 1.01*AUC_known > AUC > 0.99*AUC_known:
        print(Fore.GREEN + 'Model loading is validated by computing AUC!' + Style.RESET_ALL)
        return True
    else:
        print(Fore.RED + 'Model loading failed - inconsistent AUC values!' + Style.RESET_ALL)
        exit('Model loading error.')
        return False


def check_val_AUC(args, dataset, total_CV_idx):
    fold_history = {}
    print(Fore.YELLOW + '\n Validation AUC evaluation is in progress...' + Style.RESET_ALL)
    for fold, (train_idx, val_idx, test_idx) in enumerate(total_CV_idx):
        test_loader, test_sampler = create_CV_data_loaders_single(dataset, val_idx, args)
        network = load_model(int(str(args.code) + str(fold + 1)), args.model_path)
        criterion = nn.BCELoss()
        train_history = find_AUC_model(network, test_loader, criterion=criterion)
        fold_history['Fold{}'.format(fold + 1)] = train_history

    Ave_TPR, Ave_FPR = find_ave_AUC_info(fold_history)
    AUC = utils.find_AUC(Ave_FPR, Ave_TPR)

    print(f'Validation AUC: {AUC}')
    return AUC


def find_AUC_model(network, data_loader, criterion):
    train_history = utils.init_train_history()
    list_dict = utils.evaluate_ROC(network, criterion, data_loader)
    train_history['ROC_data_TPR'] = list_dict['TPR']
    train_history['ROC_data_FPR'] = list_dict['FPR']
    return train_history


def update_df2(df, i, ave_performance):
    for key1 in ave_performance:
        df.loc[i, key1] = ave_performance[key1]
    return df


def update_df_any(df, i, **kwargs):
    for key2 in kwargs:
        key3 = key2
        # This is to accommodate that the key words have spaces.
        if '_' in key2:
            key3 = key2.replace('_', ' ')
        df.loc[i, key3] = kwargs[key2]
    return df


def update_df(df, i, hyper_dict, ave_performance, AUC):

    df.loc[i, 'AUC'] = AUC
    for key1 in ave_performance:
        df.loc[i, key1] = ave_performance[key1]
    df.loc[i, ['Droppout rate', 'Num of epochs']] = hyper_dict['dpr'], hyper_dict['n_epoch']

    return df


def save_model_at_fold(network, args, para_only=False):
    code = int(str(args.code) + str(args.fold_count))
    if para_only:
        torch.save(network.state_dict(), model_path(code, args.model_path))
    else:
        torch.save(network, model_path(code, args.model_path))


def pick_cv_index(stage_key):
    if stage_key == 'Incidence':
        return load_CV_idx_file('DataDivision/Concatenated_incidence_idx.pkl')
    elif stage_key == 'Prevalence':
        return load_CV_idx_file('DataDivision/Concatenated_prevalence_idx.pkl')


def cross_validation(train_function, dataset, total_CV_idx, args):
    """
    This function performs cross validation given a certain train function.
    The network is set to be PointNetClsGender.
    :param train_function:
    :param dataset:
    :param total_CV_idx: index matrix for CV
    :param args:
    :return:
    """
    fold_history = {}
    for fold, (train_idx, val_idx, test_idx) in enumerate(total_CV_idx):
        print(Fore.YELLOW + '\n Fold {}'.format(fold + 1) + Style.RESET_ALL)
        train_loader, val_loader, test_loader = create_CV_data_loaders(dataset, train_idx, val_idx, test_idx, args)
        train_history1 = train_function(args, train_loader, val_loader, test_loader)
        fold_history['Fold{}'.format(fold+1)] = train_history1
    return fold_history


def cross_validation2(train_function, dataset, total_CV_idx, args):
    """
    This function performs cross validation given a certain train function.
    The network is set to be PointNetClsGender.
    :param train_function:
    :param dataset:
    :param total_CV_idx: index matrix for CV
    :param args:
    :return:
    """
    fold_history = {}
    for fold, (train_idx, val_idx, test_idx) in enumerate(total_CV_idx):
        print(Fore.YELLOW + '\n Fold {}'.format(fold + 1) + Style.RESET_ALL)
        train_loader, val_loader, test_loader = create_CV_data_loaders2(dataset, train_idx, val_idx, test_idx, args)
        train_history1 = train_function(args, train_loader, val_loader, test_loader)
        fold_history['Fold{}'.format(fold+1)] = train_history1
    return fold_history


def pick_extreme_cases_cv(args, dataset, total_CV_idx):
    fold_history = {}
    for fold, (train_idx, val_idx, test_idx) in enumerate(total_CV_idx):
        print(Fore.YELLOW + '\n Fold {}'.format(fold + 1) + Style.RESET_ALL)
        test_loader, test_sampler = create_CV_data_loaders21(dataset, train_idx, val_idx, test_idx, args)
        network = load_model(int(str(args.code) + str(fold+1)), args.model_path)
        extreme_dict = pick_extreme_cases(network, test_loader)
        verify_scores_and_id(network, extreme_dict, dataset)
        fold_history['Fold{}'.format(fold + 1)] = extreme_dict
    return fold_history


def pick_extreme_cases(network, data_loader, n=3):
    """
    This function picks the extreme cases in the data loader.
    They are TP, FP, FN and TN cases.
    :param network:
    :param data_loader:
    :param n:
    :return: a dictionary containing all the case ids and scores.
    """
    network.eval()
    device = utils.get_default_device()
    score0, id0, score1, id1 = [], [], [], []
    for x, y, case_id in data_loader:
        y = y.to(device)
        with torch.no_grad():
            yhat = network(x.to(device))  # This is the score.
            score_dict = divide_scores(y, yhat, case_id)
            score0 += score_dict[0]['score']
            score1 += score_dict[1]['score']
            id0 += score_dict[0]['id']
            id1 += score_dict[1]['id']
    score0_max, max_index0 = utils.find_max_min(score0, n)
    score1_max, max_index1 = utils.find_max_min(score1, n)
    score0_min, min_index0 = utils.find_max_min(score0, n, if_max=False)
    score1_min, min_index1 = utils.find_max_min(score1, n, if_max=False)
    extreme_dict = {}
    tp_dict = {'id': datatools.index_list_list(id1, max_index1), 'score': score1_max}
    fp_dict = {'id': datatools.index_list_list(id0, max_index0), 'score': score0_max}
    fn_dict = {'id': datatools.index_list_list(id1, min_index1), 'score': score1_min}
    tn_dict = {'id': datatools.index_list_list(id0, min_index0), 'score': score0_min}
    return {'tp': tp_dict, 'fp': fp_dict, 'fn': fn_dict, 'tn': tn_dict}


def divide_scores(label, scores, ids):
    dict0 = {'score': [], 'id': []}
    dict1 = {'score': [], 'id': []}
    for y, y_hat, case_id in zip(label, scores, ids):
        # When label is 0
        score = utils.to_numpy(y_hat).item()
        each_id = int(utils.to_numpy(case_id).item())
        if utils.to_numpy(y) == 0:
            dict0['score'].append(score)
            dict0['id'].append(each_id)
        else:
            dict1['score'].append(score)
            dict1['id'].append(each_id)
    return {0: dict0, 1: dict1}


def load_model(model_id, path):
    PATH = model_path(model_id, path)
    network = torch.load(PATH)
    network.eval()
    return network


def find_ave_AUC_info(fold_dict):
    """
    This function calculates the average TPR and FPR given a fold history.
    :param fold_dict:
    :return:
    """
    Ave_TPR = []
    Ave_FPR = []
    nfold = 0
    for key in utils.get_fold_keys(fold_dict):
        train_history_ = fold_dict[key]
        TPR = train_history_['ROC_data_TPR']
        FPR = train_history_['ROC_data_FPR']
        Ave_TPR = add_list(Ave_TPR, TPR)
        Ave_FPR = add_list(Ave_FPR, FPR)
        nfold += 1
    Ave_TPR = divide_list(Ave_TPR, nfold)
    Ave_FPR = divide_list(Ave_FPR, nfold)
    return Ave_TPR, Ave_FPR


def find_AUC(fold_dict):
    Ave_TPR, Ave_FPR = find_ave_AUC_info(fold_dict)
    Ave_FPR, Ave_TPR = utils.sort_2_list(Ave_FPR, Ave_TPR)
    AUC = np.trapz(Ave_TPR, Ave_FPR)
    return AUC


def find_AUC_in_fold(ROC_dict):
    Ave_FPR, Ave_TPR = ROC_dict['FPR'], ROC_dict['TPR']
    Ave_FPR, Ave_TPR = utils.sort_2_list(Ave_FPR, Ave_TPR)
    AUC = np.trapz(Ave_TPR, Ave_FPR)
    return AUC


def save_AUC_history(fold_history, args):
    np.save(args.AUC_path + '/fold_history_ROC_{}.npy'.format(args.code), fold_history)


def save_fold_history(fold_history, code):
    np.save('Fold_history/temp.npy', fold_history)
    change_file_name('fold', code)


def load_fold_history(his_dir):
    fold_dict = np.load(his_dir, allow_pickle=True).item()
    return fold_dict


def calculate_fold_ave(fold_dict):
    """
    This function calculates the average performance of a cross validation.
    :param fold_dict:
    :return:
    """
    Ave_history = utils.init_train_history()
    testacc_count = 0
    nfold = 0
    for key in fold_dict:
        train_history1 = fold_dict[key]
        for key1 in Ave_history:
            Ave_history[key1] = add_list(Ave_history[key1], train_history1[key1])
        test_acc_ = train_history1['Test accuracy']
        testacc_count += test_acc_
        nfold += 1
    test_acc_ave = testacc_count/nfold

    for key in Ave_history:
        Ave_history[key] = divide_list(Ave_history[key], nfold)

    return Ave_history, test_acc_ave


def load_CV_idx_file(file_dir):
    with open(file_dir, "rb") as fp:
        total_CV_idx = pickle.load(fp)
    return total_CV_idx


def create_CV_data_loaders(ds, train_idx, val_idx, test_idx, args):
    """
    This function creates the dataloaders given a dataset and the corresponding indices.
    :param ds:
    :param train_idx:
    :param val_idx:
    :param test_idx:
    :param args:
    :return:
    """
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = DataLoader(ds, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(ds, batch_size=args.batch_size, sampler=val_sampler)
    test_loader = DataLoader(ds, batch_size=args.batch_size, sampler=test_sampler)
    if check_balance(train_idx, ds) and check_balance(val_idx, ds) and check_balance(test_idx, ds):
        print(Fore.GREEN + 'Class balance is ensured in all data division')
    return train_loader, val_loader, test_loader


def create_CV_data_loaders2(ds, train_idx, val_idx, test_idx, args):
    """
    This function creates the dataloaders given a dataset and the corresponding indices.
    :param ds:
    :param train_idx:
    :param val_idx:
    :param test_idx:
    :param args:
    :return:
    """
    # print('Before duplicating: {}'.format(len(train_idx)))
    train_idx = balance_train_idx(ds, train_idx)
    # print('After duplicating: {}'.format(len(train_idx)))
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = DataLoader(ds, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(ds, batch_size=args.batch_size, sampler=val_sampler)
    test_loader = DataLoader(ds, batch_size=args.batch_size, sampler=test_sampler)
    if check_balance(train_idx, ds) and check_balance(val_idx, ds) and check_balance(test_idx, ds):
        print(Fore.GREEN + 'Class balance is ensured in all data division')
    return train_loader, val_loader, test_loader


def create_CV_data_loaders21(ds, train_idx, val_idx, test_idx, args):
    """
    This is an extension of create_CV_data_loaders2.
    It is used to find the representative cases.
    """
    test_sampler = SubsetRandomSampler(test_idx)
    test_loader = DataLoader(ds, batch_size=args.batch_size, sampler=test_sampler)
    return test_loader, test_sampler


def create_CV_data_loaders_single(ds, test_idx, args):
    """
    This is an extension of create_CV_data_loaders2.
    It is used to find the representative cases.
    """
    test_sampler = SubsetRandomSampler(test_idx)
    test_loader = DataLoader(ds, batch_size=args.batch_size, sampler=test_sampler)
    return test_loader, test_sampler


def train_one_epoch(train_dl, network, criterion, optimizer, device):
    """This function train the network for a complete epoch"""
    correct_total, loss_total, total_seen, losses = 0, 0, 0, []
    for data in train_dl:
        optimizer.zero_grad()
        x, y = data
        y = y.to(device)
        yhat = network(x.to(device))
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        yhat = utils.get_gender(yhat)
        correct_total += float((yhat == y).sum().item())
        losses.append(loss.item())
        total_seen += x.shape[0]
    mean_loss = np.mean(losses)
    acc = (correct_total / float(total_seen))
    return mean_loss, correct_total, total_seen, acc


def test_one_epoch(val_dl, network, criterion, device):
    """This function test the network with a validation dataloader for a complete epoch"""
    losses, total_seen, total_correct = [], 0, 0
    for x, y in val_dl:
        y = y.to(device)
        with torch.no_grad():
            yhat = network(x.to(device))
            loss = criterion(yhat, y)
            yhat = utils.get_gender(yhat)
            total_correct += float((yhat == y).sum().item())
            losses.append(loss.item())
            total_seen += x.shape[0]
    return np.mean(losses), total_correct, total_seen, total_correct/float(total_seen)


def train(args, train_dl, val_dl, test_dl, network_spec='3D'):
    # torch.manual_seed(0)
    train_history = utils.init_train_history()

    # Get device information
    device = utils.get_default_device()

    # Build network model and optimizer
    # network = PointNetClsGender(dp_r=args.dpr)
    network = PointNetClsGender(dp_r=args.dpr)
    if network_spec == '4D':
        network = PointNetClsDis(dp_r=args.dpr)
    network = network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.base_lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.75)
    criterion = nn.BCELoss()

    # Training
    for epoch in range(1, args.num_epochs + 1):
        b = "\033[33mTraining epoch {}/{} in progress\033[0m".format(epoch, args.num_epochs)
        sys.stdout.write('\r' + b)
        n_show = args.num_epochs
        network.eval()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loss, total_correct, total_seen, acc = test_one_epoch(val_dl, network, criterion, device)
        train_history = utils.append_train_history(train_history, acc, loss, is_val=True)
        # if epoch % n_show == 0:
        #     lr = optimizer.state_dict()['param_groups'][0]['lr']
        #     print('Validation Epoch: {} / {}, lr: {:.8f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'.format(
        #         epoch, args.num_epochs, lr, loss, total_correct, total_seen, acc))

        network.train()
        loss, total_correct, total_seen, acc = train_one_epoch(train_dl, network, criterion, optimizer, device)
        train_history = utils.append_train_history(train_history, acc, loss)
        # if epoch % n_show == 0:
        #     lr = optimizer.state_dict()['param_groups'][0]['lr']
        #     print('Train Epoch: {} / {}, lr: {:.8f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'.format(
        #         epoch, args.num_epochs, lr, loss, total_correct, total_seen, acc))
        scheduler.step()
    test_acc = utils.evaluate_binary2_output(network, criterion, test_dl)
    train_history['Test accuracy'] = test_acc
    args.fold_count += 1
    if args.fold_count == args.nfold:
        torch.save(network, model_path(args.code))
        print(Fore.BLUE + 'Model saved at fold: {}'.format(args.fold_count) + Style.RESET_ALL)
    return train_history


def train_AUC(args, train_dl, val_dl, test_dl, network_spec='3D'):
    # torch.manual_seed(0)
    train_history = utils.init_train_history()
    # Get device information
    device = utils.get_default_device()

    # Build network model and optimizer
    network = PointNetClsGender(dp_r=args.dpr)
    if network_spec == '4D':
        network = PointNetClsDis(dp_r=args.dpr)
    network = network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.base_lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.75)
    criterion = nn.BCELoss()

    # Training
    for epoch in range(1, args.num_epochs + 1):
        b = "\033[33mTraining epoch {}/{} in progress\033[0m".format(epoch, args.num_epochs)
        sys.stdout.write('\r' + b)
        n_show = args.num_epochs
        network.eval()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loss, total_correct, total_seen, acc = test_one_epoch(val_dl, network, criterion, device)
        train_history = utils.append_train_history(train_history, acc, loss, is_val=True)
        # if epoch % n_show == 0:
        #     lr = optimizer.state_dict()['param_groups'][0]['lr']
        #     print('Validation Epoch: {} / {}, lr: {:.8f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'.format(
        #         epoch, args.num_epochs, lr, loss, total_correct, total_seen, acc))

        network.train()
        loss, total_correct, total_seen, acc = train_one_epoch(train_dl, network, criterion, optimizer, device)
        train_history = utils.append_train_history(train_history, acc, loss)
        # if epoch % n_show == 0:
        #     lr = optimizer.state_dict()['param_groups'][0]['lr']
        #     print('Train Epoch: {} / {}, lr: {:.8f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'.format(
        #         epoch, args.num_epochs, lr, loss, total_correct, total_seen, acc))
        scheduler.step()
    test_acc = utils.evaluate_binary2_output(network, criterion, test_dl)
    list_dict = utils.evaluate_ROC(network, criterion, test_dl)
    train_history['Test accuracy'] = test_acc
    train_history['ROC_data_TPR'] = list_dict['TPR']
    train_history['ROC_data_FPR'] = list_dict['FPR']
    # This stores all information related to the confusion matrix at different values of k
    train_history['Confusion_info'] = list_dict
    args.fold_count += 1
    save_model_at_fold(network, args)
    return train_history


def train_AUC_tune(args, train_dl, val_dl, test_dl, network_spec='3D'):
    """
    This train function uses a non-fixed number of epochs.
    """
    # torch.manual_seed(0)
    train_history = TrainHistory()
    # Get device information
    device = utils.get_default_device()

    # Build network model and optimizer
    network = PointNetClsGender(dp_r=args.dpr)
    if network_spec == '4D':
        network = PointNetClsDis(dp_r=args.dpr)
    network = network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.base_lr, betas=(0.9, 0.999))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.75)
    criterion = nn.BCELoss()

    # Training
    for epoch in range(1, args.num_epochs + 1):
        b = "\033[33mTraining epoch {}/{} in progress\033[0m".format(epoch, args.num_epochs)
        sys.stdout.write('\r' + b)

        network.eval()
        loss, total_correct, total_seen, acc = test_one_epoch(val_dl, network, criterion, device)
        train_history.record_val(acc, loss)

        network.train()
        loss, total_correct, total_seen, acc = train_one_epoch(train_dl, network, criterion, optimizer, device)
        train_history.record_train(acc, loss)

        val_AUC = find_AUC_in_fold(utils.evaluate_ROC(network, criterion, val_dl))
        train_history(val_AUC, 'Validation AUC')

    test_acc = utils.evaluate_binary2_output(network, criterion, test_dl)
    list_dict = utils.evaluate_ROC(network, criterion, test_dl)
    train_history.add_single_item(test_acc, 'Test accuracy')
    train_history.add_single_item(list_dict['TPR'], 'ROC_data_TPR')
    train_history.add_single_item(list_dict['FPR'], 'ROC_data_FPR')
    train_history.add_single_item(list_dict, 'Confusion_info')
    args.fold_count += 1
    save_model_at_fold(network, args)
    return train_history.history_dict


def evaluate_AUC_from_models(args, train_dl, val_dl, test_dl, network_spec='3D'):
    # torch.manual_seed(0)
    train_history = utils.init_train_history()
    # Get device information
    device = utils.get_default_device()

    # Build network model and optimizer
    network = PointNetClsGender(dp_r=args.dpr)
    if network_spec == '4D':
        network = PointNetClsDis(dp_r=args.dpr)
    network = network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=args.base_lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.75)
    criterion = nn.BCELoss()

    # Training
    for epoch in range(1, args.num_epochs + 1):
        b = "\033[33mTraining epoch {}/{} in progress\033[0m".format(epoch, args.num_epochs)
        sys.stdout.write('\r' + b)
        n_show = args.num_epochs
        network.eval()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loss, total_correct, total_seen, acc = test_one_epoch(val_dl, network, criterion, device)
        train_history = utils.append_train_history(train_history, acc, loss, is_val=True)
        # if epoch % n_show == 0:
        #     lr = optimizer.state_dict()['param_groups'][0]['lr']
        #     print('Validation Epoch: {} / {}, lr: {:.8f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'.format(
        #         epoch, args.num_epochs, lr, loss, total_correct, total_seen, acc))

        network.train()
        loss, total_correct, total_seen, acc = train_one_epoch(train_dl, network, criterion, optimizer, device)
        train_history = utils.append_train_history(train_history, acc, loss)
        # if epoch % n_show == 0:
        #     lr = optimizer.state_dict()['param_groups'][0]['lr']
        #     print('Train Epoch: {} / {}, lr: {:.8f}, Loss: {:.2f}, Corr: {}, Total: {}, Acc: {:.4f}'.format(
        #         epoch, args.num_epochs, lr, loss, total_correct, total_seen, acc))
        scheduler.step()
    FPR_list, TPR_list = utils.evaluate_ROC(network, criterion, test_dl)
    train_history['ROC_data_TPR'] = TPR_list
    train_history['ROC_data_FPR'] = FPR_list
    return train_history



