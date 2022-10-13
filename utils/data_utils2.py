"""
New.datatools
"""


import pandas as pd
import glob
import numpy as np
import torch.utils.data as data
import torch
from utils.data_utils import reduce_pc_size
from matplotlib import pyplot as plt
import os
from colorama import Fore, Back, Style
import argparse


def read_metadata(args):
    csv_dir = args.csv_dir
    data_name = args.data_name
    gender = args.gender_spec
    take_extreme = extreme_param(args)
    df = pd.read_csv(csv_dir)
    df = df.drop_duplicates(subset=['case-id'])
    df = df.reset_index(drop=True)

    if gender == 'all':
        df_out = df[['case-id', data_name]]
    else:
        df = df[['case-id', data_name, 'sex']]
        df_out = pick_gender(df, gender)

    if take_extreme and data_name == 'age':
        df_out = take_extreme_age(df_out)

    return df_out


def read_metadata1(args):
    """
    This function reads the metadata file for gender classification
    """
    csv_dir = args.csv_dir
    data_name = args.data_name
    df = pd.read_csv(csv_dir)
    df = df.drop_duplicates(subset=['case-id'])
    df = df.reset_index(drop=True)
    df_out = df[['case-id', data_name]]

    return df_out


def read_metadata2(args):
    """
    This function reads the metadata file for other classification.
    Gender can be specified to only take male or female information.
    gender_spec can either be 'male' or 'female'
    """
    csv_dir = args.csv_dir
    data_name = args.data_name
    try:
        gender = args.gender_spec
    except AttributeError:
        gender = 'all'
    df = pd.read_csv(csv_dir)
    df = df.drop_duplicates(subset=['case-id'])
    df = df.reset_index(drop=True)
    df = df[['case-id', data_name, 'sex']]
    if not gender == 'all':
        df = pick_gender(df, gender)
    return df


def read_metadata_disease(csv_dir, case_id_key):
    """
    This function reads the csv file and remove any rows with duplicated id.
    Used for disease classification
    """
    df = pd.read_csv(csv_dir)
    df = df.drop_duplicates(subset=[case_id_key])
    df = df.reset_index(drop=True)
    df = df[[case_id_key, 'sex']]
    df.rename(columns={case_id_key: 'case-id'}, inplace=True)
    return df


def read_metadata_disease2(csv_dir, case_id_key, gender='male'):
    """
    This function reads the csv file and remove any rows with duplicated id.
    Used for disease classification.
    Gender can be specified.
    """
    df = pd.read_csv(csv_dir)
    df = df.drop_duplicates(subset=[case_id_key])
    df = df.reset_index(drop=True)
    df = df[[case_id_key, 'sex']]
    df.rename(columns={case_id_key: 'case-id'}, inplace=True)
    df = pick_gender(df, gender)
    return df


def read_metadata3(args):
    """
    This function reads the metadata file for extreme age classification.
    Gender can also be specified
    """
    df = read_metadata2(args)
    df = take_extreme_age(df)

    return df


def take_extreme_age(df_age):
    # This function takes in a dataframe with age as the label and only keeps cases with extreme ages
    age = df_age['age']
    max_age = age.max()
    min_age = age.min()
    step = max_age - min_age
    lower = min_age + step/4
    higher = max_age - step/4
    new_list = []
    for index, row in df_age.iterrows():
        if row['age'] <= lower or row['age'] >= higher:
            new_list.append(index)
    df_age = df_age.iloc[new_list]
    df_age = df_age.reset_index(drop=True)
    return df_age


def extreme_param(args):
    try:
        take_extreme = args.take_extreme
    except:
        take_extreme = False
    return take_extreme


def pick_gender(df, gender):
    gender_num = 0
    if gender == 'female':
        gender_num = 1
    gender_list = []
    for index, row in df.iterrows():
        if row['sex'] == gender_num:
            gender_list.append(index)
    df = df.iloc[gender_list]
    df = df.reset_index(drop=True)
    df_out = df
    return df_out


def divide_age(df, args):
    """
    This function divide age into a number of classes
    :param df:
    :param args:
    :return:
    """
    try:
        n_class = args.num_classes
    except AttributeError:
        n_class = 2
    df_out = df
    df_v = df.values
    age_list = df_v[:, 1]
    age_max = max(age_list)
    age_min = min(age_list)
    age_class = divide_range(age_min, age_max, n_class)
    for index, quantile in enumerate(age_class):
        if not index == len(age_class) - 1:
            lower = quantile
            higher = age_class[index+1]
            for i, age in enumerate(age_list):
                if age >= lower and age < higher:
                    df_out.loc[i, 'age'] = index
        else:
            lower = quantile
            higher = age_max
            for i, age in enumerate(age_list):
                if age >= lower and age <= higher:
                    df_out.loc[i, 'age'] = index
    return df_out, age_class


def plot_age_distribution(args):
    df = read_metadata(args)
    n = len(df)
    df = df.values
    age = df[:, 1]
    age_max = max(age)
    age_min = min(age)
    age_list = [i for i in range(age_min, age_max + 1)]
    prob_list = []
    for each_age in age_list:
        age_to_check = each_age
        age_count = 0
        for i in range(n):
            if age[i] == age_to_check:
                age_count += 1
        prob = age_count / n
        prob_list.append(prob)
    mean = sum([x*y for x, y in zip(age_list, prob_list)])
    print(sum(prob_list))
    plot = plt.figure()
    plt.bar(age_list, prob_list)
    plt.axvline(x=mean, color='r', linestyle='-', label='Mean value')
    plt.legend()
    plt.xlabel('Age')
    plt.ylabel('Probability')
    plt.title('Probability distribution of ' + args.gender_spec + ' age' + ' N = ' + str(n))
    return plot


def get_vectors(args):
    # If names of the npy files change, indexing of id needs to be changed.
    # Currently the function only works with a format like： latent_space_1002549.npy
    reduce_size = args.reduce_size
    file_dir = args.data_dir
    filelist = glob.glob(file_dir + '/*.npy')

    vector_dict = make_dict(filelist, reduce_size)

    return vector_dict


def make_dict(filelist, reduce_size):
    """This function takes in a file list and reads off all files in the list. It returns a dictionary where
    the keys are the corresponding case ids of files"""
    vector_dict = {}
    for fname in filelist:
        case_id = fname[-11:-4]
        lat_data = np.load(fname)
        data_in = lat_data[:, 0:3]
        data_in = reduce_pc_size(data_in, reduce_size)
        vector_dict[case_id] = data_in
    return vector_dict


def remove_id(ids, labels, vector_dict):
    """
    This function gets rid of cases in 'ids' and 'labels'
    that do not co-exist in the metadata file and the data directory
    :param ids:
    :param labels:
    :param vector_dict:
    :return:
    """
    ids_new = []
    labels_new = []
    for i, vec_id in enumerate(ids):
        try:
            vec_id_str = str(vec_id)
            vec_here = vector_dict[vec_id_str]  # This is important since it checks if a id is available in the dataset
            ids_new.append(vec_id)
            labels_new.append(labels[i])
        except KeyError:
            continue
    return ids_new, labels_new


def make_sorted_data(vector_dict, case_id):
    """
    This function returns a data matrix that contains data by the order in 'case_id'
    :param vector_dict:
    :param case_id:
    :return:
    """
    n_id = len(case_id)
    sample = vector_dict[str(case_id[0])]
    sample_dim = sample.shape
    matrix_out = np.zeros((n_id, sample_dim[0], sample_dim[1]))
    for i in range(n_id):
        id_key = str(case_id[i])
        vec_here = vector_dict[id_key]
        matrix_out[i] = vec_here

    return matrix_out


class latent_ds(data.Dataset):
    # This dataset is capable of normalising labels
    def __init__(self, args):
        self.reduce_size = args.reduce_size
        self.multi_class = args.multi_class
        df = read_metadata(args)
        if args.age_class:
            df, self.age_class = divide_age(df, args)
        df_v = df.values
        self.vector_dic = get_vectors(args)
        labels = df_v[:, 1]
        ids = df_v[:, 0]
        self.ids, self.labels = remove_id(ids, labels, self.vector_dic)
        self.data_matrix = make_sorted_data(self.vector_dic, self.ids)
        print(Fore.GREEN + 'Dataset for gender classification has been initialised.' + Style.RESET_ALL)
        # label_max = max(self.labels)
        # self.label_min = min(self.labels)
        # self.diff = label_max - self.label_min

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        x = x.permute(1, 0)  # Not required for PointNet2
        if self.multi_class:
            y = torch.tensor(y, dtype=torch.long)
        else:
            y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_class(self):
        return self.age_class

    def get_ids(self):
        return self.ids

    def get_all(self):
        return self.vector_dic, self.data_matrix, self.labels


def get_svm_data(dl):
    # data_name can be 'x_train', 'y_train', 'x_val' or 'y_val' or 'all'
    data_dict = {}
    for x, y in dl:
        data_dict['x'] = x
        data_dict['y'] = y

    for key in data_dict:
        data_dict[key] = data_dict[key].numpy()

    return data_dict['x'], data_dict['y']


def divide_range(start, end, n):
    interval = end - start
    step = int(interval/n)
    is_mul = interval % step == 0
    a = np.arange(start, end, step)
    a = list(a)
    if not is_mul:
        a.pop(-1)
    return a


def use_root_dir(root_dir):
    current_directory = os.path.dirname(__file__)
    parent_directory = os.path.split(current_directory)[0]  # Repeat as needed
    abs_dir = parent_directory + '/' + root_dir
    return abs_dir


def get_min_and_diff(labels):
    """
    This function returns the minimum value and range of the labels
    :param labels:
    :return:
    """
    label_max = max(labels)
    label_min = min(labels)
    diff = label_max - label_min
    return label_min, diff