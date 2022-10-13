"""
New.my_dataset
"""


import torch.utils.data as data
import torch
from colorama import Fore, Back, Style
from utils.data_utils2 import get_vectors, remove_id, make_sorted_data
import utils.data_utils2 as datatools
import utils.dataset_utils as utils2
import argparse
import numpy as np


class Dataset1(data.Dataset):
    """
    This dataset is for the use of PointNet.
    The supported classification type is binary
    """

    def __init__(self, args):
        self.reduce_size = args.reduce_size
        df = datatools.read_metadata1(args)
        df_v = df.values
        vector_dic = get_vectors(args)
        labels = df_v[:, 1]
        ids = df_v[:, 0]
        self.ids, self.labels = remove_id(ids, labels, vector_dic)
        self.data_matrix = make_sorted_data(vector_dic, self.ids)
        print(Fore.GREEN + 'Dataset for PointNet binary classification has been initialised.' + Style.RESET_ALL)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        x = x.permute(1, 0)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_ids(self):
        return self.ids


class Dataset11(data.Dataset):
    """
    This dataset is for the use of PointNet.
    It is for age regression.
    Age is normalised.
    """
    def __init__(self, args):
        self.reduce_size = args.reduce_size
        df = datatools.read_metadata2(args)
        df_v = df.values
        self.vector_dic = get_vectors(args)
        labels = df_v[:, 1]
        ids = df_v[:, 0]
        self.ids, self.labels = remove_id(ids, labels, self.vector_dic)
        self.data_matrix = make_sorted_data(self.vector_dic, self.ids)
        self.label_min, self.diff = datatools.get_min_and_diff(labels)
        print(Fore.GREEN + 'Dataset for PointNet normalised age regression has been initialised.' + Style.RESET_ALL)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        y = (y - self.label_min)/self.diff
        x = torch.FloatTensor(x)
        x = x.permute(1, 0)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_ids(self):
        return self.ids

    def get_all(self):
        return self.vector_dic, self.data_matrix, self.labels


class Dataset12(data.Dataset):
    """
    This dataset is for the use of PointNet.
    It is for multi-class age classification
    """
    def __init__(self, args):
        self.reduce_size = args.reduce_size
        df = datatools.read_metadata2(args)
        df, self.age_class = datatools.divide_age(df, args)
        df_v = df.values
        self.vector_dic = get_vectors(args)
        labels = df_v[:, 1]
        ids = df_v[:, 0]
        self.ids, self.labels = remove_id(ids, labels, self.vector_dic)
        self.data_matrix = make_sorted_data(self.vector_dic, self.ids)
        print(Fore.GREEN + 'Dataset for PointNet multi-class age classification has been initialised.' + Style.RESET_ALL)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        x = x.permute(1, 0)
        y = torch.tensor(y, dtype=torch.long)
        y = y.view(-1)
        return x, y

    def get_ids(self):
        return self.ids

    def get_all(self):
        return self.vector_dic, self.data_matrix, self.labels


class Dataset13(data.Dataset):
    """
    This dataset is for the use of PointNet.
    The is for extreme age binary classification
    """

    def __init__(self, args):
        self.reduce_size = args.reduce_size
        df = datatools.read_metadata3(args)
        df, self.age_class = datatools.divide_age(df, args)
        df_v = df.values
        self.vector_dic = get_vectors(args)
        labels = df_v[:, 1]
        ids = df_v[:, 0]
        self.ids, self.labels = remove_id(ids, labels, self.vector_dic)
        self.data_matrix = make_sorted_data(self.vector_dic, self.ids)
        print(Fore.GREEN + 'Dataset for PointNet binary extreme age classification has been initialised.' + Style.RESET_ALL)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        x = x.permute(1, 0)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_ids(self):
        return self.ids

    def get_all(self):
        return self.vector_dic, self.data_matrix, self.labels


class Dataset14(data.Dataset):
    """
    Binary gender classification for PointNet
    Overfitting test
    """

    def __init__(self, args):
        self.reduce_size = args.reduce_size
        df = datatools.read_metadata1(args)
        df_v = df.values
        self.vector_dic = get_vectors(args)
        labels = df_v[:, 1]
        ids = df_v[:, 0]
        self.ids, self.labels = remove_id(ids, labels, self.vector_dic)
        self.data_matrix = make_sorted_data(self.vector_dic, self.ids)
        self.data_matrix = self.data_matrix[0:4]
        self.labels = self.labels[0:4]
        print(Fore.GREEN + 'Dataset for PointNet binary overfitting test has been initialised.' + Style.RESET_ALL)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        x = x.permute(1, 0)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_ids(self):
        return self.ids

    def get_all(self):
        return self.vector_dic, self.data_matrix, self.labels


class Dataset15(data.Dataset):
    """
    This dataset is for binary disease classification. Gender is included as conditional input.
    """

    def __init__(self, args):
        self.df_disease = datatools.read_metadata_disease(args.disease_csv, 'eid')
        self.df_healthy = datatools.read_metadata_disease(args.data_csv, 'case-id')
        self.data_matrix, self.labels, self.ids = utils2.get_data_and_labels(args)
        print(Fore.GREEN + 'Dataset for PointNet binary disease classification has been initialised.' + Style.RESET_ALL)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        if y == 0:
            df = self.df_healthy
        else:
            df = self.df_disease
        id_here = self.get_id(index)
        condition = utils2.read_condition(df, id_here)
        x = utils2.concatenate_condition(x, condition)
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_id(self, index):
        case_id = self.ids[index]
        return int(case_id)


class Dataset15Conc(data.Dataset):
    """
    This dataset is for binary disease classification
    """

    def __init__(self, args):
        self.df_disease = datatools.read_metadata_disease(args.disease_csv, 'eid')
        self.df_healthy = datatools.read_metadata_disease(args.data_csv, 'case-id')
        self.data_matrix, self.labels, self.ids = utils2.get_data_and_labels3(args)
        print(Fore.GREEN + 'Dataset for PointNet binary disease classification has been initialised.' + Style.RESET_ALL)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        if y == 0:
            df = self.df_healthy
        else:
            df = self.df_disease
        id_here = self.get_id(index)
        condition = utils2.read_condition(df, id_here)
        x = utils2.concatenate_condition(x, condition)
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_id(self, index):
        case_id = self.ids[index]
        return int(case_id)


class Dataset16(data.Dataset):
    """
    This dataset is for binary disease classification, but gender is not included as the 4th dimension of the point
    clouds. Instead, the training cases are separated for male and female.
    """

    def __init__(self, args):
        data_matrix, labels, ids = utils2.get_data_and_labels1(args)
        self.data_matrix, self.labels, self.ids = utils2.pick_one_gender_balance(data_matrix, labels, ids, args)
        count0, count1 = utils2.count_0_and_1(self.labels)
        print('0: ', count0)
        print('1: ', count1)
        print(Fore.GREEN + 'Dataset for PointNet binary disease classification has been initialised (gender specified).'
              + Style.RESET_ALL)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_id(self, index):
        case_id = self.ids[index]
        return int(case_id)


class Dataset17(data.Dataset):
    """
    This dataset is for binary disease classification, but gender is not included as the 4th dimension of the point
    clouds. Instead, the training cases are separated for male and female.
    """

    def __init__(self, args):
        data_matrix, labels, ids = utils2.get_data_and_labels2(args)
        self.data_matrix, self.labels, self.ids = utils2.pick_one_gender_balance(data_matrix, labels, ids, args)
        count0, count1 = utils2.count_0_and_1(self.labels)
        print('0: ', count0)
        print('1: ', count1)
        print(Fore.GREEN + 'Dataset for PointNet binary disease classification has been initialised (gender specified).'
              + Style.RESET_ALL)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_id(self, index):
        case_id = self.ids[index]
        return int(case_id)


class Dataset18(data.Dataset):
    """
    This dataset is for binary disease classification, but gender is not included as the 4th dimension of the point
    clouds. Gender is not specified and class balance is not guaranteed.
    """

    def __init__(self, args):
        self.data_matrix, self.labels, self.ids = utils2.get_data_and_labels1(args)
        count0, count1 = utils2.count_0_and_1(self.labels)
        print('0: ', count0)
        print('1: ', count1)
        print(Fore.GREEN + 'Dataset for PointNet binary disease classification has been initialised (gender not '
                           'specified).' + Style.RESET_ALL)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_id(self, index):
        case_id = self.ids[index]
        return int(case_id)


class Dataset19(data.Dataset):
    """
    This dataset is for binary disease classification, but gender is not included as the 4th dimension of the point
    clouds. Gender is not specified and class balance is not guaranteed. This dataset includes the additional data from
    Marcel.
    """

    def __init__(self, args):
        self.data_matrix, self.labels, self.ids = utils2.get_data_and_labels4(args)
        utils2.count_0_and_1_print(self.labels)
        print(Fore.GREEN + 'Dataset for PointNet binary disease classification has been initialised (gender not '
                           'specified).' + Style.RESET_ALL)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_id(self, index):
        case_id = self.ids[index]
        return int(case_id)


class Dataset1901(data.Dataset):
    """
    This dataset is for binary disease classification, but gender is not included as the 4th dimension of the point
    clouds. Gender is not specified and class balance is not guaranteed. This dataset includes the additional data from
    Marcel.
    """

    def __init__(self, args):
        self.data_matrix, self.labels, self.ids = utils2.get_data_and_labels401(args)
        self.n0, self.n1 = utils2.count_0_and_1_print(self.labels)
        print(Fore.GREEN + 'Dataset for PointNet binary disease classification has been initialised (gender not '
                           'specified).' + Style.RESET_ALL)
        print(Fore.BLUE + 'Total number of data in the dataset: {}'.format(len(self.labels)) + Style.RESET_ALL)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_id(self, index):
        case_id = self.ids[index]
        return int(case_id)


class Dataset191(data.Dataset):
    """
    This dataset is for binary disease classification, but gender is not included as the 4th dimension of the point
    clouds. Gender is not specified and class balance is not guaranteed. This dataset includes the additional data from
    Marcel.
    This dataset is for binary disease classification. Gender is included as conditional input.
    Some of the cases are missing in the metadata file and they are removed from the total dataset.
    """

    def __init__(self, args):
        self.df = datatools.read_metadata_disease(args.csv_file, 'eid')
        self.data_matrix, self.labels, self.ids = utils2.get_data_and_labels5(args)
        utils2.count_0_and_1_print(self.labels)
        print(Fore.GREEN + 'Dataset for PointNet binary disease classification has been initialised.' + Style.RESET_ALL)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        id_here = self.get_id(index)
        condition = utils2.read_condition(self.df, id_here)
        x = utils2.concatenate_condition(x, condition)
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_id(self, index):
        case_id = self.ids[index]
        return int(case_id)


class Dataset192(data.Dataset):
    """
    This is a different version of Dataset19. It uses concatenated data.
    """

    def __init__(self, args, if_print=True):
        self.data_matrix, self.labels, self.ids = utils2.get_data_and_labels41(args)
        if if_print:
            utils2.count_0_and_1_print(self.labels)
            print(Fore.GREEN + 'Dataset for PointNet binary disease classification has been initialised (gender not '
                               'specified).' + Style.RESET_ALL)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_id(self, index):
        case_id = self.ids[index]
        return int(case_id)


class Dataset193(data.Dataset):
    """
    This is a different version of Dataset19. It uses concatenated data.
    And it forces class balance by wasting excessive data from the major class (healthy patients in this case).
    """

    def __init__(self, args, if_print=True):
        self.data_matrix, self.labels, self.ids = utils2.get_data_and_labels42(args)
        if if_print:
            utils2.count_0_and_1_print(self.labels)
            print(Fore.GREEN + 'Dataset for PointNet binary disease classification has been initialised (gender not '
                               'specified).' + Style.RESET_ALL)
            print(Fore.BLUE + 'Total number of data in the dataset: {}'.format(len(self.labels)) + Style.RESET_ALL)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_id(self, index):
        case_id = self.ids[index]
        return int(case_id)


class Dataset1931(data.Dataset):
    """
    This is a different version of Dataset193. It does not use concatenated data.
    And it forces class balance by wasting excessive data from the major class (healthy patients in this case).
    """

    def __init__(self, args, if_print=True):
        self.data_matrix, self.labels, self.ids = utils2.get_data_and_labels401(args)
        if if_print:
            utils2.count_0_and_1_print(self.labels)
            print(Fore.GREEN + 'Dataset for PointNet binary disease classification has been initialised (gender not '
                               'specified).' + Style.RESET_ALL)
            print(Fore.BLUE + 'Total number of data in the dataset: {}'.format(len(self.labels)) + Style.RESET_ALL)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_id(self, index):
        case_id = self.ids[index]
        return int(case_id)


class Dataset194(data.Dataset):
    """
    This is a different version of Dataset19. It uses concatenated data.
    It helps to ensure class balance while not wasting any data.
    The extra class 0 data are moved to the end of the data matrix.
    """

    def __init__(self, args):
        self.data_matrix, self.labels, self.ids = utils2.get_data_and_labels411(args)
        self.n0, self.n1 = utils2.count_0_and_1_print(self.labels)
        print(Fore.GREEN + 'Dataset for PointNet binary disease classification has been initialised (gender not '
                           'specified).' + Style.RESET_ALL)
        print(Fore.BLUE + 'Total number of data in the dataset: {}'.format(len(self.labels)) + Style.RESET_ALL)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_class_n(self, class_spec=0):
        if class_spec == 0:
            return self.n0
        elif class_spec == 1:
            return self.n1

    def get_case_id(self, index):
        return self.ids[index]


class Dataset194GetId(Dataset194):

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        id_here = self.get_case_id(index)
        return x, y, id_here

    def get_data_by_id(self, case_id):
        index, temp = np.where(self.ids == case_id)
        return self.__getitem__(index)


class Dataset1941(data.Dataset):
    """
    This is a different version of Dataset194. It does not uses concatenated data.
    It helps to ensure class balance while not wasting any data.
    The extra class 0 data are moved to the end of the data matrix.
    """

    def __init__(self, args):
        self.data_matrix, self.labels, self.ids = utils2.get_data_and_labels412(args)
        self.n0, self.n1 = utils2.count_0_and_1_print(self.labels)
        print(Fore.GREEN + 'Dataset for PointNet binary disease classification has been initialised (gender not '
                           'specified).' + Style.RESET_ALL)
        print(Fore.BLUE + 'Total number of data in the dataset: {}'.format(len(self.labels)) + Style.RESET_ALL)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_class_n(self, class_spec=0):
        if class_spec == 0:
            return self.n0
        elif class_spec == 1:
            return self.n1

    def get_case_id(self, index):
        return self.ids[index]


class Dataset1941GetId(Dataset1941):

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        id_here = self.get_case_id(index)
        return x, y, id_here

    def get_data_by_id(self, case_id):
        index, temp = np.where(self.ids == case_id)
        return self.__getitem__(index)


class DatasetTest(data.Dataset):
    """
    This is a different version of Dataset19. It uses concatenated data.
    And it forces class balance by wasting excessive data from the major class (healthy patients in this case).
    """

    def __init__(self):
        self.X = np.ones([10, 1])
        self.labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.labels[index]
        y = torch.tensor(y, dtype=torch.float)
        return x, y


class Dataset21(data.Dataset):
    """
    Binary age classification for PointNet2
    """

    def __init__(self, args):
        self.reduce_size = args.reduce_size
        df = datatools.read_metadata1(args)
        df_v = df.values
        self.vector_dic = get_vectors(args)
        labels = df_v[:, 1]
        ids = df_v[:, 0]
        self.ids, self.labels = remove_id(ids, labels, self.vector_dic)
        self.data_matrix = make_sorted_data(self.vector_dic, self.ids)
        print(Fore.GREEN + 'Dataset for PointNet2 gender binary classification has been initialised.' + Style.RESET_ALL)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_ids(self):
        return self.ids

    def get_all(self):
        return self.vector_dic, self.data_matrix, self.labels


class Dataset22(data.Dataset):
    """
    Extreme age binary classification with PointNet2
    """

    def __init__(self, args):
        self.reduce_size = args.reduce_size
        df = datatools.read_metadata3(args)
        df, self.age_class = datatools.divide_age(df, args)
        df_v = df.values
        self.vector_dic = get_vectors(args)
        labels = df_v[:, 1]
        ids = df_v[:, 0]
        self.ids, self.labels = remove_id(ids, labels, self.vector_dic)
        self.data_matrix = make_sorted_data(self.vector_dic, self.ids)
        print(Fore.GREEN + 'Dataset for PointNet2 extreme age binary classification has been initialised.'
              + Style.RESET_ALL)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = self.data_matrix[index]
        y = self.labels[index]
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float)
        y = y.view(-1)
        return x, y

    def get_ids(self):
        return self.ids

    def get_all(self):
        return self.vector_dic, self.data_matrix, self.labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--disease_csv', type=str, default='data/csv/Marcel_myoinfarction.csv')
    parser.add_argument('--data_csv', type=str, default='data/ukbb_samples_metadata.csv')
    parser.add_argument('--disease_dir', type=str, default='data/myo_infarction_data/ES')
    parser.add_argument('--data_dir', type=str, default='data/raw_encoder_input_data/ES')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--base_lr', type=float, default=1e-6)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--reduce_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--gender_spec', type=str, default='female')
    parser.add_argument('--train_prop', type=float, default=0.8)
    parser.add_argument('--val_prop', type=float, default=0.7)
    # args = parser.parse_args()
    # args.disease_csv = use_root_dir(args.disease_csv)
    # args.data_csv = use_root_dir(args.data_csv)
    # args.disease_dir = use_root_dir(args.disease_dir)
    # args.data_dir = use_root_dir(args.data_dir)
    # ds = Dataset16(args)

