"""
New.secondary utils
Data_Analysis.Visualisation is added.
"""

import random
import argparse
import numpy as np
import glob
import utils.utils1 as utils
import os
import pandas as pd
import utils.data_utils2 as datatools
from colorama import Fore, Back, Style
import argparse
from utils.data_utils2 import use_root_dir


# import open3d as o3d


def load_single_point_cloud(healthy_status, stage_spec, phase_spec, ventricle_spec, case_id=12345,
                            reduce_size=3, is_random=False):
    """
    This function returns a loaded 3D point cloud based on the case id provided.
    Or it randomly chooses a point cloud.
    :return:
    """
    dir_wanted = utils.get_ventricle_dir(healthy_status, stage_spec, phase_spec, ventricle_spec)
    file_list = glob.glob(dir_wanted + '/*.npy')
    n = random.randint(0, len(file_list))
    if is_random:
        for i, file_name in enumerate(file_list):
            if i == n:
                id_here = read_id(file_name)
                print('Case id: ', id_here)
                return utils.reduce_pc_size(np.load(file_name), reduce_size)
    for file_name in file_list:
        id_here = read_id(file_name)
        if id_here == case_id:
            return utils.reduce_pc_size(np.load(file_name), reduce_size)


def readXYZ(filename, delimiter=','):
    xs = []
    ys = []
    zs = []
    f = open(filename, 'r')
    line = f.readline()
    N = 0
    while line:
        x, y, z = line.split(delimiter)[0:3]
        x, y, z = float(x), float(y), float(z)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        line = f.readline()
        N += 1
    f.close()
    xs = np.array(xs).reshape(N, 1)
    ys = np.array(ys).reshape(N, 1)
    zs = np.array(zs).reshape(N, 1)
    points = np.concatenate((xs, ys, zs), axis=1)
    return points


# def visualize_PC(point_cloud):
#     """
#     This function visualize the input point cloud.
#     :param point_cloud: shape in the form of tensors.
#     :return:
#     """
#     sample_heart = point_cloud.permute(1, 0)
#     sample_heart = utils.to_numpy(sample_heart)
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(sample_heart)
#     o3d.visualization.draw_geometries([pcd])


# def visualize_ds(ds):
#     sample = ds[0]
#     sample_PC = sample[0]
#     visualize_PC(sample_PC)


def get_people_with_disease(n_disease, disease_metadata, include=True):
    """
    This function returns cases with a certain number of diseases.
    n_disease: number of diseases
    disease_metadata: pandas dataframe of disease metadata file
    """
    out_list = []  # List with all the cases
    index_list = []  # List with all indices of the cases
    for i, row in disease_metadata.iterrows():
        case_id = row[0]
        list_for_count = pd.isna(row)
        disease_count = -1
        for is_na in list_for_count:
            if not is_na:
                disease_count += 1
            else:
                break
        if include:
            if disease_count <= n_disease:
                out_list.append(str(int(case_id)))
                index_list.append(i)
        else:
            if disease_count == n_disease:
                out_list.append(str(int(case_id)))
                index_list.append(i)

    return out_list


def count_0_and_1(labels):
    count0 = 0
    count1 = 0
    for label in labels:
        if int(label) == 1:
            count1 += 1
        elif int(label) == 0:
            count0 += 1
        else:
            print('Error: label not equal to 0 or 1')
    return count0, count1


def count_0_and_1_print(labels):
    count0 = 0
    count1 = 0
    for label in labels:
        if int(label) == 1:
            count1 += 1
        elif int(label) == 0:
            count0 += 1
        else:
            print('Error: label not equal to 0 or 1')
    print('Number in class 0: ', count0)
    print('Number in class 1: ', count1)
    return count0, count1


def pick_one_gender(matrix, labels, ids, args):
    df_disease = datatools.read_metadata_disease2(args.disease_csv, 'eid', gender=args.gender_spec)
    df_healthy = datatools.read_metadata_disease2(args.data_csv, 'case-id', gender=args.gender_spec)
    wanted_list = []
    for i, label in enumerate(labels):
        if label == 0:
            df = df_healthy
        else:
            df = df_disease
        gender_id_list = df['case-id'].values
        if ids[i] in gender_id_list:
            wanted_list.append(i)
    return matrix[wanted_list], labels[wanted_list], ids[wanted_list]


def pick_one_gender_balance(matrix, labels, ids, args):
    """
    This function takes in the data matrix and labels. It picks out cases of one specified gender from both the healthy
    and diseased cases. It forced class balance between healthy and diseased cases by picking the smaller number.
    For example, if we have 80 healthy male cases and 70 diseased male cases, the function returns 70 male healthy and
    diseased cases respectively.
    """

    # Pick out cases with a single specified gender
    df_disease = datatools.read_metadata_disease2(args.disease_csv, 'eid', gender=args.gender_spec)
    df_healthy = datatools.read_metadata_disease2(args.data_csv, 'case-id', gender=args.gender_spec)
    wanted_list = []
    for i, label in enumerate(labels):
        # Pick the right metadata for healthy or diseased cases
        if label == 0:
            df = df_healthy
        else:
            df = df_disease
        gender_id_list = df['case-id'].values
        if ids[i] in gender_id_list:
            wanted_list.append(i)
    count0, count1 = count_0_and_1(labels[wanted_list])
    if count0 > count1:
        del wanted_list[count1:count0]
    elif count1 > count0:
        del wanted_list[2 * count0:(count1 + count0)]
    return matrix[wanted_list], labels[wanted_list], ids[wanted_list]


def read_condition(df, case_id):
    """
    This function reads the conditionals from the metadata df based on the case id provided.
    :param df:
    :param case_id:
    :return:
    """
    gender_list = df['sex'].values
    temp = df.index[df['case-id'] == case_id].tolist()
    index = temp[0]
    gender = gender_list[index]
    return gender


def concatenate_condition(x, condition):
    """
    This function add one more dimension to the point cloud x to make it contain the conditional information.
    For example, if x is 1000x3, new x will be 1000x4 with all elements in the 4th dimension being the condition.
    This function by default assume x has a shape of 3 x number of points and condition is assumed to be an integer.
    :param x:
    :param condition:
    :return:
    """
    dim = x.shape
    new_x = np.zeros((4, dim[1]))
    new_x[0:3, :] = x
    new_x[3, :] = condition
    return new_x


def get_data_and_labels(args):
    matrix0, matrix1, id_list0, id_list1 = get_vectors(args)
    label0, label1 = label_data(matrix0, matrix1)
    total_data = np.concatenate((matrix0, matrix1), axis=0)
    total_label = np.concatenate((label0, label1), axis=0)
    total_id = np.concatenate((id_list0, id_list1), axis=0)
    return total_data, total_label, total_id


def get_data_and_labels1(args):
    matrix0, matrix1, id_list0, id_list1 = get_vectors_not_same(args)
    label0, label1 = label_data(matrix0, matrix1)
    total_data = np.concatenate((matrix0, matrix1), axis=0)
    total_label = np.concatenate((label0, label1), axis=0)
    total_id = np.concatenate((id_list0, id_list1), axis=0)
    return total_data, total_label, total_id


def get_data_and_labels2(args):
    matrix0, matrix1, id_list0, id_list1 = get_vectors2(args)
    label0, label1 = label_data(matrix0, matrix1)
    total_data = np.concatenate((matrix0, matrix1), axis=0)
    total_label = np.concatenate((label0, label1), axis=0)
    total_id = np.concatenate((id_list0, id_list1), axis=0)
    return total_data, total_label, total_id


def get_data_and_labels3(args):
    matrix0, matrix1, id_list0, id_list1 = get_vectors3(args)
    label0, label1 = label_data(matrix0, matrix1)
    total_data = np.concatenate((matrix0, matrix1), axis=0)
    total_label = np.concatenate((label0, label1), axis=0)
    total_id = np.concatenate((id_list0, id_list1), axis=0)
    return total_data, total_label, total_id


def get_data_and_labels4(args):
    """
    This function reads the data from healthy and diseased directories directly.
    No reading from metadata file is required.
    Positive and negative cases are of different quantity.
    """
    matrix0, matrix1, id_list0, id_list1 = get_vectors_not_same2(args)
    label0, label1 = label_data(matrix0, matrix1)
    total_data = np.concatenate((matrix0, matrix1), axis=0)
    total_label = np.concatenate((label0, label1), axis=0)
    total_id = np.concatenate((id_list0, id_list1), axis=0)
    return total_data, total_label, total_id


def get_data_and_labels401(args):
    """
    Extension of get_data_and_labels4.
    It forces class balance by removing the excessive healthy data.
    """
    matrix0, matrix1, id_list0, id_list1 = get_vectors_not_same2(args)
    matrix0 = np.array(remove_excessive(matrix1, matrix0))
    id_list0 = np.array(remove_excessive(id_list1, id_list0))
    label0, label1 = label_data(matrix0, matrix1)
    total_data = np.concatenate((matrix0, matrix1), axis=0)
    total_label = np.concatenate((label0, label1), axis=0)
    total_id = np.concatenate((id_list0, id_list1), axis=0)
    return total_data, total_label, total_id


def get_data_and_labels41(args):
    """
    Extension of get_data_and_labels4.
    It uses concatenated data.
    """
    matrix0, matrix1, id_list0, id_list1 = get_vectors_not_same21(args)
    label0, label1 = label_data(matrix0, matrix1)
    total_data = np.concatenate((matrix0, matrix1), axis=0)
    total_label = np.concatenate((label0, label1), axis=0)
    total_id = np.concatenate((id_list0, id_list1), axis=0)
    return total_data, total_label, total_id


def get_data_and_labels42(args):
    """
    Extension of get_data_and_labels4.
    It uses concatenated data.
    It removes excessive cases from the major class to ensure class balance.
    Class 1 is assumed to be the minor class.
    """
    matrix0, matrix1, id_list0, id_list1 = get_vectors_not_same21(args)
    matrix0 = np.array(remove_excessive(matrix1, matrix0))
    id_list0 = np.array(remove_excessive(id_list1, id_list0))
    label0, label1 = label_data(matrix0, matrix1)
    total_data = np.concatenate((matrix0, matrix1), axis=0)
    total_label = np.concatenate((label0, label1), axis=0)
    total_id = np.concatenate((id_list0, id_list1), axis=0)
    return total_data, total_label, total_id


def get_data_and_labels411(args):
    """
    Extension of get_data_and_labels41.
    It moves the extra class 0 data to the end of the overall data matrix.
    """
    matrix0, matrix1, id_list0, id_list1 = get_vectors_not_same21(args)
    label0, label1 = label_data(matrix0, matrix1)
    total_data = np.concatenate((matrix0, matrix1), axis=0)
    total_label = np.concatenate((label0, label1), axis=0)
    total_id = np.concatenate((id_list0, id_list1), axis=0)
    n_extra = int(matrix0.shape[0] - matrix1.shape[0])
    return move_to_the_end(total_data, n_extra), move_to_the_end(total_label, n_extra), move_to_the_end(total_id,
                                                                                                        n_extra)


def get_data_and_labels412(args):
    """
    Extension of get_data_and_labels411. It does not use concatenated data.
    It moves the extra class 0 data to the end of the overall data matrix.
    """
    matrix0, matrix1, id_list0, id_list1 = get_vectors_not_same2(args)
    label0, label1 = label_data(matrix0, matrix1)
    total_data = np.concatenate((matrix0, matrix1), axis=0)
    total_label = np.concatenate((label0, label1), axis=0)
    total_id = np.concatenate((id_list0, id_list1), axis=0)
    n_extra = int(matrix0.shape[0] - matrix1.shape[0])
    return move_to_the_end(total_data, n_extra), move_to_the_end(total_label, n_extra), move_to_the_end(total_id,
                                                                                                        n_extra)


def move_to_the_end(data_matrix, n_extra):
    """
    This function moves the extra class 0 data to the end of the entire data matrix.
    So that the training set could be made balanced.
    :param data_matrix:
    :param n_extra:
    :return:
    """
    n_total = data_matrix.shape[0]
    n0 = int((n_total + n_extra) / 2)
    n1 = int((n_total - n_extra) / 2)
    matrix0 = data_matrix[:n1]
    matrix1 = data_matrix[n0:]
    matrix_extra = data_matrix[n1:n0]
    return np.concatenate((matrix0, matrix1, matrix_extra))


def remove_excessive(data_small, data_large):
    """
    This function removes the excessive data from 'data_large' so that it contains the same number
    of data as 'data_small'
    :param data_small: first dimension has to be the number of data
    :param data_large: the larger dataset to be removed
    :return:
    """
    data_large_new = index_list_list(data_large, list(range(data_small.shape[0])))
    return data_large_new


def duplicate_minor_class(matrix0, matrix1, id_list0, id_list1):
    """
    This forces matrix0 and matrix1 to have the same number of cases.
    The id lists are modified accordingly.
    :param matrix0:
    :param matrix1:
    :param id_list0:
    :param id_list1:
    :return:
    """
    num0 = matrix0.shape[0]
    num1 = matrix1.shape[0]
    num_extra = np.abs(num0 - num1)
    if num0 > num1:
        matrix1, id_list1 = make_large_small_equal(matrix1, id_list1, num_extra)
    elif num0 < num1:
        matrix0, id_list0 = make_large_small_equal(matrix0, id_list0, num_extra)
    return matrix0, matrix1, id_list0, id_list1


def make_large_small_equal(matrix_s, id_list_s, num_extra):
    num_s = matrix_s.shape[0]
    if num_extra <= num_s:
        list_extra = generate_list(0, num_s, num_extra, repeat=False)
    else:
        list_extra = generate_list(0, num_s, num_extra, repeat=True)
    matrix_extra = index_list_list(matrix_s, list_extra)
    id_extra = index_list_list(id_list_s, list_extra)
    matrix_s = np.concatenate((matrix_s, matrix_extra), axis=0)
    id_list_s = np.concatenate((id_list_s, id_extra), axis=0)
    return matrix_s, id_list_s


def generate_list(start, end, num, repeat=True):
    temp_list = range(start, end)
    if repeat:
        return random.choices(temp_list, k=num)
    else:
        return random.sample(temp_list, num)


def get_data_and_labels5(args):
    """
    This function is an extended version of get_data_and_labels4().
    But it removes cases whose information is not available in the metadata file.
    """
    df = datatools.read_metadata_disease(args.csv_file, 'eid')
    matrix0, matrix1, id_list0, id_list1 = get_vectors_not_same2(args)
    total_id = np.concatenate((id_list0, id_list1), axis=0)
    total_id, final_list = check_missing_meta(df, total_id)
    label0, label1 = label_data(matrix0, matrix1)
    total_data = np.concatenate((matrix0, matrix1), axis=0)
    total_label = np.concatenate((label0, label1), axis=0)
    total_data = index_list_list(total_data, final_list)
    total_label = index_list_list(total_label, final_list)
    return total_data, total_label, total_id


def index_list_list(list_in, index_list):
    """
    This function allows one to index a list using a list.
    For example, if we want to pick the 1st, 3rd, 4th element of a list a.
    We use a as list_in and [1, 3, 4] as the index_list.
    :param list_in: List to be indexed
    :param index_list: List of indices
    :return: List made of elements of interests
    """
    list_out = [list_in[i] for i in index_list]
    return list_out


def check_missing_meta(df, id_list):
    """
    This function reads the metadata file and returns a list of ids whose information is available in the metadata.
    :param df: data frame of the metadata
    :param id_list: list of all ids of the point clouds available.
    :return: an id list that only contain cases appeared in the metadata file.
    """
    wanted_list = []
    for i, case_id in enumerate(id_list):
        try:
            temp = read_condition(df, int(id_list[i]))
            wanted_list.append(i)
        except IndexError:
            pass
    id_out = [id_list[i] for i in wanted_list]
    return id_out, wanted_list


def get_vectors(args):
    # If names of the npy files change, indexing of id needs to be changed.
    # Currently the function only works with a format like： latent_space_1002549.npy
    data_dir = args.data_dir
    disease_dir = args.disease_dir
    reduce_size = args.reduce_size

    matrix0, id_list0 = readPC2(data_dir, reduce_size)
    matrix1, id_list1 = readPC(disease_dir, reduce_size)

    # Shuffle the data of healthy cases and then take the same number as the diseased cases
    i_list = shuffle_matrix(matrix0)
    matrix0 = matrix0[i_list]
    id_list0 = id_list0[i_list]
    dim1 = matrix1.shape
    n_positive = dim1[0]
    n_negative = n_positive * 1  # Tune ratio between positive and negative cases
    matrix0 = matrix0[0:n_negative]  # To make diseased cases and healthy cases balance
    id_list0 = id_list0[0:n_negative]
    return matrix0, matrix1, id_list0, id_list1


def get_vectors_not_same(args):
    """
    This function reads and stores the point clouds of diseased and healthy cases in the given directories.
    0 stands for healthy and 1 stands for diseased.
    """
    # If names of the npy files change, indexing of id needs to be changed.
    # Currently the function only works with a format like： latent_space_1002549.npy
    data_dir = args.data_dir
    disease_dir = args.disease_dir
    reduce_size = args.reduce_size

    # This step chooses cases with 0 disease on record and avoid mismatch between the data in the directory and the
    # metadata.
    matrix0, id_list0 = readPC2(data_dir, reduce_size)
    matrix1, id_list1 = readPC(disease_dir, reduce_size)

    i_list = shuffle_matrix(matrix0)
    matrix0 = matrix0[i_list]
    id_list0 = id_list0[i_list]
    return matrix0, matrix1, id_list0, id_list1


def get_vectors_not_same2(args):
    """
    This function reads and stores the point clouds of diseased and healthy cases in the given directories.
    0 stands for healthy and 1 stands for diseased.
    """
    # If names of the npy files change, indexing of id needs to be changed.
    # Currently the function only works with a format like： latent_space_1002549.npy
    data_dir = args.data_dir
    disease_dir = args.disease_dir
    reduce_size = args.reduce_size

    # This step chooses cases with 0 disease on record and avoid mismatch between the data in the directory and the
    # metadata.
    matrix0, id_list0 = readPC(data_dir, reduce_size)
    matrix1, id_list1 = readPC(disease_dir, reduce_size + 1)
    print(Fore.GREEN + 'Single phase data(ES/ED) are read correctly' + Style.RESET_ALL)
    # i_list = shuffle_matrix(matrix0)
    # matrix0 = matrix0[i_list]
    # id_list0 = id_list0[i_list]
    return matrix0, matrix1, id_list0, id_list1


def get_vectors_not_same21(args):
    """
    Extension of get_vectors_not_same2.
    It reads the point clouds of ED and ES respectively and concatenates them to make the overall point clouds.
    """
    # If names of the npy files change, indexing of id needs to be changed.
    # Currently the function only works with a format like： latent_space_1002549.npy
    reduce_size = args.reduce_size
    disease_es_dir, disease_ed_dir = get_es_ed_path(args.disease_dir)
    healthy_es_dir, healthy_ed_dir = get_es_ed_path(args.data_dir)

    # This step chooses cases with 0 disease on record and avoid mismatch between the data in the directory and the
    # metadata.
    matrix00, id_list0 = readPC(healthy_es_dir, reduce_size)
    matrix01, id_list0 = readPC(healthy_ed_dir, reduce_size)
    matrix10, id_list1 = readPC(disease_es_dir, reduce_size + 1)
    matrix11, id_list1 = readPC(disease_ed_dir, reduce_size + 1)
    matrix0 = np.concatenate((matrix00, matrix01), axis=2)
    matrix1 = np.concatenate((matrix10, matrix11), axis=2)
    print_concatenation_info(matrix00, matrix0)
    print(Fore.GREEN + 'ES and ED point clouds have been successfully concatenated' + Style.RESET_ALL)
    # Below shuffles the healthy data.
    # i_list = shuffle_matrix(matrix0)
    # matrix0 = matrix0[i_list]
    # id_list0 = id_list0[i_list]
    return matrix0, matrix1, id_list0, id_list1


def print_concatenation_info(m_before, m_after):
    shape_b = m_before.shape
    shape_a = m_after.shape
    print(Fore.GREEN + 'Number of points before concatenation: {}'.format(shape_b[2]) + Style.RESET_ALL)
    print(Fore.GREEN + 'Number of points after concatenation: {}'.format(shape_a[2]) + Style.RESET_ALL)


def get_vectors2(args):
    # If names of the npy files change, indexing of id needs to be changed.
    # Currently the function only works with a format like： latent_space_1002549.npy
    data_dir = args.data_dir
    disease_dir = args.disease_dir
    reduce_size = args.reduce_size

    matrix0, id_list0 = readPC22(data_dir, reduce_size)
    matrix1, id_list1 = readPC12(disease_dir, reduce_size)

    # Shuffle the data of healthy cases and then take the same number as the diseased cases
    i_list = shuffle_matrix(matrix0)
    matrix0 = matrix0[i_list]
    id_list0 = id_list0[i_list]
    # dim1 = matrix1.shape
    # n_positive = dim1[0]
    # n_negative = n_positive*1  # Tune ratio between positive and negative cases
    # matrix0 = matrix0[0:n_negative]  # To make diseased cases and healthy cases balance
    # id_list0 = id_list0[0:n_negative]
    return matrix0, matrix1, id_list0, id_list1


def get_vectors3(args):
    # If names of the npy files change, indexing of id needs to be changed.
    # Currently the function only works with a format like： latent_space_1002549.npy
    data_dir = args.data_dir
    disease_dir = args.disease_dir
    reduce_size = args.reduce_size

    matrix0, id_list0 = readPC22(data_dir, reduce_size)
    matrix1, id_list1 = readPC12(disease_dir, reduce_size)

    # Shuffle the data of healthy cases and then take the same number as the diseased cases
    i_list = shuffle_matrix(matrix0)
    matrix0 = matrix0[i_list]
    id_list0 = id_list0[i_list]
    dim1 = matrix1.shape
    n_positive = dim1[0]
    n_negative = n_positive * 1  # Tune ratio between positive and negative cases
    matrix0 = matrix0[0:n_negative]  # To make diseased cases and healthy cases balance
    id_list0 = id_list0[0:n_negative]
    return matrix0, matrix1, id_list0, id_list1


def label_data(matrix0, matrix1):
    shape0 = matrix0.shape
    shape1 = matrix1.shape
    label0 = np.zeros((shape0[0], 1))
    label1 = np.ones((shape1[0], 1))
    return label0, label1


def readPC(file_dir, reduce_size):
    """This function takes in a file list and reads off all files in the list. It returns
    a matrix containing all the point clouds"""
    filelist = glob.glob(file_dir + '/*.npy')
    file_count = 0
    nfiles = len(filelist)
    Case_id = np.zeros((nfiles, 1))
    print('Reading point clouds from directories...')
    for fname in filelist:
        case_id = read_id(fname)
        Case_id[file_count] = int(case_id)
        lat_data = np.load(fname)
        data_in = lat_data[:, 0:3]
        data_in = utils.reduce_pc_size(data_in, reduce_size)
        data_in = np.transpose(data_in)
        if file_count == 0:
            PC_dim = data_in.shape
            n_points = PC_dim[1]
            matrix_out = np.zeros((nfiles, 3, n_points))
        matrix_out[file_count] = data_in
        file_count += 1

    if file_count < nfiles:
        # This step is to get rid of the effect of cases that appear in metadata but not in the point cloud directory.
        matrix_out = matrix_out[:file_count]
        Case_id = Case_id[:file_count]

    return matrix_out, Case_id


def read_id(file_name):
    """
    This function extracts the case id in a file name.
    :param file_name: name of the file in string
    :return: case id of the file in string
    """
    case_id = ''
    temp = os.path.normpath(file_name)
    last = os.path.basename(temp)
    for elem in last:
        if elem.isdigit():
            case_id += elem
    return int(case_id)


def readPC12(file_dir, reduce_size):
    """This function takes in a file list and reads off all files in the list. It returns
    a matrix containing all the point clouds"""
    filelist = glob.glob(file_dir + '/*.npy')
    file_count = 0
    nfiles = len(filelist)
    Case_id = np.zeros((nfiles, 1))
    for fname in filelist:
        case_id = int(fname[-11:-4])
        Case_id[file_count] = int(case_id)
        lat_data = np.load(fname)
        data_in = lat_data
        if file_count == 0:
            PC_dim = data_in.shape
            n_points = PC_dim[1]
            matrix_out = np.zeros((nfiles, 3, n_points))
        matrix_out[file_count] = data_in
        file_count += 1

    if file_count < nfiles:
        # This step is to get rid of the effect of cases that appear in metadata but not in the point cloud directory.
        matrix_out = matrix_out[:file_count]
        Case_id = Case_id[:file_count]

    return matrix_out, Case_id


def readPC2(file_dir, reduce_size):
    """
    This function takes in a file list and reads off all files in the list. It returns
    a matrix containing all the point clouds.
    It only reads cases with no disease at all, based on the metadata 'No_MF.csv'
    """
    csv_dir = PathString('data/csv/No_MF.csv')
    df_no_MF = pd.read_csv(csv_dir.ab)
    # case_list1 = get_people_with_disease(1, df_no_MF, include=False)
    # case_list2 = get_people_with_disease(2, df_no_MF, include=False)
    # case_list = case_list1 + case_list2
    case_list = get_people_with_disease(0, df_no_MF)
    filelist = glob.glob(file_dir + '/*.npy')
    file_count = 0
    nfiles = len(case_list)
    Case_id = np.zeros((nfiles, 1))
    for fname in filelist:
        case_id = fname[-11:-4]
        if case_id in case_list:
            Case_id[file_count] = int(case_id)
            lat_data = np.load(fname)
            data_in = lat_data[:, 0:3]
            data_in = utils.reduce_pc_size(data_in, reduce_size)
            data_in = np.transpose(data_in)
            if file_count == 0:
                PC_dim = data_in.shape
                n_points = PC_dim[1]
                matrix_out = np.zeros((nfiles, 3, n_points))
            matrix_out[file_count] = data_in
            file_count += 1

    if file_count < nfiles:
        # This step is to get rid of the effect of cases that appear in metadata but not in the point cloud directory.
        matrix_out = matrix_out[:file_count]
        Case_id = Case_id[:file_count]
    return matrix_out, Case_id


def readPC22(file_dir, reduce_size):
    """
    This function takes in a file list and reads off all files in the list. It returns
    a matrix containing all the point clouds.
    It only reads cases with no disease at all, based on the metadata 'No_MF.csv'
    """
    csv_dir = PathString('data/csv/No_MF.csv')
    df_no_MF = pd.read_csv(csv_dir.ab)
    case_list = get_people_with_disease(0, df_no_MF)
    filelist = glob.glob(file_dir + '/*.npy')
    file_count = 0
    nfiles = len(case_list)
    Case_id = np.zeros((nfiles, 1))
    for fname in filelist:
        case_id = fname[-11:-4]
        if case_id in case_list:
            Case_id[file_count] = int(case_id)
            lat_data = np.load(fname)
            data_in = lat_data
            if file_count == 0:
                PC_dim = data_in.shape
                n_points = PC_dim[1]
                matrix_out = np.zeros((nfiles, 3, n_points))
            matrix_out[file_count] = data_in
            file_count += 1

    if file_count < nfiles:
        # This step is to get rid of the effect of cases that appear in metadata but not in the point cloud directory.
        matrix_out = matrix_out[:file_count]
        Case_id = Case_id[:file_count]
    return matrix_out, Case_id


def count_nfiles(file_dir):
    initial_count = 0
    for path in os.listdir(file_dir):
        if os.path.isfile(os.path.join(file_dir, path)):
            initial_count += 1
    return initial_count


def shuffle_matrix(data_in):
    """
    This function shuffles any arrays along its first dimension. It returns the shuffled index list so that
    any other arrays could be shuffled exactly the same.
    :param data_in:
    :return:
    """
    dim = data_in.shape
    i_list = list(range(dim[0]))
    random.shuffle(i_list)
    return i_list


def shuffle_list(data_in):
    """
    This function shuffles any arrays along its first dimension. It returns the shuffled index list so that
    any other arrays could be shuffled exactly the same.
    :param data_in:
    :return:
    """
    dim = len(data_in)
    i_list = list(range(dim))
    random.shuffle(i_list)
    return i_list


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


class PathString(str):
    def __init__(self, path):
        self.value = path
        self.ab = use_root_dir(path)


def get_es_ed_path(data_dir):
    ES_path = data_dir + '/ES'
    ED_path = data_dir + '/ED'
    return ES_path, ED_path


if __name__ == '__main__':
    a = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    print(move_to_the_end(a, 3))
    # print(a[:3])
