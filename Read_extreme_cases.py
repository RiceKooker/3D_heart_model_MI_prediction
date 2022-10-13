import numpy as np


def read_extreme_cases(extreme_dict, exp_id=0, error_type='tp', get_list=False):
    """
    This function reads from the total dictionary that contains all the case ids and scores for 4 different
    types of errors.
    :param extreme_dict: the overall file
    :param exp_id: the specific test scenario wanted. Needs to be int type.
    :param error_type: this can either be 'tp', 'fp', 'fn', 'tn', which stand for true positive, false positive,
    false negative, true negative.
    :param get_list: if True, this function prints the list of experiment ids.
    :return:
    """
    if get_list:
        for key1 in extreme_dict:
            print(key1)
        return
    case_dict = extreme_dict[exp_id]
    error_cases = case_dict[error_type]
    return error_cases['id'], error_cases['score']


if __name__ == "__main__":
    file_path = '12_each.npy'
    total_dict = np.load(file_path, allow_pickle=True).item()
    case_id, scores = read_extreme_cases(total_dict, exp_id=7581, error_type='fn')
    print('Case id Score')
    print('-------------')
    for each_id, each_score in zip(case_id, scores):
        print(each_id, round(each_score, 3))
