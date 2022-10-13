import os
import colorama
from colorama import Fore, Back, Style


def change_file_name(mode, new_code):
    """
    This function changes the name of the temp files generated from each experiment.
    :param mode: this can be 'fold' or 'AUC', indicating different experiment mode.
    :param new_code: this is the code to be assigned in the form of int.
    :return:
    """
    new_code = str(new_code)
    colorama.init(autoreset=True)
    print(os.getcwd())
    try:
        if mode == 'fold':
            old_name = 'temp.npy'
            base_dir = os.getcwd() + '/Fold_history/'
            new_name = 'Exp' + old_name.replace('temp', new_code)
            os.rename(base_dir + old_name, base_dir + new_name)
            print(Fore.BLUE + 'File name ' + old_name + ' is changed to ' + new_name)
        elif mode == 'AUC':
            old_names = ['fold_history_ROC_temp.npy', 'FPR_temp.npy', 'TPR_temp.npy']
            base_dir = os.getcwd() + '/AUC_history/'
            for old_name in old_names:
                new_name = old_name.replace('temp', new_code)
                os.rename(base_dir + old_name, base_dir + new_name)
                print(Fore.BLUE + 'File name ' + old_name + ' is changed to ' + new_name)
    except FileNotFoundError:
        print(Fore.RED + 'Temporary files do not exist')


if __name__ == "__main__":
    change_file_name('AUC', 123123123)