import argparse
import utils.utils1 as utils
import dataset.my_dataset as my_dataset
import utils.training_utils as utils_train
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    df = pd.read_pickle('Results_dataframe/Full_with_AUC_test_5.pkl')
    df.to_excel('Results_table/Results_table_Val_AUC_optimised.xlsx')
