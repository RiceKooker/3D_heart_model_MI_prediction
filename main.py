import pandas as pd


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    df = pd.read_pickle('Results_dataframe/Full_with_AUC.pkl')
    print(df.head())
    df.loc[0, ['Exp id', 'Validation loss']] = [12345, 0.12345]
    print(df.head())
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
