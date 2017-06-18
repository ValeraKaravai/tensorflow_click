import pandas as pd
import feature_constant


def modification_set(path_data, new_path):
    dataset = pd.read_csv(path_data, sep=' ')
    # dataset_modification = dataset[feature_constant.INTEGER_COLUMN_NAME + feature_constant.CATEGORICAL_COLUMN_NAME]
    dataset_modification = dataset[['conversions', 'impressions', 'zone_id', 'geo']]
    dataset_modification.to_csv(new_path, sep='\t', header=False, index=False )


def modification_set_criteo(path_data, new_path):
    dataset = pd.read_csv(path_data, sep='\t', header=None, dtype=str)
    # dataset_modification = dataset.iloc[:, [0, 1, 14, 15]]
    # print dataset_modification.iloc[0:3, :]
    dataset_modification = dataset
    dataset_modification.to_csv(new_path, sep='\t', header=False, index=False)


# TODO delete name files
modification_set('data/propeller/onclick_train.tsv', 'data/propeller/train.tsv')
modification_set('data/propeller/onclick_test.tsv', 'data/propeller/eval.tsv')


modification_set_criteo('data/criteo/eval_1k.txt', 'data/criteo_big/eval.tsv')
modification_set_criteo('data/criteo/train_10k.txt', 'data/criteo_big/train.tsv')
