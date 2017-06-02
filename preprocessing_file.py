import pandas as pd
import feature_constant


def modification_set(path_data, new_path):
    dataset = pd.read_csv(path_data, sep=' ')
    # dataset_modification = dataset[feature_constant.INTEGER_COLUMN_NAME + feature_constant.CATEGORICAL_COLUMN_NAME]
    dataset_modification = dataset[['conversions', 'impressions', 'zone_id', 'geo']]
    dataset_modification.to_csv(new_path, sep='\t', header=False, index=False )

# TODO delete name files
modification_set('data/onclick_train.tsv', 'data/onclick_train_mod.tsv')
modification_set('data/onclick_test.tsv', 'data/onclick_test_mod.tsv')
