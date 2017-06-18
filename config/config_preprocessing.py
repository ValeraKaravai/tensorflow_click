import config.path_constants as path_constants
import config.config_constants as config
import os

# Generate path
training_set = os.path.join(path_constants.INPUT_PATH, config.ID_CONFIGURATION, path_constants.TRAIN_DATA)
eval_set = os.path.join(path_constants.INPUT_PATH, config.ID_CONFIGURATION, path_constants.EVAL_DATA)
# test_set = os.path.join(path_constants.INPUT_PATH, ID_CONFIGURATION, path_constants.TEST_DATA)
test_set = None
output_dir = os.path.join(path_constants.OUTPUT_PREPROCESSING, config.ID_CONFIGURATION)


INTEGER_COLUMN = [
    'int-feature-{}'.format(column_idx) for column_idx in config.INTEGER_COLUMN_NUM
]

CATEGORICAL_COLUMN = [
    'categorical-feature-{}'.format(column_idx) for column_idx in config.CATEGORICAL_COLUMN_NUM
]