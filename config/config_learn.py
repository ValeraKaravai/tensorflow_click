import glob
import config.path_constants as path_constants
import config.config_constants as config
import os

# Path do not parameters
input_dir = os.path.join(path_constants.OUTPUT_PREPROCESSING, config.ID_CONFIGURATION)
output_dir = os.path.join(path_constants.TRAIN_RESULTS_FILE, config.ID_CONFIGURATION)
train_data_paths = glob.glob(
    os.path.join(input_dir, path_constants.TRANSFORMED_TRAIN_DATA_FILE_PREFIX) + '*')
eval_data_paths = glob.glob(
    os.path.join(input_dir, path_constants.TRANSFORMED_EVAL_DATA_FILE_PREFIX) + '*')

# Files for transform tn
raw_metadata_path = os.path.join(input_dir, path_constants.RAW_METADATA_DIR)
transformed_metadata_path = os.path.join(input_dir, path_constants.TRANSFORMED_METADATA_DIR)
transform_savedmodel = os.path.join(input_dir, path_constants.TRANSFORM_FN_DIR)
