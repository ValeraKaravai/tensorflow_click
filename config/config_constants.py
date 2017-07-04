# PARAMETERS

# Parameters preprocessing
ID_CONFIGURATION = 'onclick'
FEATURE_NUM = 4
INTEGER_COLUMN_NUM = range(1, 2)
CATEGORICAL_COLUMN_NUM = range(2, 4)
FREQUENCY_TRESHOLD = 100
CROSSES = [(2, 3)]
# Dataset structure
CROSS_HASH_BUCKET_SIZE = int(1e6)

# Format feature
FORMAT_CATEGORICAL_FEATURE_ID = 'categorical-feature-{}_id'
FORMAT_INT_FEATURE = 'int-feature-{}'
KEY_FEATURE_COLUMN = 'example_id'
TARGET_FEATURE_COLUMN = 'clicked'

# Parameters learn (model and column
batch_size = 3000
eval_batch_size = 500
train_steps = 1e3
eval_steps = 100
num_epochs = 5
ignore_crosses = True
l2_regularization = 60




