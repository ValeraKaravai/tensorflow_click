import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import coders
from tensorflow_transform.tf_metadata import dataset_schema
import config.config_preprocessing as args


def make_input_schema(mode=tf.contrib.learn.ModeKeys.TRAIN):
    result = ({} if mode == tf.contrib.learn.ModeKeys.INFER
              else {'clicked': tf.FixedLenFeature(shape=[], dtype=tf.int64)})
    for name in args.INTEGER_COLUMN:
        result[name] = tf.FixedLenFeature(
            shape=[], dtype=tf.int64, default_value=-1
        )
    for name in args.CATEGORICAL_COLUMN:
        result[name] = tf.FixedLenFeature(
            shape=[], dtype=tf.string, default_value=''
        )
    return dataset_schema.from_feature_spec(result)


def make_tsv_coder(schema, mode=tf.contrib.learn.ModeKeys.TRAIN):
    column_names = [] if mode == tf.contrib.learn.ModeKeys.INFER else ['clicked']
    for name in args.INTEGER_COLUMN:
        column_names.append(name)
    for name in args.CATEGORICAL_COLUMN:
        column_names.append(name)

    return coders.CsvCoder(column_names, schema, delimiter='\t')


def make_preprocessing_f(frequency_treshold):
    def preprocessing_f(inputs):
        result = {'clicked': inputs['clicked']}
        for name in args.INTEGER_COLUMN:
            result[name] = inputs[name]
        for name in args.CATEGORICAL_COLUMN:
            result[name + '_id'] = tft.string_to_int(
                inputs[name], frequency_threshold=frequency_treshold
            )

        return result

    return preprocessing_f




