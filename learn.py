import argparse
import sys

import tensorflow as tf

from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.saved import input_fn_maker

MODEL_TYPES = 'linear'
LINEAR = MODEL_TYPES
DATASETS = 'onclick'
ONCLICK = DATASETS
CROSSES = 'crosses'
NUM_EXAMPLES = 'num_examples'
L2_REGULARIZATION = 'l2_regularization'

KEY_FEATURE_COLUMN = 'example_id'
TARGET_FEATURE_COLUMN = 'clicked'

CROSS_HASH_BUCKET_SIZE = int(1e6)

MODEL_DIR = 'model'

FORMAT_CATEGORICAL_FEATURE_ID = 'categorical-feature-{}_id'
FORMAT_INT_FEATURE = 'int-feature-{}'

PIPELINE_CONFIG = {
    ONCLICK: {
        NUM_EXAMPLES:
            45 * 1e2,
        L2_REGULARIZATION:
            60,
        CROSSES: [(2, 3)]
    }
}


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        help='Onclick dataset',
        choices=DATASETS,
        # required=True
        default='onclick')
    parser.add_argument(
        '--model_type',
        help='Model type to train on',
        choices=MODEL_TYPES,
        default=LINEAR)
    parser.add_argument(
        '--train_data_paths', type=str, action='append',
        # required=True
        default='data/onclick_train_mod.tsv')
    parser.add_argument(
        '--eval_data_paths', type=str, action='append',
        # required=True
        default='data/onclick_test_mod.tsv')
    parser.add_argument('--output_path', type=str,
                        # required=True
                        default='train')
    # After preprocessing
    parser.add_argument('--raw_metadata_path', type=str,
                        # required=True
                        default='output/raw_metadata')
    parser.add_argument('--transformed_metadata_path', type=str,
                        # required=True
                        default='output/transformed_metadata')
    parser.add_argument('--transform_savedmodel', type=str,
                        # required=True
                        default='output/transform_fn')
    parser.add_argument(
        '--batch_size',
        help='Number of input records used per batch',
        default=3000,
        type=int)
    parser.add_argument(
        '--eval_batch_size',
        help='Number of eval records used per batch',
        default=500,
        type=int)
    parser.add_argument(
        '--train_steps', help='Number of training steps to perform.', type=int)
    parser.add_argument(
        '--eval_steps',
        help='Number of evaluation steps to perform.',
        type=int,
        default=100)
    parser.add_argument(
        '--train_set_size',
        help='Number of samples on the train dataset.',
        type=int)
    parser.add_argument('--l2_regularization', help='L2 Regularization', type=int)
    parser.add_argument(
        '--num_epochs', help='Number of epochs', default=5, type=int)
    parser.add_argument(
        '--ignore_crosses',
        action='store_true',
        default=True,
        help='Whether to ignore crosses (linear model only).')
    return parser


def get_vocab_sizes():
    return {FORMAT_CATEGORICAL_FEATURE_ID.format(index): int(10 * 1000)
            for index in range(2, 4)}


def feature_columns(config, model_type, vocab_sizes, use_crosses):
    result = []
    boundaries = [1.5 ** j - 0.51 for j in range(4)]

    for index in range(1, 2):
        column = tf.contrib.layers.bucketized_column(
            tf.contrib.layers.real_valued_column(
                FORMAT_INT_FEATURE.format(index),
                dtype=tf.int64),
            boundaries)
        result.append(column)

    for index in range(2, 4):
        column_name = FORMAT_CATEGORICAL_FEATURE_ID.format(index)
        vocab_size = vocab_sizes[column_name]
        column = tf.contrib.layers.sparse_column_with_integerized_feature(
            column_name, vocab_size, combiner='sum')
        result.append(column)
    if use_crosses:
        for cross in config[CROSSES]:
            column = tf.contrib.layers.crossed_column(
                [result[index - 1] for index in cross],
                hash_bucket_size=CROSS_HASH_BUCKET_SIZE,
                hash_key=tf.contrib.layers.SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY,
                combiner='sum')
            result.append(column)
    return result


def gzip_reader_fn():
    return tf.TFRecordReader(options=tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP))


def get_transformed_reader_input_fn(transformed_metadata,
                                    transformed_data_paths,
                                    batch_size,
                                    mode):
    return input_fn_maker.build_training_input_fn(
      metadata=transformed_metadata,
      file_pattern=(
          transformed_data_paths[0] if len(transformed_data_paths) == 1
          else transformed_data_paths),
      training_batch_size=batch_size,
      label_keys=[TARGET_FEATURE_COLUMN],
      reader=gzip_reader_fn,
      key_feature_name=KEY_FEATURE_COLUMN,
      reader_num_threads=4,
      queue_capacity=batch_size * 2,
      randomize_input=(mode != tf.contrib.learn.ModeKeys.EVAL),
      num_epochs=(1 if mode == tf.contrib.learn.ModeKeys.EVAL else None))


def get_experiment_fn(args):
    vocab_sizes = get_vocab_sizes()
    use_crosses = not args.ignore_crosses

    def get_experiment(output_dir):
        config = PIPELINE_CONFIG.get(args.dataset)
        columns = feature_columns(config, args.model_type, vocab_sizes, use_crosses)

        runconfig = tf.contrib.learn.RunConfig()
        cluster = runconfig.cluster_spec
        num_table_shards = max(1, runconfig.num_ps_replicas * 3)
        num_partitions = max(1, 1 + cluster.num_tasks('worker') if cluster and 'worker' in cluster.jobs else 0)

        l2_regularization = args.l2_regularization
        estimator = tf.contrib.learn.LinearClassifier(
            model_dir=output_dir,
            feature_columns=columns,
            optimizer=tf.contrib.linear_optimizer.SDCAOptimizer(
                example_id_column=KEY_FEATURE_COLUMN,
                symmetric_l2_regularization=l2_regularization,
                num_loss_partitions=num_partitions,  # workers
                num_table_shards=num_table_shards))  # ps

        transformed_metadata = metadata_io.read_metadata(
            args.transformed_metadata_path)
        raw_metadata = metadata_io.read_metadata(args.raw_metadata_path)
        serving_input_fn = (
            input_fn_maker.build_parsing_transforming_serving_input_fn(
                raw_metadata,
                args.transform_savedmodel,
                raw_label_keys=[TARGET_FEATURE_COLUMN]))
        export_strategy = (
            tf.contrib.learn.utils.make_export_strategy(
                serving_input_fn, exports_to_keep=5,
                default_output_alternative_key=None))

        train_input_fn = get_transformed_reader_input_fn(
            transformed_metadata, args.train_data_paths, args.batch_size,
            tf.contrib.learn.ModeKeys.TRAIN)
        eval_input_fn = get_transformed_reader_input_fn(
            transformed_metadata, args.eval_data_paths, args.batch_size,
            tf.contrib.learn.ModeKeys.EVAL)

        train_set_size = args.train_set_size or config[NUM_EXAMPLES]
        m = tf.contrib.learn.Experiment(
            estimator=estimator,
            train_steps=10,
            # train_steps=(args.train_steps or
            #              args.num_epochs * train_set_size // args.batch_size),
            eval_steps=args.eval_steps,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            export_strategies=export_strategy,
            min_eval_frequency=500)
        return m

    return get_experiment


def main(argv=None):
    argv = sys.argv if argv is None else argv
    args = create_parser().parse_args(args=argv[1:])

    output_dir = args.output_path

    # print get_experiment_fn(args)
    # print get_experiment_fn(args)
    learn_runner.run(experiment_fn=get_experiment_fn(args),
                     output_dir=output_dir)


if __name__ == '__main__':
    main()
