import config.config_learn as args
import config.config_constants as config
import tensorflow as tf

from tensorflow_transform.saved import input_fn_maker
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow.contrib.learn.python.learn import learn_runner


def feature_columns(vocab_sizes, use_crosses):
    result = []
    boundaries = [1.5 ** j - 0.51 for j in range(config.FEATURE_NUM)]

    for index in config.INTEGER_COLUMN_NUM:
        column = tf.contrib.layers.bucketized_column(
            tf.contrib.layers.real_valued_column(
                config.FORMAT_INT_FEATURE.format(index),
                dtype=tf.int64),
            boundaries)
        result.append(column)

    for index in config.CATEGORICAL_COLUMN_NUM:
        column_name = config.FORMAT_CATEGORICAL_FEATURE_ID.format(index)
        vocab_size = vocab_sizes[column_name]
        column = tf.contrib.layers.sparse_column_with_integerized_feature(
            column_name, vocab_size, combiner='sum')
        result.append(column)
    if use_crosses:
        for cross in config.CROSSES:
            column = tf.contrib.layers.crossed_column(
                [result[index - 1] for index in cross],
                hash_bucket_size=config.CROSS_HASH_BUCKET_SIZE,
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
        label_keys=[config.TARGET_FEATURE_COLUMN],
        reader=gzip_reader_fn,
        key_feature_name=config.KEY_FEATURE_COLUMN,
        reader_num_threads=4,
        queue_capacity=batch_size * 2,
        randomize_input=(mode != tf.contrib.learn.ModeKeys.EVAL),
        num_epochs=(1 if mode == tf.contrib.learn.ModeKeys.EVAL else None))


def get_vocab_sizes():
    return {config.FORMAT_CATEGORICAL_FEATURE_ID.format(index): int(10 * 1000)
            for index in config.CATEGORICAL_COLUMN_NUM}


def get_experiment_fn():
    vocab_sizes = get_vocab_sizes()
    use_crosses = not config.ignore_crosses

    def get_experiment(output_dir):
        columns = feature_columns(
            vocab_sizes, use_crosses)

        runconfig = tf.contrib.learn.RunConfig()
        cluster = runconfig.cluster_spec
        num_table_shards = max(1, runconfig.num_ps_replicas * 3)
        num_partitions = max(1, 1 + cluster.num_tasks('worker')
                             if cluster and 'worker' in cluster.jobs else 0)

        l2_regularization = config.l2_regularization
        estimator = tf.contrib.learn.LinearClassifier(
            model_dir=output_dir,
            feature_columns=columns,
            optimizer=tf.contrib.linear_optimizer.SDCAOptimizer(
                example_id_column=config.KEY_FEATURE_COLUMN,
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
                raw_label_keys=[config.TARGET_FEATURE_COLUMN]))
        export_strategy = (
            tf.contrib.learn.utils.make_export_strategy(
                serving_input_fn, exports_to_keep=5,
                default_output_alternative_key=None))

        train_input_fn = get_transformed_reader_input_fn(
            transformed_metadata, args.train_data_paths, config.batch_size,
            tf.contrib.learn.ModeKeys.TRAIN)
        eval_input_fn = get_transformed_reader_input_fn(
            transformed_metadata, args.eval_data_paths, config.batch_size,
            tf.contrib.learn.ModeKeys.EVAL)

        m = tf.contrib.learn.Experiment(
            estimator=estimator,
            train_steps=config.train_steps,
            eval_steps=config.eval_steps,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            export_strategies=export_strategy,
            min_eval_frequency=500)
        return m
    return get_experiment


def main():
    learn_runner.run(experiment_fn=get_experiment_fn(),
                     output_dir=args.output_dir)


if __name__ == '__main__':
    main()
