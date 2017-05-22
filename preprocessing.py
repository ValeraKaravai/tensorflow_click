import apache_beam as beam
import tensorflow as tf
from tensorflow_transform import coders
from tensorflow_transform.beam import impl as tft
from tensorflow_transform.beam import tft_beam_io
from tensorflow_transform.tf_metadata import dataset_metadata

import argparse
import sys
import os
import random

import ads
import path_constants

@beam.ptransform_fn
def _Shuffle(pcoll):
    return (pcoll
            | 'PairWithRandom' >> beam.Map(lambda x: (random.random(), x))
            | 'GroupByRandom' >> beam.GroupByKey()
            | 'Droprandom'>> beam.FlatMap(lambda (k, vs):vs))


def _encode_as_b64_json(serialized_example):
    import base64  # pylint: disable=g-import-not-at-top
    import json  # pylint: disable=g-import-not-at-top
    return json.dumps({'b64': base64.b64encode(serialized_example)})


def parse_argument(argv):
    """Parse command line parameters"""
    parser = argparse.ArgumentParser(
        description='Runs preprocessing data on the Propeller data model')

    parser.add_argument(
        '--project_id', help='The project to which the job will be submitted'
    )

    parser.add_argument(
        '--frequency_treshold',
        type=int,
        default=100,
        help='The frequency threshold categorical values are'
    )

    parser.add_argument(
        '--training_set',
        default='data/train_10k.txt',
        #required=True,
        help='Training set'
    )

    parser.add_argument(
        '--eval_set',
        default='data/eval_1k.txt',
        #required=True,
        help='Test set'
    )

    parser.add_argument(
        '--test_set',
        #required=True,
        help='Test set'
    )

    parser.add_argument(
        '--output_dir',
        default='output',
        #required=True,
        help='Output directory'
    )

    args, _ = parser.parse_known_args(args=argv[1:])

    return args


def preprocessing(pipline, training_set, eval_set, test_set, output_dir, frequency_treshold):
    input_schema = ads.make_input_schema()

    coder = ads.make_tsv_coder(input_schema)

    training_data = (
        pipline
        | 'ReadTrainingSet' >> beam.io.ReadFromText(training_set)
        | 'ParseTrainingSet' >> beam.Map(coder.decode)
    )

    evaluate_data = (
        pipline
        | 'ReadEvalData' >> beam.io.ReadFromText(eval_set)
        | 'ParseEvalCsv' >> beam.Map(coder.decode))

    input_metadata = dataset_metadata.DatasetMetadata(schema=input_schema)
    _= (input_metadata
        | 'WriteInputMetadata' >> tft_beam_io.WriteMetadata(
            os.path.join(output_dir, path_constants.RAW_METADATA_DIR),
            pipeline=pipline))

    preprocessing_f = ads.make_preprocessing_f(frequency_treshold)
    (train_dataset, train_metadata), transform_f = (
        (training_data, input_metadata)
        | 'AnalyzeAndTransform' >> tft.AnalyzeAndTransformDataset(preprocessing_f)
    )

    _ = (transform_f
         | 'WriteTransformFn' >> tft_beam_io.WriteTransformFn(output_dir))

    ##TODO eval dataset

    (evaluate_dataset, evaluate_metadata) = (
        ((evaluate_data, input_metadata), transform_f)
        | 'TransformEval' >> tft.TransformDataset())

    train_coder = coders.ExampleProtoCoder(train_metadata.schema)
    _ = (train_dataset
         | 'SerializeTrainExample' >> beam.Map(train_coder.encode)
         | 'ShuffleTraining' >> _Shuffle()
         | 'WriteTraining' >> beam.io.WriteToTFRecord(
                            os.path.join(output_dir, path_constants.TRANSFORMED_TRAIN_DATA_FILE_PREFIX),
                            file_name_suffix='.tfrecord.gz'))

    evaluate_coder = coders.ExampleProtoCoder(evaluate_metadata.schema)
    _ = (evaluate_dataset
         | 'SerializeEvalExamples' >> beam.Map(evaluate_coder.encode)
         | 'ShuffleEval' >> _Shuffle()  # pylint: disable=no-value-for-parameter
         | 'WriteEval'
         >> beam.io.WriteToTFRecord(
        os.path.join(output_dir,
                     path_constants.TRANSFORMED_EVAL_DATA_FILE_PREFIX),
        file_name_suffix='.tfrecord.gz'))
    if test_set:
        predict_mode = tf.contrib.learn.ModeKeys.INFER
        predict_schema = ads.make_input_schema(mode=predict_mode)
        tsv_coder = ads.make_tsv_coder(predict_schema, mode = predict_mode)
        predict_coder = coders.ExampleProtoCoder(predict_schema)

        serialized_example = (
            pipline
            | 'ReadPredictData' >> beam.io.ReadFromText(test_set)
            | 'ParsePredictCsv' >> beam.Map(tsv_coder.decode)
            | 'EncodePredictData' >> beam.Map(predict_coder.encode))
        _ = (serialized_example
             | 'WritePredictDataAsTFRecord' >> beam.io.WriteToTFRecord(
                    os.path.join(output_dir, path_constants.TRANSFORMED_PREDICT_DATA_FILE_PREFIX),
                    file_name_suffix ='.tfrecord.gz'))

        _ = (serialized_example
             | 'EncodePredictAsB64JSON' >> beam.Map(_encode_as_b64_json)
             | 'WritePredictDataAsText' >> beam.io.WriteToText(
                    os.path.join(output_dir, path_constants.TRANSFORMED_PREDICT_DATA_FILE_PREFIX),
                    file_name_suffix='.txt'))



def main(argv=None):
    args = parse_argument(sys.argv if argv is None else argv)

    ##TODO add parameters for cloud

    pipline_name = 'DirectRunner'
    pipline_options = None

    tmp_dir = os.path.join(args.output_dir, 'tmp')
    with beam.Pipeline(pipline_name, options=pipline_options) as p:
        with tft.Context(temp_dir=tmp_dir):
            preprocessing(pipline=p, training_set=args.training_set, eval_set=args.eval_set, test_set=args.test_set,
                          output_dir=args.output_dir, frequency_treshold=args.frequency_treshold)



if __name__ == '__main__':
    main()



