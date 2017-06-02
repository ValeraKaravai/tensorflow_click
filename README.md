# Preprocessing function

```
python preprocess.py --training_data data/train-10k.txt \
                     --eval_data data/eval-1k.txt \
                     --output_dir output
```

# Function

## main()

1. args_parse (read input/output)
2. Type pipeline (`DirectRunner`)
3. Create tmp dir
4. Create object `p -> pipeline beam` with name and options
5. Context (tmp_dir)
6. Run preprocessing function

## preprocessing()

* **Описание:** Pre-processing функция с Pipeline.
* **Input:**
    * pipeline: beam pipeline
    * training_data: file paths to input csv files.
    * eval_data: file paths to input csv files.
    * predict_data: file paths to input csv files.
    * output_dir: папка с output
    * frequency_threshold: используется для категориальных переменных

1. Run `ads.make_input_schema()` создание схемы данных (`input_schema type Schema`)
2. Run `ads.make_tsv_coder(input_schema)`  сопаставление исходных данных-колонок схеме данных, 
создание тензоров (`coder type coders`)
3. `training_data:` чтение input файла с использование `coder` в pipeline = p. 
`'ReadTrainigSet' >> 'ParseTrainingSet'` 
4. `evaluate_data` аналогично с `training_data`
5. `input_metadata`:  dataset_metadata.DatasetMetadata записывают в метадату нашу схему.
6. `_` запись файлов на диск ` tft_beam_io.WriteMetadata(path, pipeline)` в RAW_METADATA_DIR
7. `preprocessing_f`: ads.make_preprocessing_f(treshold)
8. `train_dataset, train_metadata с помощью input_metadata+training_data преобразовываем 
файл через preprocessing_f
9. `transform_f` с помощью transform_f записываем файл в фиксированные каталоги. `output_dir` и через параметры
TRANSFORMED_METADATA_DIR и TRANSFORM_FN_DIR записывает метаданные.
10. Делаем тоже самое с evaluate_data. Только в данном случае Analyze уже не производим, 
даем готовый transform_f.
11. `train_coder` кодируем train сет и записывем преобразованны
12. `eval_coder` тоже самое. 
13. В случае тестового сета:
    * меняем mode на INFER
    * запускаем `ads.make_input_schema()` с новым mode
    * и весь процесс снова только для test_set.

### ads.make_input_schema()

* **Описание:** определение схемы input
* **Input:** mode - указывает режим, в котором необходимо составить схему (train/eval/prediction)
* **Returns:** объект типа `Schema`

1. mode: train [doc](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/ModeKeys).
Стандартные имена для режимов модели. Определены следующие стандартные режимы:
    * `TRAIN`: Режим обучения.
    * `EVAL`: Режим оценки.
    * `INFER`: Режим вывода.
2. `result`. `If INFER then {} else create clicked (Fixed Lenght Feature = type int)`
3. result add INTEGER and CATEGORICAL features.


### ads.make_tsv_coder(input_schema)

* **Описание:** производит кодирование файла csv (tab = separator) из схемы данных
* **Input:** 
    * schema: input schema (type tf.transform Schema)
    * mode - указывает режим, в котором необходимо составить схему (train/eval/prediction)
* **Returns:** объект типа `tf.Transform CsvCoder.`

1. `column_name:` пустой list, если INFER mode иначе clicked
2. Добавление имен колонок INTEGER/CATEGORICAL  в переменную 
`column_name: {'clicked', 'int-feature-1', 'categorical-feature-1'}`
3. Ковeртирование csv with `delimiter = '\t'`. По сути - map(column_name, schema)

### ads.make_preprocessing_f(treshold)

* **Описание:** создает функцию препроцессинга.
* **Input:** 
    * frequency_threshold: порог частоты, используемый при генерации словаря для категориальных фич.
* **Returns:** функцию препроцессинга. 

### preprocessing_f(inputs)

* **Описание:** пользовательская функция для препроцессинга столбцов
* **Input:** 
    * inputs: словарь входных значений
* **Returns:** словарь преобразованных колонок  
