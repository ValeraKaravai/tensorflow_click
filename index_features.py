import MySQLdb
from pandas.io.json import json_normalize
import json
import feature_constant


def get_index_feature(name_table, id_config):
    db = MySQLdb.connect(user='', passwd='',
                         host="",
                         db='')
    cur = db.cursor()
    query = 'SELECT value FROM %s WHERE id = \'%s\'' % (name_table, id_config)
    cur.execute(query)
    configuration_regression = json_normalize(json.loads(cur.fetchall()[0][0]))
    db.close()

    # features = configuration_regression['features'][0]
    factors = feature_constant.DEFAULT_FEATURE + configuration_regression['factors'][0]

    integer_column = []
    float_name = []
    categorical_name = []

    print factors

    for i, factor in enumerate(factors):
        if factor in feature_constant.INTEGER_COLUMN_NAME:
            integer_column.append(i)
        if factor in feature_constant.FLOAT_COLUMN_NAME:
            float_name.append(i)
        if factor in feature_constant.CATEGORICAL_COLUMN_NAME:
            categorical_name.append(i)

    return integer_column, float_name, categorical_name


print get_index_feature('test', 'onclick')
