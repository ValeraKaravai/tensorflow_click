import MySQLdb
from pandas.io.json import json_normalize
import json
import feature_constant


def get_index_feature(id_config):
    db = MySQLdb.connect(user='read_only', passwd='PKNwYXha',
                         host="dbread.propellerads.com",
                         db='openx')
    cur = db.cursor()
    query = 'SELECT value FROM openx.ox_ecpm_regression_configuration WHERE id = \'%s\'' % id_config
    cur.execute(query)
    configuration_regression = json_normalize(json.loads(cur.fetchall()[0][0]))
    db.close()

    # features = configuration_regression['features'][0]
    factors = feature_constant.DEFAULT_FEATURE + configuration_regression['factors'][0]

    integer_column = []
    float_name = []
    categorical_name = []

    for i, factor in enumerate(factors):
        if factor in feature_constant.INTEGER_COLUMN_NAME:
            integer_column.append(i)
        if factor in feature_constant.FLOAT_COLUMN_NAME:
            float_name.append(i)
        if factor in feature_constant.CATEGORICAL_COLUMN_NAME:
            categorical_name.append(i)

    return integer_column, float_name, categorical_name


integer_column_1, float_name_1, categorical_name_1 = get_index_feature('onclick')
