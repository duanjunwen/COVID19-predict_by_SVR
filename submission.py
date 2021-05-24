import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error


def generate_new_df_by_dict(old_df):
    columns = old_df.columns.values[1:]  # 旧列标
    new_df_columns = ['day']  # 生成新df的列index
    count = 30
    for name in columns:
        curr_name_list = []
        while count > 0:
            curr_name_list.append(f'{name}-{count}')
            count -= 1
        count = 30
        new_df_columns = new_df_columns + curr_name_list
    new_df_dict = {column_name: [] for column_name in new_df_columns}  # 生成511个key的字典
    # len(old_df) + 1，还有第一行是index
    for i in range(30, len(old_df)):  # 前0-29提供信息，从第30行开始
        pre_thirty = old_df.iloc[i - 30: i]  # 取下old_df前30行
        new_df_dict['day'].append(i + 1)  # 先加上day
        for index, curr_column in pre_thirty.iteritems():  # 按列遍历
            if index == 'day':
                continue
            count = 30
            the_column = list(curr_column)
            for j in range(len(the_column)):
                new_df_dict[f'{index}-{count}'].append(the_column[j])
                count -= 1
    new_df = pd.DataFrame(new_df_dict)
    return new_df


#  根据要求的feature和interval切割
def cut_by_feature_and_interval(original_df, feature_name_list, weather_interval, cases_interval):
    weather_start = weather_interval  # 每个weather feature开始的位置
    case_start = cases_interval  # 每个daily_case开始的位置

    columns = feature_name_list

    weather_features = list(columns[:len(columns) - 1])  # weather_features 名字 list
    dailly_cases = columns[len(columns) - 1]  # weather_features 名字 string
    new_df_columns = []  # 生成新df的列column index

    weather_count = weather_start  # weather name 的计数器
    for name in weather_features:  # 除了dailly cases的weather feature name list
        curr_name_list = []
        while weather_count > 0:
            curr_name_list.append(f'{name}-{weather_count}')
            weather_count -= 1
        weather_count = weather_start  # 重置计数器
        new_df_columns = new_df_columns + curr_name_list

    daily_case_count = case_start  # daily_case的计数器
    daily_case_name_list = []
    while daily_case_count > 0:
        daily_case_name_list.append(f'{dailly_cases}-{daily_case_count}')
        daily_case_count -= 1
    new_df_columns = new_df_columns + daily_case_name_list

    new_df_dict = {column_name: [] for column_name in new_df_columns}  # 生成511个key的字典

    for index, curr_columns in original_df.iteritems():  # 按列遍历
        the_column = list(curr_columns)
        if index in new_df_dict.keys():
            new_df_dict[index] = the_column

    new_df = pd.DataFrame(new_df_dict)
    return new_df


## Project-Part1
def predict_COVID_part1(svm_model, train_df, train_labels_df, past_cases_interval, past_weather_interval, test_feature):
    new_train_df = generate_new_df_by_dict(train_df)  # 生成大的train_df

    train_X = cut_by_feature_and_interval(new_train_df, ["max_temp", "max_dew", "max_humid", "dailly_cases"],
                                          past_weather_interval,
                                          past_cases_interval)

    test_X_dict = dict()
    for index, item in test_feature.iteritems():
        if index != "day":
            test_X_dict[index] = [item]
    test_X_before_cut = pd.DataFrame(test_X_dict)  # 输入的是列，把它变成行
    test_X = cut_by_feature_and_interval(test_X_before_cut, ["max_temp", "max_dew", "max_humid", "dailly_cases"],
                                         past_weather_interval,
                                         past_cases_interval)

    train_Y = train_labels_df.loc[30:, 'dailly_cases']  # 30 -192 天的dailly_cases

    model = svm_model.fit(train_X, train_Y)
    test_Y = model.predict(test_X)
    return int(test_Y)


## Project-Part2
def predict_COVID_part2(train_df, train_labels_df, test_feature):
    new_train_df = generate_new_df_by_dict(train_df)  # 生成大的train_df
    new_train_df = new_train_df.iloc[73:]
    train_X = cut_by_feature_and_interval(new_train_df, ['max_temp', 'avg_temp', 'dailly_cases'],
                                          30, 15)

    test_X_dict = dict()
    for index, item in test_feature.iteritems():
        if index != "day":
            test_X_dict[index] = [item]
    test_X_before_cut = pd.DataFrame(test_X_dict)  # 输入的是列，把它变成行
    test_X = cut_by_feature_and_interval(test_X_before_cut,
                                         ['max_temp', 'avg_temp', 'dailly_cases'],
                                         30, 15)

    train_Y = train_labels_df.loc[103:, 'dailly_cases']  # 73 -192 天的dailly_cases

    svm_model = SVR()
    svm_model.set_params(**{'kernel': "poly", 'degree': 1, 'C': 15000,
                            'gamma': 'scale', 'coef0': 0.0, 'tol': 0.001, 'epsilon': 1})

    model = svm_model.fit(train_X, train_Y)
    test_Y = model.predict(test_X)
    return int(test_Y)
