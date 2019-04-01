import numpy as np
import pandas as pd
import math
from src2.model import *


train = pd.read_csv('../data/train_dataset.csv')
test = pd.read_csv('../data/test_dataset.csv')
target = train['信用分']
data = pd.concat([train,test],axis=0,ignore_index=True)
data = data.fillna(0)

####处理数据
data.loc[data['用户年龄']==0, '用户年龄'] = None
data.loc[data['用户话费敏感度']==0,'用户话费敏感度'] = data['用户话费敏感度'].median()


def feature_count(data, features=[]):
    if len(set(features)) != len(features):
        print('equal feature !!!!')
        return data
    new_feature = 'count'
    for i in features:
        new_feature += '_' + i.replace('add_', '')
    temp = data.groupby(features).size().reset_index().rename(columns={0: new_feature})
    data = data.merge(temp, 'left', on=features)
    return data

def endwords(x):
    va = str(x).split(".")
    x = int(va[-1])
    return x

def trans_set(x, countdict):
    if x == 0:
        return 0
    elif countdict[x] > 1000:
        return 1
    else:
        return 2

def feature1(data):
    data['charge_type'] = 0
    data['charge_type'][(data['缴费用户最近一次缴费金额（元）'] % 10 == 0) & \
                       data['缴费用户最近一次缴费金额（元）'] != 0] = 1
    data['buy_rate'] = data['当月物流快递类应用使用次数'] / (data['当月网购类应用使用次数'] + 1)
    data['sixfee_nowfee'] = data['用户近6个月平均消费值（元）'] - data['用户账单当月总费用（元）']
    data['fivefee_nowfee'] = data['用户近6个月平均消费值（元）'] * 6 - data['用户账单当月总费用（元）']
    data['month'] = data['用户网龄（月）'].apply(lambda x: x % 12)
    data['year'] = data['用户网龄（月）'].apply(lambda x: x / 12)
    data['word1'] = data['用户账单当月总费用（元）'].apply(lambda x: endwords(x))

    features = ['缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）', '用户账单当月总费用（元）']
    for feature in features:
        data = feature_count(data, [feature])
    data['current_fee_stability'] = \
        data['用户账单当月总费用（元）'] / (data['用户近6个月平均消费值（元）'] + 5)

    data['use_left_rate'] = data['用户账单当月总费用（元）'] / (data['用户当月账户余额（元）'] + 5)

    data['payment_rate'] = data['用户账单当月总费用（元）'] / (data['缴费用户最近一次缴费金额（元）'] + 5)

    data['balance_6month_rate'] = data['用户当月账户余额（元）'] / (data['用户近6个月平均消费值（元）'] + 5)

    data['payment_6month_rate'] = data['缴费用户最近一次缴费金额（元）'] / (data['用户近6个月平均消费值（元）'] + 5)

    data['balance_payment_rate'] = data['用户当月账户余额（元）'] / (data['缴费用户最近一次缴费金额（元）'] + 5)

    data['cosume_fee'] = data['用户账单当月总费用（元）'] - data['用户近6个月平均消费值（元）']

    data['当月金融理财类应用使用总次数/all'] = \
        data['当月金融理财类应用使用总次数'] / (data['当月网购类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                                  data['当月旅游资讯类应用使用次数'] + data['当月飞机类应用使用次数'] + \
                                  data['近三个月月均商场出现次数'] / 3 + data['当月火车类应用使用次数'] + \
                                  data['当月视频播放类应用使用次数'])  # 还ok点

    data['当月视频播放类应用使用次数/all'] = \
        data['当月视频播放类应用使用次数'] / (data['当月网购类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                                 data['当月旅游资讯类应用使用次数'] + data['当月飞机类应用使用次数'] + \
                                 data['近三个月月均商场出现次数'] / 3 + data['当月火车类应用使用次数'] + \
                                 data['当月金融理财类应用使用总次数'])

    data['当月网购类应用使用次数/all'] = \
        data['当月网购类应用使用次数'] / (data['当月视频播放类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                               data['当月旅游资讯类应用使用次数'] + data['当月飞机类应用使用次数'] + \
                               data['近三个月月均商场出现次数'] / 3 + data['当月火车类应用使用次数'] + \
                               data['当月金融理财类应用使用总次数'])

    data['当月飞机类应用使用次数'] = \
        data['当月飞机类应用使用次数'] / (data['当月视频播放类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                               data['当月网购类应用使用次数'] + data['当月旅游资讯类应用使用次数'] + \
                               data['近三个月月均商场出现次数'] / 3 + data['当月火车类应用使用次数'] + \
                               data['当月金融理财类应用使用总次数'])

    data['total-six'] = data['用户账单当月总费用（元）'] / (data['用户近6个月平均消费值（元）'])
    data['total-curr'] = (data['用户账单当月总费用（元）'] - data['用户当月账户余额（元）'])
    data['total-curr'] = data.apply(lambda x: x['total-curr'] * x['用户话费敏感度'] if x['total-curr'] > 0 else
    x['total-curr'] * (6 - x['用户话费敏感度']), axis=1)
    data['cz_times'] = data['用户账单当月总费用（元）'] / (data['缴费用户最近一次缴费金额（元）'] + 1)
    data['last_five_fee'] = 6 * data['用户近6个月平均消费值（元）'] - data['用户账单当月总费用（元）']
    data['now-five'] = data['用户账单当月总费用（元）'] - data['last_five_fee'] / 5
    data.drop(columns=['是否大学生客户','是否黑名单客户','当月是否到过福州山姆会员店','当月是否逛过福州仓山万达','是否经常逛商场的人'],inplace=True)
    return data

def feature2(data):
    data['charge_type'] = 0
    data['charge_type'][(data['缴费用户最近一次缴费金额（元）'] % 10 == 0) & \
                       data['缴费用户最近一次缴费金额（元）'] != 0] = 1
    data['buy_rate'] = data['当月物流快递类应用使用次数'] / (data['当月网购类应用使用次数'] + 1)
    data['sixfee_nowfee'] = data['用户近6个月平均消费值（元）'] - data['用户账单当月总费用（元）']
    data['fivefee_nowfee'] = data['用户近6个月平均消费值（元）'] * 6 - data['用户账单当月总费用（元）']
    data['month'] = data['用户网龄（月）'].apply(lambda x: x % 12)
    data['year'] = data['用户网龄（月）'].apply(lambda x: x / 12)
    data['rate'] = data['缴费用户最近一次缴费金额（元）'].apply(
        lambda x: float('%.3f' % (x / math.ceil(x))) if int(x) != 0 else 0)
    features = ['缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）', '用户账单当月总费用（元）']
    for feature in features:
        data = feature_count(data, [feature])
    data['current_fee_stability'] = \
        data['用户账单当月总费用（元）'] / (data['用户近6个月平均消费值（元）'] + 5)

    data['use_left_rate'] = data['用户账单当月总费用（元）'] / (data['用户当月账户余额（元）'] + 5)

    data['payment_rate'] = data['用户账单当月总费用（元）'] / (data['缴费用户最近一次缴费金额（元）'] + 5)

    data['balance_6month_rate'] = data['用户当月账户余额（元）'] / (data['用户近6个月平均消费值（元）'] + 5)

    data['payment_6month_rate'] = data['缴费用户最近一次缴费金额（元）'] / (data['用户近6个月平均消费值（元）'] + 5)

    data['balance_payment_rate'] = data['用户当月账户余额（元）'] / (data['缴费用户最近一次缴费金额（元）'] + 5)

    data['cosume_fee'] = data['用户账单当月总费用（元）'] - data['用户近6个月平均消费值（元）']

    data['当月金融理财类应用使用总次数/all'] = \
        data['当月金融理财类应用使用总次数'] / (data['当月网购类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                                  data['当月旅游资讯类应用使用次数'] + data['当月飞机类应用使用次数'] + \
                                  data['近三个月月均商场出现次数'] / 3 + data['当月火车类应用使用次数'] + \
                                  data['当月视频播放类应用使用次数'])  # 还ok点

    data['当月视频播放类应用使用次数/all'] = \
        data['当月视频播放类应用使用次数'] / (data['当月网购类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                                 data['当月旅游资讯类应用使用次数'] + data['当月飞机类应用使用次数'] + \
                                 data['近三个月月均商场出现次数'] / 3 + data['当月火车类应用使用次数'] + \
                                 data['当月金融理财类应用使用总次数'])

    data['当月网购类应用使用次数/all'] = \
        data['当月网购类应用使用次数'] / (data['当月视频播放类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                               data['当月旅游资讯类应用使用次数'] + data['当月飞机类应用使用次数'] + \
                               data['近三个月月均商场出现次数'] / 3 + data['当月火车类应用使用次数'] + \
                               data['当月金融理财类应用使用总次数'])

    data['当月飞机类应用使用次数'] = \
        data['当月飞机类应用使用次数'] / (data['当月视频播放类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                               data['当月网购类应用使用次数'] + data['当月旅游资讯类应用使用次数'] + \
                               data['近三个月月均商场出现次数'] / 3 + data['当月火车类应用使用次数'] + \
                               data['当月金融理财类应用使用总次数'])

    data['total-six'] = data['用户账单当月总费用（元）'] / (data['用户近6个月平均消费值（元）'])
    data['total-curr'] = (data['用户账单当月总费用（元）'] - data['用户当月账户余额（元）'])
    data['total-curr'] = data.apply(lambda x: x['total-curr'] * x['用户话费敏感度'] if x['total-curr'] > 0 else
    x['total-curr'] * (6 - x['用户话费敏感度']), axis=1)
    data['cz_times'] = data['用户账单当月总费用（元）'] / (data['缴费用户最近一次缴费金额（元）'] + 1)
    data['last_five_fee'] = 6 * data['用户近6个月平均消费值（元）'] - data['用户账单当月总费用（元）']
    data['now-five'] = data['用户账单当月总费用（元）'] - data['last_five_fee'] / 5
    data.drop(columns=['是否大学生客户', '是否黑名单客户', '当月是否到过福州山姆会员店', '当月是否逛过福州仓山万达', '是否经常逛商场的人'], inplace=True)
    return data

def feature3(data):
    data['充值途径'] = 0
    data['充值途径'][(data['缴费用户最近一次缴费金额（元）'] % 10 == 0) & \
                       data['缴费用户最近一次缴费金额（元）'] != 0] = 1
    data['buy_rate'] = data['当月物流快递类应用使用次数'] / (data['当月网购类应用使用次数'] + 1)
    data['sixfee_nowfee'] = data['用户近6个月平均消费值（元）'] - data['用户账单当月总费用（元）']
    data['fivefee_nowfee'] = data['用户近6个月平均消费值（元）'] * 6 - data['用户账单当月总费用（元）']
    data['month'] = data['用户网龄（月）'].apply(lambda x: x % 12)
    data['year'] = data['用户网龄（月）'].apply(lambda x: x / 12)
    data = feature_count(data, ['缴费用户最近一次缴费金额（元）'])
    data = feature_count(data, ['用户近6个月平均消费值（元）'])
    data = feature_count(data, ['用户账单当月总费用（元）'])
    data.drop(columns=['是否大学生客户', '是否黑名单客户', '是否经常逛商场的人'], inplace=True)
    return data

def feature4(data):
    data['充值途径'] = 0
    data['充值途径'][(data['缴费用户最近一次缴费金额（元）'] % 10 == 0) & \
                       data['缴费用户最近一次缴费金额（元）'] != 0] = 1
    data['buy_rate'] = data['当月物流快递类应用使用次数'] / (data['当月网购类应用使用次数'] + 1)
    data['sixfee_nowfee'] = data['用户近6个月平均消费值（元）'] - data['用户账单当月总费用（元）']
    data['fivefee_nowfee'] = data['用户近6个月平均消费值（元）'] * 6 - data['用户账单当月总费用（元）']
    data['month'] = data['用户网龄（月）'].apply(lambda x: x % 12)
    data['year'] = data['用户网龄（月）'].apply(lambda x: x / 12)
    data['word1'] = data['用户账单当月总费用（元）'].apply(lambda x: endwords(x))
    features = ['缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）', '用户账单当月总费用（元）']
    for feature in features:
        data = feature_count(data, [feature])

    data['current_fee_stability'] = \
        data['用户账单当月总费用（元）'] / (data['用户近6个月平均消费值（元）'] + 5)

    data['use_left_rate'] = data['用户账单当月总费用（元）'] / (data['用户当月账户余额（元）'] + 5)

    data['payment_rate'] = data['用户账单当月总费用（元）'] / (data['缴费用户最近一次缴费金额（元）'] + 5)

    data['balance_6month_rate'] = data['用户当月账户余额（元）'] / (data['用户近6个月平均消费值（元）'] + 5)

    data['payment_6month_rate'] = data['缴费用户最近一次缴费金额（元）'] / (data['用户近6个月平均消费值（元）'] + 5)

    data['balance_payment_rate'] = data['用户当月账户余额（元）'] / (data['缴费用户最近一次缴费金额（元）'] + 5)

    data['当月金融理财类应用使用总次数/all'] = \
        data['当月金融理财类应用使用总次数'] / (data['当月网购类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                                  data['当月旅游资讯类应用使用次数'] + data['当月飞机类应用使用次数'] + \
                                  data['近三个月月均商场出现次数'] / 3 + data['当月火车类应用使用次数'] + \
                                  data['当月视频播放类应用使用次数'])  # 还ok点

    data['当月视频播放类应用使用次数/all'] = \
        data['当月视频播放类应用使用次数'] / (data['当月网购类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                                 data['当月旅游资讯类应用使用次数'] + data['当月飞机类应用使用次数'] + \
                                 data['近三个月月均商场出现次数'] / 3 + data['当月火车类应用使用次数'] + \
                                 data['当月金融理财类应用使用总次数'])

    data['当月网购类应用使用次数/all'] = \
        data['当月网购类应用使用次数'] / (data['当月视频播放类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                               data['当月旅游资讯类应用使用次数'] + data['当月飞机类应用使用次数'] + \
                               data['近三个月月均商场出现次数'] / 3 + data['当月火车类应用使用次数'] + \
                               data['当月金融理财类应用使用总次数'])

    data['当月飞机类应用使用次数'] = \
        data['当月飞机类应用使用次数'] / (data['当月视频播放类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                               data['当月网购类应用使用次数'] + data['当月旅游资讯类应用使用次数'] + \
                               data['近三个月月均商场出现次数'] / 3 + data['当月火车类应用使用次数'] + \
                               data['当月金融理财类应用使用总次数'])
    data.drop(columns=['是否大学生客户', '是否黑名单客户', '当月是否到过福州山姆会员店', '当月是否逛过福州仓山万达', '是否经常逛商场的人'], inplace=True)
    return data

def feature5(data):
    data['充值途径'] = 0
    data['充值途径'][(data['缴费用户最近一次缴费金额（元）'] % 10 == 0) & \
                       data['缴费用户最近一次缴费金额（元）'] != 0] = 1
    data['buy_rate'] = data['当月物流快递类应用使用次数'] / (data['当月网购类应用使用次数'] + 1)
    data['sixfee_nowfee'] = data.apply(lambda x: x['用户近6个月平均消费值（元）'] - x['用户账单当月总费用（元）'], axis=1)
    data['cz_times'] = (data['用户账单当月总费用（元）']) / (data['缴费用户最近一次缴费金额（元）'] + 1)
    data['fivefee_nowfee'] = data['用户近6个月平均消费值（元）'] * 6 - data['用户账单当月总费用（元）']
    data['month'] = data['用户网龄（月）'].apply(lambda x: x % 12)
    data['year'] = data['用户网龄（月）'].apply(lambda x: x / 12)

    data['word1'] = data['用户账单当月总费用（元）'].apply(lambda x: endwords(x))
    data['word2'] = data['缴费用户最近一次缴费金额（元）'].apply(lambda x: endwords(x))
    data['word3'] = data['用户近6个月平均消费值（元）'].apply(lambda x: endwords(x))
    features = ['缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）', '用户账单当月总费用（元）']
    data = feature_count(data, features)
    countdict = dict(data['用户账单当月总费用（元）'].value_counts())
    data['set_fee'] = data['用户账单当月总费用（元）'].apply(lambda x: trans_set(x, countdict))
    data['shopping_level'] = data['当月是否逛过福州仓山万达'] + data['当月是否到过福州山姆会员店'] + data['当月是否看电影'] + data['当月是否景点游览'] + data[
        '当月是否体育场馆消费']
    data['当月网购类应用使用次数/all'] = \
        data['当月网购类应用使用次数'] / (data['当月视频播放类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                               data['当月旅游资讯类应用使用次数'] + data['当月飞机类应用使用次数'] + data['当月火车类应用使用次数'] + \
                               data['当月金融理财类应用使用总次数'])

    data['当月交通类应用使用次数/all'] = \
        (data['当月飞机类应用使用次数'] + data['当月火车类应用使用次数']) / (data['当月视频播放类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                                                       data['当月网购类应用使用次数'] + data['当月旅游资讯类应用使用次数'] +
                                                       data['当月金融理财类应用使用总次数'])
    data.drop(columns=['是否大学生客户', '是否黑名单客户', '当月是否到过福州山姆会员店', '当月是否逛过福州仓山万达', '是否经常逛商场的人'], inplace=True)
    return data

def feature6(data):
    data['充值途径'] = 0
    data['充值途径'][(data['缴费用户最近一次缴费金额（元）'] % 10 == 0) & \
                       data['缴费用户最近一次缴费金额（元）'] != 0] = 1
    data['buy_rate'] = data['当月物流快递类应用使用次数'] / (data['当月网购类应用使用次数'] + 1)
    data['sixfee_nowfee'] = data.apply(lambda x: x['用户近6个月平均消费值（元）'] - x['用户账单当月总费用（元）'], axis=1)
    data['cz_times'] = (data['用户账单当月总费用（元）']) / (data['缴费用户最近一次缴费金额（元）'] + 1)
    data['fivefee_nowfee'] = data['用户近6个月平均消费值（元）'] * 6 - data['用户账单当月总费用（元）']
    data['month'] = data['用户网龄（月）'].apply(lambda x: x % 12)
    data['year'] = data['用户网龄（月）'].apply(lambda x: x / 12)
    data['word1'] = data['用户账单当月总费用（元）'].apply(lambda x: endwords(x))
    data['word2'] = data['缴费用户最近一次缴费金额（元）'].apply(lambda x: endwords(x))
    data['word3'] = data['用户近6个月平均消费值（元）'].apply(lambda x: endwords(x))
    features = ['缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）', '用户账单当月总费用（元）']
    data = feature_count(data, features)
    countdict = dict(data['用户账单当月总费用（元）'].value_counts())
    data['set_fee'] = data['用户账单当月总费用（元）'].apply(lambda x: trans_set(x, countdict))
    data['shopping_level'] = data['当月是否逛过福州仓山万达'] + data['当月是否到过福州山姆会员店'] + data['当月是否看电影'] + data['当月是否景点游览'] + data[
        '当月是否体育场馆消费']
    data['当月网购类应用使用次数/all'] = \
        data['当月网购类应用使用次数'] / (data['当月视频播放类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                               data['当月旅游资讯类应用使用次数'] + data['当月飞机类应用使用次数'] + data['当月火车类应用使用次数'] + \
                               data['当月金融理财类应用使用总次数'])
    data['当月交通类应用使用次数/all'] = \
        (data['当月飞机类应用使用次数'] + data['当月火车类应用使用次数']) / (data['当月视频播放类应用使用次数'] + data['当月物流快递类应用使用次数'] + \
                                                       data['当月网购类应用使用次数'] + data['当月旅游资讯类应用使用次数'] +
                                                       data['当月金融理财类应用使用总次数'])
    data.drop(columns=['是否大学生客户', '是否黑名单客户', '当月是否到过福州山姆会员店', '当月是否逛过福州仓山万达', '是否经常逛商场的人'], inplace=True)
    return data
