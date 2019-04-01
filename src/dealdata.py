import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np



train = pd.read_csv('../data/train_dataset.csv')
test = pd.read_csv('../data/test_dataset.csv')
target = train['信用分']

data1 = pd.read_csv('../result/lgb1_model_1.csv')
data2 = pd.read_csv('../result/lgb2_model_2.csv')
data3 = pd.read_csv('../result/xgb_model_3.csv')
data4 = pd.read_csv('../result/cat_model_4.csv')
data5 = pd.read_csv('../result/lgb3_model_5.csv')
data6 = pd.read_csv('../result/lgb4_model_6.csv')


predictions_blends2 = 0.17 * (data1['score2']) + 0.17 * data4['score2'] + 0.17 * data3['score2'] \
                    + 0.17 * (data6['score2']+1) + 0.15 * (data5['score2']+1) + 0.17 * (data2['score2']+1)
predictions_blends2 = predictions_blends2.apply(lambda x:int(x + 0.5))


MAE = mean_absolute_error(train['信用分'],predictions_blends2)
score = 1/(1+MAE)
print('线下测评分数： best score is %8.8f' % score)


predictions_blends = 0.17 * (data1['score']) + 0.17 * data4['score'] + 0.17 * data3['score'] \
                    + 0.17 * (data6['score']+1) + 0.15 * (data5['score']+1) + 0.17 * (data2['score']+1)

test_data_sub = test[['用户编码']]
test_data_sub['score'] = predictions_blends
test_data_sub['score'] = test_data_sub['score'].apply(lambda x:int(x+0.5))
test_data_sub.columns = ['id','score']
test_data_sub.to_csv('ronghe.csv', index=False)