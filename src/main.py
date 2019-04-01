import pandas as pd
from src2.feature import *
from src2.model import *

y_train = target
def get_data(data,feature,shape):
    datacopy = data.copy()
    fea_data = feature(datacopy)
    cate_columns = [i for i in fea_data.columns]
    cate_columns.remove('用户编码')
    cate_columns.remove('信用分')
    train_data = fea_data[:shape]
    test_data = fea_data[shape:]
    X_train = train_data[cate_columns].values
    X_test = test_data[cate_columns].values
    return X_train,X_test


x_train_1,x_test_1 = get_data(data,feature1,50000)
x_train_2,x_test_2 = get_data(data,feature2,50000)
x_train_3,x_test_3 = get_data(data,feature3,50000)
x_train_4,x_test_4 = get_data(data,feature4,50000)
x_train_5,x_test_5 = get_data(data,feature5,50000)
x_train_6,x_test_6 = get_data(data,feature6,50000)


lgb1_model(1,x_train_1,y_train,x_test_1,'1')
lgb2_model(1,x_train_2,y_train,x_test_2,'2')
xgb_model(1,x_train_3,y_train,x_test_3,'3')
cat_model(1,x_train_4,y_train,x_test_4,'4')
lgb3_model(1,x_train_5,y_train,x_test_5,'5')
lgb4_model(1,x_train_6,y_train,x_test_6,'6')

f.close()

