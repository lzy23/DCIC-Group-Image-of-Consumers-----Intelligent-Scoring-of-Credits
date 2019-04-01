import lightgbm as lgb
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as cat
f = open('../result/result.txt','w')




def lgb1_model(num_model_seed,x_train,y_train,x_test,name):
    predictions_lgb = np.zeros(len(x_test))
    oof = np.zeros(len(x_train))
    seeds = [2019, 2019 * 2 + 1024, 4096, 2048, 1024]
    for model_seed in range(num_model_seed):
        param = {'num_leaves': 31,
                 'min_data_in_leaf': 20,
                 'objective': 'regression_l1',
                 'max_depth': 5,
                 'learning_rate': 0.0081,
                 "min_child_samples": 30,
                 "boosting": "gbdt",
                 "feature_fraction": 0.7,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.8,
                 "bagging_seed": 11,
                 "metric": 'mae',
                 "lambda_l1": 0.60,
                 "verbosity": -1}
        folds = KFold(n_splits=6, shuffle=True, random_state=seeds[0])
        oof_lgb = np.zeros(len(x_train))

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
            print(len(trn_idx))
            print("fold n°{}".format(fold_ + 1))
            trn_data = lgb.Dataset(x_train[trn_idx], y_train[trn_idx])
            val_data = lgb.Dataset(x_train[val_idx], y_train[val_idx])
            num_round = 10000
            clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                            early_stopping_rounds=500)
            oof_lgb[val_idx] = clf.predict(x_train[val_idx], num_iteration=clf.best_iteration)
            predictions_lgb += clf.predict(x_test, num_iteration=clf.best_iteration) / folds.n_splits / num_model_seed
        oof += oof_lgb / num_model_seed
        MAE = mean_absolute_error(oof_lgb, y_train)
        score = 1 / (1 + MAE)
        print("CV score: {:<8.8f}".format(MAE))
        print("score: {:<8.8f}".format(score))
    MAE = mean_absolute_error(oof, y_train)
    score = 1 / (1 + MAE)
    print("CV score: {:<8.8f}".format(MAE))
    print("score: {:<8.8f}".format(score))
    sub_df = pd.read_csv('../data/submit_example.csv')
    sub_df['score'] = predictions_lgb
    sub_df['score2'] = oof_lgb
    sub_df.to_csv('../result/lgb1_model_{}.csv'.format(name), index=0, header=1, sep=',')
    f.write("lgb1_model_{}---score: {:<8.8f}".format(name,score))
    f.write('\n')

def lgb2_model(num_model_seed,x_train,y_train,x_test,name):
    predictions_lgb = np.zeros(len(x_test))
    oof = np.zeros(len(x_train))
    seeds = [2019, 2019 * 2 + 1024, 4096, 2048, 1024]
    for model_seed in range(num_model_seed):
        param = {'num_leaves': 31,
                 'min_data_in_leaf': 20,
                 'objective': 'regression_l2',
                 'max_depth': 5,
                 'learning_rate': 0.0081,
                 "min_child_samples": 30,
                 "boosting": "gbdt",
                 "feature_fraction": 0.7,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.8,
                 "bagging_seed": 11,
                 "metric": 'mae',
                 "lambda_l1": 0.60,
                 "verbosity": -1}
        folds = KFold(n_splits=6, shuffle=True, random_state=seeds[0])  #
        oof_lgb = np.zeros(len(x_train))
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
            print(len(trn_idx))
            print("fold n°{}".format(fold_ + 1))
            trn_data = lgb.Dataset(x_train[trn_idx], y_train[trn_idx])
            val_data = lgb.Dataset(x_train[val_idx], y_train[val_idx])
            num_round = 10000
            clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                            early_stopping_rounds=500)
            oof_lgb[val_idx] = clf.predict(x_train[val_idx], num_iteration=clf.best_iteration)
            predictions_lgb += clf.predict(x_test, num_iteration=clf.best_iteration) / folds.n_splits / num_model_seed
        oof += oof_lgb / num_model_seed
        MAE = mean_absolute_error(oof_lgb, y_train)
        score = 1 / (1 + MAE)
        print("CV score: {:<8.8f}".format(MAE))
        print("score: {:<8.8f}".format(score))
    MAE = mean_absolute_error(oof, y_train)
    score = 1 / (1 + MAE)
    print("CV score: {:<8.8f}".format(MAE))
    print("score: {:<8.8f}".format(score))

    sub_df = pd.read_csv('../data/submit_example.csv')
    sub_df['score'] = predictions_lgb
    sub_df['score2'] = oof_lgb
    sub_df.to_csv('../result/lgb2_model_{}.csv'.format(name), index=0, header=1, sep=',')
    f.write("lgb2_model_{}---score: {:<8.8f}".format(name, score))
    f.write('\n')

def xgb_model(num_model_seed,x_train,y_train,x_test,name):
    predictions_xgb = np.zeros(len(x_test))
    oof_xgb = np.zeros(len(x_train))
    seeds = [2019, 4096, 2019 * 2 + 1024, 2048, 1024]
    for seed in range(num_model_seed):
        xgb_params = {'eta': 0.004, 'max_depth': 6, 'subsample': 0.5, 'colsample_bytree': 0.5, 'alpha': 0.2,
                      'objective': 'reg:gamma', 'eval_metric': 'mae', 'silent': True, 'nthread': -1
                      }
        folds = KFold(n_splits=5, shuffle=True, random_state=seeds[seed])
        oof = np.zeros(len(x_train))
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
            print("fold n°{}".format(fold_ + 1))
            trn_data = xgb.DMatrix(x_train[trn_idx], y_train[trn_idx])
            val_data = xgb.DMatrix(x_train[val_idx], y_train[val_idx])

            watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
            clf = xgb.train(dtrain=trn_data, num_boost_round=10000, evals=watchlist, early_stopping_rounds=200,
                            verbose_eval=1000, params=xgb_params)
            oof[val_idx] = clf.predict(xgb.DMatrix(x_train[val_idx]), ntree_limit=clf.best_ntree_limit)
            predictions_xgb += clf.predict(xgb.DMatrix(x_test),
                                           ntree_limit=clf.best_ntree_limit) / folds.n_splits / num_model_seed
        oof_xgb += oof / num_model_seed
        MAE = mean_absolute_error(y_train, oof)
        score = 1 / (1 + MAE)
        print("CV score: {:<8.8f}".format(MAE))
        print("score: {:<8.8f}".format(score))

    MAE = mean_absolute_error(oof_xgb, y_train)
    score = 1 / (1 + MAE)
    print("CV score: {:<8.8f}".format(MAE))
    print("score: {:<8.8f}".format(score))

    sub_df = pd.read_csv('../data/submit_example.csv')
    sub_df['score'] = predictions_xgb
    sub_df['score2'] = oof_xgb
    sub_df.to_csv('../result/xgb_model_{}.csv'.format(name), index=0, header=1, sep=',')
    f.write("xgb_model_{}---score: {:<8.8f}".format(name, score))
    f.write('\n')

def cat_model(num_model_seed,x_train,y_train,x_test,name):
    cat_params = {'depth': 7, 'learning_rate': 0.8, 'l2_leaf_reg': 2, 'num_boost_round': 10000, 'random_seed': 94,
                  'loss_function': 'MAE'}

    seeds = [2019, 2019 * 2 + 1024, 4096, 2048, 1024]
    oof_cat = np.zeros(len(x_train))
    predictions_cat = np.zeros(len(x_test))
    for seed in range(num_model_seed):
        folds = KFold(n_splits=5, shuffle=True, random_state=seeds[seed])
        oof = np.zeros(len(x_train))
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
            print("fold n°{}".format(fold_ + 1))
            clf = cat.CatBoostRegressor(**cat_params)
            clf.fit(x_train[trn_idx], y_train[trn_idx], early_stopping_rounds=200, verbose_eval=3000,
                    use_best_model=True,
                    eval_set=(x_train[val_idx], y_train[val_idx]))
            oof[val_idx] = clf.predict(x_train[val_idx])
            predictions_cat += clf.predict(x_test) / folds.n_splits / num_model_seed
        oof_cat += oof / num_model_seed
        MAE = mean_absolute_error(oof, y_train)
        score = 1 / (1 + MAE)
        print("CV score: {:<8.8f}".format(MAE))
        print("score: {:<8.8f}".format(score))

    MAE = mean_absolute_error(oof_cat, y_train)
    score = 1 / (1 + MAE)
    print("CV score: {:<8.8f}".format(MAE))
    print("score: {:<8.8f}".format(score))

    sub_df = pd.read_csv('../data/submit_example.csv')
    sub_df['score'] = predictions_cat
    sub_df['score2'] = oof_cat
    sub_df.to_csv('../result/cat_model_{}.csv'.format(name), index=0, header=1, sep=',')
    f.write("cat_model_{}---score: {:<8.8f}".format(name, score))
    f.write('\n')


def lgb3_model(num_model_seed,x_train,y_train,x_test,name):
    predictions_lgb = np.zeros(len(x_test))
    oof = np.zeros(len(x_train))
    seeds = [2019, 2019 * 2 + 1024, 4096, 2048, 1024]
    for model_seed in range(num_model_seed):
        param = {'num_leaves': 31,
                 'min_data_in_leaf': 20,
                 'objective': 'regression_l1',
                 'max_depth': 5,
                 'learning_rate': 0.01,
                 "min_child_samples": 30,
                 "boosting": "gbdt",
                 "feature_fraction": 0.45,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.8,
                 "bagging_seed": 11,
                 "metric": 'mae',
                 "lambda_l1": 0.60,
                 "verbosity": -1}
        folds = KFold(n_splits=6, shuffle=True, random_state=seeds[0])  #
        oof_lgb = np.zeros(len(x_train))

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
            print(len(trn_idx))
            print("fold n°{}".format(fold_ + 1))
            trn_data = lgb.Dataset(x_train[trn_idx], y_train[trn_idx])
            val_data = lgb.Dataset(x_train[val_idx], y_train[val_idx])
            num_round = 10000
            clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                            early_stopping_rounds=500)
            oof_lgb[val_idx] = clf.predict(x_train[val_idx], num_iteration=clf.best_iteration)
            predictions_lgb += clf.predict(x_test, num_iteration=clf.best_iteration) / folds.n_splits / num_model_seed

        oof += oof_lgb / num_model_seed
        MAE = mean_absolute_error(oof_lgb, y_train)
        score = 1 / (1 + MAE)
        print("CV score: {:<8.8f}".format(MAE))
        print("score: {:<8.8f}".format(score))
    MAE = mean_absolute_error(oof, y_train)
    score = 1 / (1 + MAE)
    print("CV score: {:<8.8f}".format(MAE))
    print("score: {:<8.8f}".format(score))

    sub_df = pd.read_csv('../data/submit_example.csv')
    sub_df['score'] = predictions_lgb
    sub_df['score2'] = oof_lgb
    sub_df.to_csv('../result/lgb3_model_{}.csv'.format(name), index=False)
    f.write("lgb3_model_{}---score: {:<8.8f}".format(name, score))
    f.write('\n')

def lgb4_model(num_model_seed,x_train,y_train,x_test,name):
    predictions_lgb = np.zeros(len(x_test))
    oof = np.zeros(len(x_train))
    seeds = [2018, 2019 * 2 + 1024, 4096, 2048, 1024]
    for model_seed in range(num_model_seed):
        param = {'num_leaves': 31,
                 'min_data_in_leaf': 20,
                 'objective': 'regression_l2',
                 'max_depth': 5,
                 'learning_rate': 0.01,
                 "min_child_samples": 30,
                 "boosting": "gbdt",
                 "feature_fraction": 0.45,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.8,
                 "bagging_seed": 11,
                 "metric": 'mae',
                 "lambda_l1": 0.60,
                 "verbosity": -1}
        folds = KFold(n_splits=6, shuffle=True, random_state=seeds[0])
        oof_lgb = np.zeros(len(x_train))
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
            print(len(trn_idx))
            print("fold n°{}".format(fold_ + 1))
            trn_data = lgb.Dataset(x_train[trn_idx], y_train[trn_idx])
            val_data = lgb.Dataset(x_train[val_idx], y_train[val_idx])
            num_round = 10000
            clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=1000,
                            early_stopping_rounds=500)
            oof_lgb[val_idx] = clf.predict(x_train[val_idx], num_iteration=clf.best_iteration)
            predictions_lgb += clf.predict(x_test, num_iteration=clf.best_iteration) / folds.n_splits / num_model_seed

        oof += oof_lgb / num_model_seed
        MAE = mean_absolute_error(oof_lgb, y_train)
        score = 1 / (1 + MAE)
        print("CV score: {:<8.8f}".format(MAE))
        print("score: {:<8.8f}".format(score))
    MAE = mean_absolute_error(oof, y_train)
    score = 1 / (1 + MAE)
    print("CV score: {:<8.8f}".format(MAE))
    print("score: {:<8.8f}".format(score))

    sub_df = pd.read_csv('../data/submit_example.csv')
    sub_df['score'] = predictions_lgb
    sub_df['score2'] = oof_lgb
    sub_df.to_csv('../result/lgb4_model_{}.csv'.format(name), index=False)
    f.write("lgb4_model_{}---score: {:<8.8f}".format(name, score))
    f.write('\n')