from sklearn.model_selection import cross_val_score, validation_curve, KFold
import lightgbm as lgbm
from lightgbm import early_stopping
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib.pyplot import plt

train = pd.read_csv('train.csv')


class Config:
    seed = 2020
    n_splits = 5
    target = 'target'


y = train[Config.target].values
X = train.drop(Config.target, axis=1).values

model = lgbm.LGBMClassifier(boosting_type='gbdt',
                            n_estimators=10000,
                            objective='binary',
                            random_state=Config.seed)

fit_params = {'callbacks': [early_stopping(
    stopping_rounds=10,
    verbose=0)],
    'eval_metric': 'auc',
    'eval_set': [(X, y)]}

kf = KFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.seed)

scores = cross_val_score(estimator=model,
                         X=X,
                         y=y,
                         scoring='roc_auc',
                         cv=kf,
                         n_jobs=-1,
                         fit_params=fit_params)

print(f'scores={scores}')
print(f'average_score={np.mean(scores)}')


cv_params = {'reg_alpha': [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
             'reg_lambda': [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
             'num_leaves': [2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256],
             'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
             'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
             'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
             'min_child_samples': [0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
             }

param_scales = {'reg_alpha': 'log',
                'reg_lambda': 'log',
                'num_leaves': 'linear',
                'colsample_bytree': 'linear',
                'subsample': 'linear',
                'subsample_freq': 'linear',
                'min_child_samples': 'linear'
                }

for i, (k, v) in enumerate(tqdm(cv_params.items())):
    train_scores, valid_scores = validation_curve(estimator=model,
                                                  X=X, y=y,
                                                  param_name=k,
                                                  param_range=v,
                                                  fit_params=fit_params,
                                                  cv=kf, scoring='roc_auc',
                                                  n_jobs=-1)

    # ---------- mean, std for train set ----------
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    train_center = train_mean
    train_high = train_mean + train_std
    train_low = train_mean - train_std
    # ---------- mean, std for validation set ----------
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    valid_center = valid_mean
    valid_high = valid_mean + valid_std
    valid_low = valid_mean - valid_std
    # ---------- plot train set ----------
    plt.plot(
        v,
        train_center,
        color='blue',
        marker='o',
        markersize=5,
        label='training score')
    plt.fill_between(v, train_high, train_low, alpha=0.15, color='blue')
    # ---------- plot validation set ----------
    plt.plot(
        v,
        valid_center,
        color='green',
        linestyle='--',
        marker='o',
        markersize=5,
        label='validation score')
    plt.fill_between(v, valid_high, valid_low, alpha=0.15, color='green')
    # ---------- settings ----------
    plt.xscale(param_scales[k])
    plt.xlabel(k)
    plt.ylabel('roc_auc')
    plt.legend(loc='lower right')
    plt.show()
