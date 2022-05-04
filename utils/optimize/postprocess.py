import optuna
from sklearn.model_selection import KFold, learning_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgbm
from lightgbm import early_stopping

train = pd.read_csv('train.csv')


class Config:
    seed = 2020
    n_splits = 5
    target = 'target'


y = train[Config.target].values
X = train.drop(Config.target, axis=1).values
cv_params = {'reg_alpha': [0.3, 0.15, 1, 1.5, 3, 6, 10],
             'reg_lambda': [0.1, 0.2, 0.3, 0.6, 1],
             'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
             'min_child_samples': [0, 2, 5, 10]}
fit_params = {'callbacks': [early_stopping(
    stopping_rounds=10,
    verbose=0)],
    'eval_metric': 'auc',
    'eval_set': [(X, y)]}
kf = KFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.seed)
model = lgbm.LGBMClassifier(boosting_type='gbdt',
                            n_estimators=10000,
                            objective='binary',
                            random_state=Config.seed)

model.set_params(**best_params)

# 学習曲線の取得
train_sizes, train_scores, valid_scores = learning_curve(
    estimator=model, X=X, y=y, fit_params=fit_params, cv=kf, scoring='roc_auc', n_jobs=-1)

# 学習データ指標の平均±標準偏差を計算
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
train_center = train_mean
train_high = train_mean + train_std
train_low = train_mean - train_std
# 検証データ指標の平均±標準偏差を計算
valid_mean = np.mean(valid_scores, axis=1)
valid_std = np.std(valid_scores, axis=1)
valid_center = valid_mean
valid_high = valid_mean + valid_std
valid_low = valid_mean - valid_std
# training_scoresをプロット
plt.plot(
    train_sizes,
    train_center,
    color='blue',
    marker='o',
    markersize=5,
    label='training score')
plt.fill_between(train_sizes, train_high, train_low, alpha=0.15, color='blue')
# validation_scoresをプロット
plt.plot(
    train_sizes,
    valid_center,
    color='green',
    linestyle='--',
    marker='o',
    markersize=5,
    label='validation score')
plt.fill_between(train_sizes, valid_high, valid_low, alpha=0.15, color='green')
# 最高スコアの表示
best_score = valid_center[len(valid_center) - 1]
plt.text(np.amax(train_sizes),
         valid_low[len(valid_low) - 1],
         f'best_score={best_score}',
         color='black',
         verticalalignment='top',
         horizontalalignment='right')
# 軸ラベルおよび凡例の指定
plt.xlabel('training examples')  # 学習サンプル数を横軸ラベルに
plt.ylabel('roc_auc')  # スコア名を縦軸ラベルに
plt.legend(loc='lower right')  # 凡例
