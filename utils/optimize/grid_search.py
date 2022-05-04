from sklearn.model_selection import GridSearchCV
import lightgbm as lgbm
from lightgbm import early_stopping
import pandas as pd

train = pd.read_csv('train.csv')


class Config:
    seed = 2020
    n_splits = 5
    target = 'target'


y = train[Config.target].values
X = train.drop(Config.target, axis=1).values
fit_params = {'callbacks': [early_stopping(
    stopping_rounds=10,
    verbose=0)],
    'eval_metric': 'auc',
    'eval_set': [(X, y)]}

model = lgbm.LGBMClassifier(boosting_type='gbdt',
                            n_estimators=10000,
                            objective='binary',
                            random_state=Config.seed)

fit_params = {'callbacks': [early_stopping(
                  stopping_rounds=10,
                  verbose=0)],
              'eval_metric': 'auc',
              'eval_set': [(X, y)]}

cv_params = {'reg_alpha': [0.3, 1, 3, 10],
             'reg_lambda': [0.1, 0.3, 1],
             'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
             'min_child_samples': [0, 2, 5, 10]}

gridcv = GridSearchCV(estimator=model,
                      param_grid=cv_params,
                      scoring='roc_auc',
                      verbose=10,
                      n_jobs=-1)

gridcv.fit(X, y, **fit_params)
best_params = gridcv.best_params_
print(best_params)
best_score = gridcv.best_score_
print(best_score)
