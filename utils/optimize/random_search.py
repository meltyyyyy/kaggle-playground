from sklearn.model_selection import RandomizedSearchCV, KFold
import lightgbm as lgbm
import pandas as pd
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
model = lgbm.LGBMClassifier(boosting_type='gbdt',
                            n_estimators=10000,
                            objective='binary',
                            random_state=Config.seed)
kf = KFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.seed)

randcv = RandomizedSearchCV(estimator=model,
                            cv=kf,
                            param_distributions=cv_params,
                            random_state=Config.seed,
                            n_iter=1120,
                            scoring='roc_auc',
                            verbose=10,
                            n_jobs=-1)

randcv.fit(X, y, **fit_params)
best_params = randcv.best_params_
print(best_params)
best_score = randcv.best_score_
print(best_score)
