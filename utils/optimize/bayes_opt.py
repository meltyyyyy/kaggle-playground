import optuna
from sklearn.model_selection import KFold, cross_val_score
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
kf = KFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.seed)
model = lgbm.LGBMClassifier(boosting_type='gbdt',
                            n_estimators=10000,
                            objective='binary',
                            random_state=Config.seed)


def bayes_objective(trial):
    params = {
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 0.1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0001, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 6),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 0, 10)
    }
    # モデルにパラメータ適用
    model.set_params(**params)
    # cross_val_scoreでクロスバリデーション
    scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring='roc_auc',
        fit_params=fit_params,
        n_jobs=-1)
    val = scores.mean()
    return val


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(
        seed=Config.seed))
study.optimize(bayes_objective, n_trials=100)

best_params = study.best_trial.params
best_score = study.best_trial.value
print(f'最適パラメータ {best_params}\nスコア {best_score}')
