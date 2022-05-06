# %%
import sys
from pathlib import Path
sys.path.append(str(Path('__file__').resolve().parent.parent.parent))

# %%
from configs.data import INPUT_DIR, OUTPUT_DIR
from configs.lightGBM import Config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
train = pd.read_csv(INPUT_DIR + 'train.csv')
test = pd.read_csv(INPUT_DIR + 'test.csv')

# %%
train.head()

# %%
train.info(memory_usage='deep')

# %% [markdown]
# # Preprocess

# %%
# From https://www.kaggle.com/ambrosm/tpsmay22-eda-which-makes-sense
for i in range(10):
    train[f'ch{i}'] = train.f_27.str.get(i).apply(ord) - ord('A')
    test[f'ch{i}'] = test.f_27.str.get(i).apply(ord) - ord('A')

train.drop('f_27', inplace=True, axis=1)
test.drop('f_27', inplace=True, axis=1)

# %%
train.head()

# %% [markdown]
# # Optimization

# %%
y = train[Config.target]
X = train.drop(Config.target, axis=1)

# %%
from sklearn.model_selection import cross_val_score
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold

model = xgb.XGBClassifier(objective='binary:logistic',
                          use_label_encoder=False)

fit_params = {'verbose': 0,
              'early_stopping_rounds': 10,
              'eval_metric': 'auc',
              'eval_set': [(X, y)]}

kf = KFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.seed)

scores = cross_val_score(model, X, y, cv=kf,
                         scoring='roc_auc', n_jobs=-1, fit_params=fit_params)
print(f'scores={scores}')
print(f'average_score={np.mean(scores)}')

# %%
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_auc_score

cv_params = {'reg_alpha': [0.01, 0.03, 0.1, 0.3, 1.0],
             'reg_lambda': [0.01, 0.03, 0.1, 0.3, 1.0],
             'learning_rate': [0.0001, 0.001, 0.01, 0.03, 0.1],}

param_scales = {'reg_alpha': 'log',
                'reg_lambda': 'log',
                'learning_rate': 'log',}

for i, (k, v) in enumerate(cv_params.items()):
    train_scores, valid_scores = validation_curve(estimator=model,
                                                  X=X, y=y,
                                                  param_name=k,
                                                  param_range=v,
                                                  fit_params=fit_params,
                                                  cv=kf, scoring='roc_auc',
                                                  n_jobs=-1)
    # 学習データに対するスコアの平均±標準偏差を算出
    train_mean = np.mean(train_scores, axis=1)
    train_std  = np.std(train_scores, axis=1)
    train_center = train_mean
    train_high = train_mean + train_std
    train_low = train_mean - train_std
    # テストデータに対するスコアの平均±標準偏差を算出
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std  = np.std(valid_scores, axis=1)
    valid_center = valid_mean
    valid_high = valid_mean + valid_std
    valid_low = valid_mean - valid_std
    # training_scoresをプロット
    plt.plot(v, train_center, color='blue', marker='o', markersize=5, label='training score')
    plt.fill_between(v, train_high, train_low, alpha=0.15, color='blue')
    # validation_scoresをプロット
    plt.plot(v, valid_center, color='green', linestyle='--', marker='o', markersize=5, label='validation score')
    plt.fill_between(v, valid_high, valid_low, alpha=0.15, color='green')
    # スケールをparam_scalesに合わせて変更
    plt.xscale(param_scales[k])
    # 軸ラベルおよび凡例の指定
    plt.xlabel(k)  # パラメータ名を横軸ラベルに
    plt.ylabel('roc_auc')  # スコア名を縦軸ラベルに
    plt.legend(loc='lower right')  # 凡例
    # グラフを描画
    plt.show()

# %%
import optuna

def bayes_objective(trial):
    y = train[Config.target].values
    X = train.drop(Config.target, axis=1).values
    kf = KFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.seed)
    model = xgb.XGBClassifier(**params,
                              objective='binary:logistic',
                              use_label_encoder=False)


    params = {
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1, log=True),
    }
    fit_params = {'verbose': 0,
              'early_stopping_rounds': 10,
              'eval_metric': 'auc',
              'eval_set': [(X, y)]}

    model.set_params(**params)
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
    sampler=optuna.samplers.TPESampler(seed=Config.seed))
study.optimize(bayes_objective, n_trials=100)

best_params = study.best_trial.params
best_score = study.best_trial.value
print(f'best parameter: {best_params}\nscore: {best_score}')

# %% [markdown]
# # Train model

# %%
def fit_xgb(train, params=None):
    models = []
    valid_scores = []

    kf = KFold(n_splits=Config.n_splits, shuffle=True)
    train_y = train.pop(Config.target)
    train_X = train

    for fold, (train_indices, valid_indices) in enumerate(kf.split(X=train_X, y=train_y)):
        X_train, X_valid = train_X.iloc[train_indices], train_X.iloc[valid_indices]
        y_train, y_valid = train_y.iloc[train_indices], train_y.iloc[valid_indices]

        model = xgb.XGBClassifier(objective='binary:logistic',
                                  use_label_encoder=False)

        model.fit(X=X_train,
                  y=y_train,
                  eval_set=[(X_valid, y_valid)],
                  eval_metric='auc',
                  early_stopping_rounds=10,
                  verbose=True)

        y_valid_pred = model.predict_proba(X=X_valid)
        y_valid_pred = np.argmax(y_valid_pred, axis=1)
        score = roc_auc_score(y_true=y_valid, y_score=y_valid_pred)

        print(f'fold {fold} AUC: {score}')
        valid_scores.append(score)
        models.append(model)

    cv_score = np.mean(valid_scores)
    print(f'CV score: {cv_score}')
    return models

def inference_xgb(models, feat):
    pred = np.array([model.predict_proba(feat) for model in models])
    pred = np.argmax(np.mean(pred, axis=0) , axis=1)
    return pred

# %%
train.drop(['id'], axis=1, inplace=True)
feat = test.drop(['id'], axis=1)

# %%
models = fit_xgb(train=train, params=best_params)
pred = inference_xgb(models=models, feat=feat)

# %% [markdown]
# # Postprocess

# %%
def plot_importances(model):
    importance_df = pd.DataFrame(model.feature_importances_,
                                 index=model.get_booster().feature_names,
                                 columns=['importance'])\
                        .sort_values("importance", ascending=False)

    plt.subplots(figsize=(len(model.get_booster().feature_names) // 4, 5))
    plt.bar(importance_df.index, importance_df.importance)
    plt.grid()
    plt.xticks(rotation=90)
    plt.ylabel("importance")
    plt.tight_layout()
    plt.show()

# %%
plot_importances(model=models[0])

# %% [markdown]
# # Submission

# %%
sub = pd.read_csv(INPUT_DIR + 'sample_submission.csv')
test[Config.target] = pred
sub = test.loc[:, ['id', Config.target]].reset_index(drop=True)
sub.to_csv(OUTPUT_DIR + 'xgb_baseline.csv', index=False)


