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
train = pd.read_csv(INPUT_DIR + 'train_small.csv')
test = pd.read_csv(INPUT_DIR + 'test.csv')

# %%
train.head()

# %%
train.info(memory_usage='deep')

# %% [markdown]
# # Preprocess

# %%
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
oe.fit(train['f_27'].values.reshape(-1,1))
train['f_27'] = oe.transform(train['f_27'].values.reshape(-1,1))
test['f_27'] = oe.transform(test['f_27'].values.reshape(-1,1))

# %% [markdown]
# # Train model

# %%
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgbm
from lightgbm import early_stopping

def fit_lgbm(train, params):
    models = []
    valid_scores = []

    kf = KFold(n_splits=Config.n_splits, shuffle=True)
    train_y = train.pop(Config.target)
    train_X = train

    for fold, (train_indices, valid_indices) in enumerate(kf.split(X=train_X, y=train_y)):
        X_train, X_valid = train_X.iloc[train_indices], train_X.iloc[valid_indices]
        y_train, y_valid = train_y.iloc[train_indices], train_y.iloc[valid_indices]

        model = lgbm.LGBMClassifier(**params,
                                    boosting_type='gbdt',
                                    objective='binary',
                                    random_state=Config.seed,
                                    n_estimators=10000,
                                    verbose=-1)
        model.fit(X_train,
                  y_train,
                  callbacks=[early_stopping(
                      stopping_rounds=10,
                      verbose=0)],
                  eval_metric='auc',
                  eval_set=[(X_valid, y_valid)])

        y_valid_pred = model.predict_proba(X=X_valid)
        y_valid_pred = np.argmax(y_valid_pred, axis=1)
        score = roc_auc_score(y_true=y_valid, y_score=y_valid_pred)

        print(f'fold {fold} AUC: {score}')
        valid_scores.append(score)
        models.append(model)

    cv_score = np.mean(valid_scores)
    print(f'CV score: {cv_score}')
    return models

def inference_lgbm(models, feat):
    pred = np.array([model.predict_proba(feat) for model in models])
    pred = np.argmax(np.mean(pred, axis=0) , axis=1)
    return pred

# %% [markdown]
# ### Optimization

# %%
from sklearn.model_selection import cross_val_score

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

# %%
# from sklearn.model_selection import validation_curve


# cv_params = {'reg_alpha': [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
#              'reg_lambda': [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
#              'num_leaves': [2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256],
#              'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#              'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
#              'subsample_freq': [0, 1, 2, 3, 4, 5, 6, 7],
#              'min_child_samples': [0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#              }

# param_scales = {'reg_alpha': 'log',
#                 'reg_lambda': 'log',
#                 'num_leaves': 'linear',
#                 'colsample_bytree': 'linear',
#                 'subsample': 'linear',
#                 'subsample_freq': 'linear',
#                 'min_child_samples': 'linear'
#                 }

# for i, (k, v) in enumerate(tqdm(cv_params.items())):
#     train_scores, valid_scores = validation_curve(estimator=model,
#                                                   X=X, y=y,
#                                                   param_name=k,
#                                                   param_range=v,
#                                                   fit_params=fit_params,
#                                                   cv=kf, scoring='roc_auc',
#                                                   n_jobs=-1)

#     # ---------- mean, std for train set ----------
#     train_mean = np.mean(train_scores, axis=1)
#     train_std  = np.std(train_scores, axis=1)
#     train_center = train_mean
#     train_high = train_mean + train_std
#     train_low = train_mean - train_std
#     # ---------- mean, std for validation set ----------
#     valid_mean = np.mean(valid_scores, axis=1)
#     valid_std  = np.std(valid_scores, axis=1)
#     valid_center = valid_mean
#     valid_high = valid_mean + valid_std
#     valid_low = valid_mean - valid_std
#     # ---------- plot train set ----------
#     plt.plot(v, train_center, color='blue', marker='o', markersize=5, label='training score')
#     plt.fill_between(v, train_high, train_low, alpha=0.15, color='blue')
#     # ---------- plot validation set ----------
#     plt.plot(v, valid_center, color='green', linestyle='--', marker='o', markersize=5, label='validation score')
#     plt.fill_between(v, valid_high, valid_low, alpha=0.15, color='green')
#     # ---------- settings ----------
#     plt.xscale(param_scales[k])
#     plt.xlabel(k)
#     plt.ylabel('roc_auc')
#     plt.legend(loc='lower right')
#     plt.show()

# %%
import optuna

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
    scores = cross_val_score(model, X, y, cv=kf,
                             scoring='roc_auc', fit_params=fit_params, n_jobs=-1)
    val = scores.mean()
    return val

study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=Config.seed))
study.optimize(bayes_objective, n_trials=100)

best_params = study.best_trial.params
best_score = study.best_trial.value
print(f'最適パラメータ {best_params}\nスコア {best_score}')

# %%
from sklearn.model_selection import learning_curve
model.set_params(**best_params)

# 学習曲線の取得
train_sizes, train_scores, valid_scores = learning_curve(estimator=model,
                                                         X=X, y=y,
                                                         fit_params=fit_params,
                                                         cv=kf, scoring='roc_auc', n_jobs=-1)

# 学習データ指標の平均±標準偏差を計算
train_mean = np.mean(train_scores, axis=1)
train_std  = np.std(train_scores, axis=1)
train_center = train_mean
train_high = train_mean + train_std
train_low = train_mean - train_std
# 検証データ指標の平均±標準偏差を計算
valid_mean = np.mean(valid_scores, axis=1)
valid_std  = np.std(valid_scores, axis=1)
valid_center = valid_mean
valid_high = valid_mean + valid_std
valid_low = valid_mean - valid_std
# training_scoresをプロット
plt.plot(train_sizes, train_center, color='blue', marker='o', markersize=5, label='training score')
plt.fill_between(train_sizes, train_high, train_low, alpha=0.15, color='blue')
# validation_scoresをプロット
plt.plot(train_sizes, valid_center, color='green', linestyle='--', marker='o', markersize=5, label='validation score')
plt.fill_between(train_sizes, valid_high, valid_low, alpha=0.15, color='green')
# 最高スコアの表示
best_score = valid_center[len(valid_center) - 1]
plt.text(np.amax(train_sizes), valid_low[len(valid_low) - 1], f'best_score={best_score}',
                color='black', verticalalignment='top', horizontalalignment='right')
# 軸ラベルおよび凡例の指定
plt.xlabel('training examples')  # 学習サンプル数を横軸ラベルに
plt.ylabel('roc_auc')  # スコア名を縦軸ラベルに
plt.legend(loc='lower right')  # 凡例

# %%
train.drop(['id'], axis=1, inplace=True)
feat = test.drop(['id'], axis=1)

# %%
params = {
    'learning_rate': 0.1,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'num_leaves': 31,
    'colsample_bytree': 1.0,
    'subsample': 1.0,
    'subsample_freq': 0,
    'min_child_samples': 20
}

models = fit_lgbm(train=train, params=params)
pred = inference_lgbm(models=models, feat=feat)

# %% [markdown]
# # Postprocess

# %%
def plot_importances(model):
    importance_df = pd.DataFrame(model.feature_importances_,
                                 index=model.feature_name_,
                                 columns=['importance'])\
                        .sort_values("importance", ascending=False)

    plt.subplots(figsize=(len(model.feature_name_) // 4, 5))
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
sub.to_csv(OUTPUT_DIR + 'lgbm_baseline.csv', index=False)


