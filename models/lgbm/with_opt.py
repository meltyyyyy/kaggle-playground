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
train.drop('id', inplace=True, axis=1)

# %%
# From https://www.kaggle.com/ambrosm/tpsmay22-eda-which-makes-sense
for i in range(10):
    train[f'ch{i}'] = train.f_27.str.get(i).apply(ord) - ord('A')
    test[f'ch{i}'] = test.f_27.str.get(i).apply(ord) - ord('A')

train.drop('f_27', inplace=True, axis=1)
test.drop('f_27', inplace=True, axis=1)

# %% [markdown]
# # Optimize

# %%
import optuna
from sklearn.model_selection import KFold, cross_val_score
import lightgbm as lgbm
from lightgbm import early_stopping

def bayes_objective(trial):
    y = train[Config.target].values
    X = train.drop(Config.target, axis=1).values
    kf = KFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.seed)
    model = lgbm.LGBMClassifier(boosting_type='gbdt',
                            n_estimators=10000,
                            objective='binary',
                            random_state=Config.seed)

    fit_params = {'callbacks': [
        early_stopping(
            stopping_rounds=10,
            verbose=0)],
                  'eval_metric': 'auc',
                  'eval_set': [(X, y)]}
    params = {
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 16),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
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

# %%
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(
        seed=Config.seed))
study.optimize(bayes_objective, n_trials=100)

best_params = study.best_trial.params
best_score = study.best_trial.value
print(f'best parameter: {best_params}\nscore: {best_score}')

# %% [markdown]
# # Train model

# %%
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgbm

def fit_lgbm(train, params, fit_params):
    models = []
    valid_scores = []

    kf = KFold(n_splits=Config.n_splits)
    train_y = train.pop(Config.target)
    train_X = train

    for fold, (train_indices, valid_indices) in enumerate(kf.split(X=train_X, y=train_y)):
        X_train, X_valid = train_X.iloc[train_indices], train_X.iloc[valid_indices]
        y_train, y_valid = train_y.iloc[train_indices], train_y.iloc[valid_indices]

        model = lgbm.LGBMClassifier(**params,
                                    boosting_type='gbdt',
                                    objective='binary',
                                    n_estimators=10000,
                                    random_state=Config.seed,
                                    verbose=-1)
        fit_params = {'callbacks': [
            early_stopping(stopping_rounds=10,
                           verbose=0)],
                      'eval_metric': 'auc',
                      'eval_set': [(X_valid, y_valid)]}

        model.set_params(**fit_params)
        model.fit(X_train, y_train)

        y_valid_pred = model.predict_proba(X=X_valid)
        y_valid_pred = np.argmax(y_valid_pred, axis=1)
        score = roc_auc_score(y_true=y_valid, y_score=y_valid_pred)

        print(f'fold {fold} RMSE: {score}')
        valid_scores.append(score)
        models.append(model)

    cv_score = np.mean(valid_scores)
    print(f'CV score: {cv_score}')
    return models

def inference_lgbm(models, feat):
    pred = np.array([model.predict_proba(feat) for model in models])
    pred = np.argmax(np.mean(pred, axis=0) , axis=1)
    return pred

# %%
feat = test.drop('id', axis=1)
models = fit_lgbm(train=train, params=best_params)
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
sub.to_csv(OUTPUT_DIR + 'lgbm_opt.csv', index=False)


