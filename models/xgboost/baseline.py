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

train["unique_characters"] = train.f_27.apply(lambda s: len(set(s)))
test["unique_characters"] = test.f_27.apply(lambda s: len(set(s)))
train.drop('f_27', inplace=True, axis=1)
test.drop('f_27', inplace=True, axis=1)

# %%
train.head()

# %% [markdown]
# # Train model

# %%
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

def fit_xgb(train, params=None):
    models = []
    valid_scores = []

    kf = KFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.seed)
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

        y_valid_pred = model.predict_proba(X=X_valid)[:, 1]
        score = roc_auc_score(y_true=y_valid, y_score=y_valid_pred)

        print(f'fold {fold} AUC: {score}')
        valid_scores.append(score)
        models.append(model)

    cv_score = np.mean(valid_scores)
    print(f'CV score: {cv_score}')
    return models

def inference_catboost(models, feat):
    pred = [model.predict_proba(feat)[:, 1] for model in models]
    pred = np.mean(pred, axis=0)
    return pred

# %%
train.drop(['id'], axis=1, inplace=True)
feat = test.drop(['id'], axis=1)

# %%
models = fit_xgb(train=train)
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


