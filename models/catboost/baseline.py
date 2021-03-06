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
char_features = []
for i in range(10):
    train[f'ch{i}'] = train.f_27.str.get(i).apply(ord) - ord('A')
    test[f'ch{i}'] = test.f_27.str.get(i).apply(ord) - ord('A')
    char_features.append(f'ch{i}')

train["unique_characters"] = train.f_27.apply(lambda s: len(set(s)))
test["unique_characters"] = test.f_27.apply(lambda s: len(set(s)))
char_features.append('unique_characters')
train.drop('f_27', inplace=True, axis=1)
test.drop('f_27', inplace=True, axis=1)

# %%
train.head()

# %% [markdown]
# # Train model

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import catboost as cb

def fit_catboost(train, params=None):
    models = []
    valid_scores = []

    skf = StratifiedKFold(n_splits=Config.n_splits, shuffle=True, random_state=Config.seed)
    train_y = train.pop(Config.target)
    train_X = train

    for fold, (train_indices, valid_indices) in enumerate(skf.split(X=train_X, y=train_y)):
        X_train, X_valid = train_X.iloc[train_indices], train_X.iloc[valid_indices]
        y_train, y_valid = train_y.iloc[train_indices], train_y.iloc[valid_indices]

        model = cb.CatBoostClassifier(cat_features=char_features,
                                      learning_rate=0.03,
                                      eval_metric='AUC',
                                      early_stopping_rounds=10,
                                      verbose=True,)

        model.fit(X=X_train,
                  y=y_train,
                  eval_set=[(X_valid, y_valid)],
                  plot=True,
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
    pred = np.array([model.predict_proba(feat) for model in models])
    pred = np.argmax(np.mean(pred, axis=0) , axis=1)
    return pred

# %%
train.drop(['id'], axis=1, inplace=True)
feat = test.drop(['id'], axis=1)

# %%
models = fit_catboost(train=train)
pred = inference_catboost(models=models, feat=feat)

# %% [markdown]
# # Postprocess

# %%
def plot_importances(model):
    importance_df = pd.DataFrame(model.feature_importances_,
                                 index=model.feature_names_,
                                 columns=['importance'])\
                        .sort_values("importance", ascending=False)

    plt.subplots(figsize=(len(model.feature_names_) // 4, 5))
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
sub.to_csv(OUTPUT_DIR + 'catboost_baseline.csv', index=False)


