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

def fit_lgbm(train, params):
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
                                    random_state=Config.seed,
                                    verbose=-1)
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


