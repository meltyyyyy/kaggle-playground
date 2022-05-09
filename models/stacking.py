# %%
import sys
from pathlib import Path
sys.path.append(str(Path('__file__').resolve().parent.parent))

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
# # Ensambling & Stacking models

# %%
# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=Config.seed, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self,x,y):
        return self.clf.fit(x,y)

    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)

# %%
from sklearn.model_selection import KFold

def get_oof(clf, train_X, train_y, test_X, n_train, n_test):
    oof_train = np.zeros((n_train,))
    oof_test = np.zeros((n_test,))
    oof_test_skf = np.empty((5, n_test))
    kf = KFold(n_splits= 5, random_state=2022, shuffle=True)

    for fold, (train_indices, valid_indices) in enumerate(tqdm(kf.split(X=train_X, y=train_y))):
        X_train = train_X.iloc[train_indices]
        y_train = train_y.iloc[train_indices]
        X_valid = train_X.iloc[valid_indices]

        clf.fit(X_train, y_train)

        oof_train[valid_indices] = clf.predict_proba(X_valid)[:,1]
        oof_test_skf[fold, :] = clf.predict_proba(test_X)[:,1]

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# %%
lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'n_estimators': 10000,
    'learning_rate': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'num_leaves': 200,
    'max_bins': 511,
    'min_child_samples': 90
}

cat_params = {
    'cat_features': char_features,
}

rfc_params = {
    'n_estimators': 10000,
    'criterion': 'gini',
    'bootstrap': True,
}

# %%
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

lgbm = LGBMClassifier(**lgbm_params, random_state=2022)
cat = CatBoostClassifier(**cat_params, random_state=2022)
rfc = RandomForestClassifier(**rfc_params, random_state=2022)

# %%
train_X = train.drop(['id', 'target'], axis=1)
train_y = train['target']
test_X = test.drop('id', axis=1)
n_train = train_X.shape[0]
n_test = test_X.shape[0]

# %%
lgbm_oof_train, lgbm_oof_test = get_oof(lgbm, train_X, train_y, test_X, n_train, n_test)
cat_oof_train, cat_oof_test = get_oof(cat, train_X, train_y, test_X, n_train, n_test)
rfc_oof_train, rfc_oof_test = get_oof(rfc, train_X, train_y, test_X, n_train, n_test)

# %%
base_predictions_train = pd.DataFrame({
    'RandomForest': rfc_oof_train.ravel(),
    'CatBoost': cat_oof_train.ravel(),
    'lightGBM': lgbm_oof_train.ravel()})
base_predictions_train.head()

# %%
X_train = np.concatenate((rfc_oof_train, cat_oof_train, lgbm_oof_train), axis=1)
X_test = np.concatenate((rfc_oof_train, cat_oof_train, lgbm_oof_train), axis=1)

print('x_train.shape : ', X_train.shape)
print('x_test.shape : ', X_test.shape)

# %%
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=2000,
                    max_depth=4,
                    min_child_weight=2,
                    gamma=0.9,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    nthread=-1,
                    scale_pos_weight=1)

xgb.fit(X_train, train_y)
pred = xgb.predict_proba(X_test)[:, 1]

# %% [markdown]
# # Submission

# %%
sub = pd.read_csv(INPUT_DIR + 'sample_submission.csv')
test[Config.target] = pred
sub = test.loc[:, ['id', Config.target]].reset_index(drop=True)
sub.to_csv(OUTPUT_DIR + 'stacking.csv', index=False)
