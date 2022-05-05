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
train.drop('id', inplace=True, axis=1)
test.drop('id', inplace=True, axis=1)

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
# # Analysys

# %% [markdown]
# ### The 16 float features

# %%
float_features = [f for f in train.columns if train[f].dtype == 'float64']

# Training histograms
fig, axs = plt.subplots(4, 4, figsize=(16, 16))
for f, ax in zip(float_features, axs.ravel()):
    ax.hist(train[f], density=True, bins=100)
    ax.set_title(f'Train {f}, std={train[f].std():.1f}')
plt.suptitle('Histograms of the float features')
plt.show()

# %% [markdown]
# ### Correlation matrix of the float features

# %%
import seaborn as sns
plt.figure(figsize=(12, 12))
sns.heatmap(train[float_features + ['target']].corr(), center=0, annot=True, fmt='.1f')
plt.show()

# %% [markdown]
# ### Plot dependence between every feature and the target

# %%
fig, axs = plt.subplots(4, 4, figsize=(16, 16))
for f, ax in zip(float_features, axs.ravel()):
    temp = pd.DataFrame({f: train[f].values,
                         'state': train[Config.target].values})
    temp = temp.sort_values(f)
    temp.reset_index(inplace=True)
    ax.scatter(temp[f], temp.state.rolling(15000, center=True).mean(), s=2)
    ax.set_xlabel(f'{f}')
plt.suptitle('How the target probability depends on single features')
plt.show()

# %% [markdown]
# ### The integer features

# %%
from matplotlib.ticker import MaxNLocator
int_features = [f for f in test.columns if test[f].dtype == 'int64' and f != 'id']

# Training histograms
fig, axs = plt.subplots(8, 3, figsize=(16, 16))
# plt.figure(figsize=(16, 16))
# for f, ax in zip(int_features, axs.ravel()):
for i, f in enumerate(int_features):
    plt.subplot(8, 3, i+1)
    ax = plt.gca()
    vc = train[f].value_counts()
    ax.bar(vc.index, vc)
    #ax.hist(train[f], density=False, bins=(train[f].max()-train[f].min()+1))
    ax.set_xlabel(f'Train {f}')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # only integer labels
plt.suptitle('Histograms of the integer features')
plt.show()

# %% [markdown]
# # Principal Component Analysys

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X=train[float_features])
train[float_features] = scaler.transform(X=train[float_features])

# %%
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X=train.drop(Config.target, axis=1))
pc = pca.transform(X=train.drop(Config.target, axis=1))

# %% [markdown]
# ### Check principal components

# %%
pd.DataFrame(pc, columns=["PC{}".format(x + 1) for x in range(len(train.drop(Config.target, axis=1).columns))]).head()

# %%
plt.figure(figsize=(6, 6))
plt.scatter(pc[:, 0], pc[:, 1], alpha=0.8)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# %% [markdown]
# ### Contribution rate

# %%
pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(train.drop(Config.target, axis=1).columns))])

# %% [markdown]
# ### Cumulative contribution rate

# %%
import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid()
plt.show()

# %% [markdown]
# ### Eigenvalue

# %%
pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(train.drop(Config.target, axis=1).columns))])

# %% [markdown]
# ### Eigenvector

# %%
pd.DataFrame(pca.components_, columns=train.drop(Config.target, axis=1).columns, index=["PC{}".format(x + 1) for x in range(len(train.drop(Config.target, axis=1).columns))])

# %%
plt.figure(figsize=(6, 6))
for x, y, name in zip(pca.components_[0], pca.components_[1], train.drop(Config.target, axis=1).columns):
    plt.text(x, y, name)
plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# %%
train.columns

# %%
selected_features = ['f_10', 'f_11', 'f_17', 'ch3', 'ch6', 'ch7']

pca = PCA()
pca.fit(X=train[selected_features])
pc = pca.transform(X=train[selected_features])

# %%
plt.figure(figsize=(6, 6))
plt.scatter(pc[:, 0], pc[:, 1], alpha=0.8)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# %%
pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(selected_features))])

# %%
plt.figure(figsize=(6, 6))
for x, y, name in zip(pca.components_[0], pca.components_[1], selected_features):
    plt.text(x, y, name)
plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


