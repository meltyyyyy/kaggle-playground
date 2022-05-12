# %%
import sys
from pathlib import Path
sys.path.append(str(Path('__file__').resolve().parent.parent.parent))

# %%
from configs.data import INPUT_DIR, OUTPUT_DIR
from configs.nn import Config
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

# %%
train["unique_characters"] = train.f_27.apply(lambda s: len(set(s)))
test["unique_characters"] = test.f_27.apply(lambda s: len(set(s)))

# %%
train.drop('f_27', inplace=True, axis=1)
test.drop('f_27', inplace=True, axis=1)

# %%
train.head()

# %% [markdown]
# # Build model

# %%
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'], label = 'Validation Error')
    plt.ylim([0,0.4])
    plt.legend()
    plt.show()

# %%
from tensorflow.keras import Sequential, layers, optimizers, losses, regularizers
import tensorflow as tf

def build_model():
    model = Sequential()
    model.add(layer=layers.Input(shape=(41,)))
    model.add(layer=layers.Dense(128, kernel_regularizer=regularizers.l2(30e-6) ,activation=tf.nn.swish))
    model.add(layer=layers.Dense(32, kernel_regularizer=regularizers.l2(30e-6), activation=tf.nn.swish))
    model.add(layer=layers.Dense(192, kernel_regularizer=regularizers.l2(30e-6), activation=tf.nn.swish))
    model.add(layer=layers.Dropout(rate=0.2))
    model.add(layer=layers.Dense(32, kernel_regularizer=regularizers.l2(30e-6), activation=tf.nn.swish))
    model.add(layer=layers.Dense(32, kernel_regularizer=regularizers.l2(30e-6), activation=tf.nn.swish))
    model.add(layer=layers.Dropout(rate=0.3))
    model.add(layer=layers.Dense(1, activation=tf.nn.sigmoid))

    optimizer = optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer=optimizer, loss=losses.binary_crossentropy)

    return model

# %%
model = build_model()
model.summary()

# %%
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

def fit_nn(train):
    models = []
    valid_scores = []

    kf = KFold(n_splits=Config.n_splits, shuffle=True, random_state=0)
    train_y = train.pop(Config.target)
    train_X = train

    for fold, (train_indices, valid_indices) in enumerate(kf.split(X=train_X, y=train_y)):
        X_train, X_valid = train_X.iloc[train_indices], train_X.iloc[valid_indices]
        y_train, y_valid = train_y.iloc[train_indices], train_y.iloc[valid_indices]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X=X_train)
        X_valid = scaler.transform(X=X_valid)

        model = build_model()

        callbacks = [EarlyStopping(monitor='val_loss',
                                   restore_best_weights=True,
                                   patience=Config.early_stopping_rounds),
                     ReduceLROnPlateau(monitor='val_loss',
                                       patience=Config.lr_reducing_rounds,
                                       factor=Config.reducing_factor,
                                       mode='min',
                                       verbose=1),
                     TerminateOnNaN()]

        history = model.fit(x=X_train,
                            y=y_train,
                            validation_data=[X_valid, y_valid],
                            batch_size=Config.batch_size,
                            epochs=Config.epochs,
                            shuffle=True,
                            callbacks=[callbacks])

        # ------ plot ------
        plot_history(history=history)

        # ------ prediction ------
        y_valid_pred = model.predict(X_valid, batch_size=Config.batch_size, verbose=1)
        score = roc_auc_score(y_true=y_valid, y_score=y_valid_pred)

        print(f'fold {fold} AUC: {score}')
        valid_scores.append(score)
        models.append(model)

    cv_score = np.mean(valid_scores)
    print(f'CV score: {cv_score}')
    return models

def inference_nn(models, feat):
    pred = [rankdata(model.predict(feat, batch_size=Config.batch_size, verbose=1)) for model in models]
    pred = np.mean(pred, axis=0)
    return pred

# %%
train.drop(['id'], axis=1, inplace=True)
feat = test.drop(['id'], axis=1)

# %%
models = fit_nn(train=train.copy())
pred = inference_nn(models=models, feat=feat)

# %% [markdown]
# # Submission

# %%
sub = pd.read_csv(INPUT_DIR + 'sample_submission.csv')
test[Config.target] = pred
sub = test.loc[:, ['id', Config.target]].reset_index(drop=True)
sub.to_csv(OUTPUT_DIR + 'nn_baseline.csv', index=False)


