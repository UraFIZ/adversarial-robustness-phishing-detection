"""
common.py — спільні компоненти експерименту.
Дані, моделі (RandomForest, MLP), повний набір метрик, обгортки атак ART з
підтримкою масок збурення (для реалістичних обмежених атак).
Усе детерміноване: SEED=42.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import warnings
warnings.filterwarnings("ignore")

SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path, test_size=0.25, seed=SEED):
    """Завантаження, нормалізація ознак у [0,1], стратифікований спліт."""
    import pandas as pd
    df = pd.read_csv(path)
    names = list(df.columns[:-1])
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)
    mn, mx = X.min(0), X.max(0)
    rng = np.where((mx - mn) == 0, 1, (mx - mn))
    X = ((X - mn) / rng).astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    return Xtr, Xte, ytr, yte, names


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


def make_art_classifier(model):
    from art.estimators.classification import PyTorchClassifier
    return PyTorchClassifier(
        model=model, loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        input_shape=(next(model.parameters()).shape[1] if False else model.net[0].in_features,),
        nb_classes=2, clip_values=(0.0, 1.0))


def train_rf(Xtr, ytr, seed=SEED):
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(Xtr, ytr)
    return rf


def train_mlp(Xtr, ytr, epochs=15, batch=256, seed=SEED):
    set_seed(seed)
    d = Xtr.shape[1]
    clf = make_art_classifier(MLP(d))
    clf.fit(Xtr, ytr, nb_epochs=epochs, batch_size=batch, verbose=False)
    return clf


def full_metrics(y_true, y_pred, y_score=None):
    """Повний набір метрик. pos_label=1 (фішинг). FPR із матриці помилок."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    m = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall":    recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1":        f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "fpr":       fpr,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_score is not None:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            m["roc_auc"] = float("nan")
    return m


def predict_labels_scores(clf, X, is_torch=True):
    """Повертає (мітки, скори_класу_1)."""
    if is_torch:
        proba = clf.predict(X)
        # ART PyTorchClassifier повертає логіти/ймовірності за класами
        scores = _softmax(proba)[:, 1]
        labels = np.argmax(proba, axis=1)
    else:  # sklearn RF
        proba = clf.predict_proba(X)
        scores = proba[:, 1]
        labels = clf.predict(X)
    return labels, scores


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run_attack(clf, X, attack_type, eps, mask=None, eps_step=0.01, max_iter=20):
    """
    Генерує adversarial-приклади.
    attack_type: 'fgsm' | 'pgd'. mask (булева, shape=(d,)) — де дозволено збурення.
    """
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
    if attack_type == "fgsm":
        atk = FastGradientMethod(clf, eps=eps)
    elif attack_type == "pgd":
        atk = ProjectedGradientDescent(
            clf, eps=eps, eps_step=eps_step, max_iter=max_iter, verbose=False)
    else:
        raise ValueError(attack_type)
    if mask is not None:
        m = mask.astype(np.float32)  # broadcastable (d,) -> (n,d)
        return atk.generate(X, mask=m)
    return atk.generate(X)
