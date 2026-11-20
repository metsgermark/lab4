
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
)

from grader_contracts.ensembles import EnsembleInput
from ensemble_models import run_ensemble_experiment

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_german(path: str = "german.csv"):
    df = pd.read_csv(path, sep=";")
    target = df.columns[0]
    X = df.drop(columns=[target]).values.astype(np.float64)
    y = df[target].values.astype(np.int64)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_tr, X_te, y_tr, y_te


def metrics(y_true, y_pred, proba):
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def build_models(seed: int):
    """Словарь name -> estimator. Для MLP добавлен scaler внутри pipeline."""
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2,
                                n_jobs=-1, random_state=seed)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                    max_depth=3, subsample=0.8, random_state=seed)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         random_state=seed)
    return {
        "LogisticRegression (baseline lab3)": LogisticRegression(max_iter=1000, random_state=seed),
        "RandomForest": rf,
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, max_depth=12,
                                            min_samples_leaf=2, n_jobs=-1, random_state=seed),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=seed),
        "GradientBoosting": gb,
        "HistGradientBoosting": hgb,
        "MLP (64,32)": make_pipeline(StandardScaler(),
                                     MLPClassifier(hidden_layer_sizes=(64, 32),
                                                   max_iter=800, random_state=seed)),
        "Voting (RF+GB+HGB)": VotingClassifier(
            estimators=[("rf", rf), ("gb", gb), ("hgb", hgb)], voting="soft", n_jobs=-1),
        "Stacking (RF+GB->LR)": StackingClassifier(
            estimators=[("rf", RandomForestClassifier(n_estimators=200, random_state=seed)),
                        ("gb", GradientBoostingClassifier(n_estimators=150, random_state=seed))],
            final_estimator=LogisticRegression(max_iter=1000), cv=3, n_jobs=-1),
    }


def main():
    X_tr, X_te, y_tr, y_te = load_german()
    table = []

    for name, model in build_models(RANDOM_STATE).items():
        model.fit(X_tr, y_tr)                       # fit только на train
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(np.int64)
        row = {"model": name, **metrics(y_te, pred, proba)}
        table.append(row)
        print(f"{name:42s} ROC-AUC={row['roc_auc']:.4f}  acc={row['accuracy']:.4f}  f1={row['f1']:.4f}")

    # Финальные метрики = результат контрактной функции на том же split
    data = EnsembleInput(
        train_features=X_tr, train_target=y_tr,
        test_features=X_te, test_target=y_te,
        random_seed=RANDOM_STATE,
    )
    res = run_ensemble_experiment(data)
    final = {
        "model_name": res.model_name,
        "roc_auc": round(res.roc_auc, 4),
        "accuracy": round(res.accuracy, 4),
        "precision": round(res.precision, 4),
        "recall": round(res.recall, 4),
        "f1": round(res.f1, 4),
    }

    best = max(table, key=lambda r: r["roc_auc"])
    payload = {
        "experiments": table,
        "best_by_roc_auc": best,
        "selected_model": final,
    }
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n Лучшая по ROC-AUC в таблице ")
    print(best)
    print("\n Финальная модель из run_ensemble_experiment ")
    print(final)
    print("\n experiment_results.json записан")


if __name__ == "__main__":
    main()