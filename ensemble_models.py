"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    X_train = np.asarray(data.train_features, dtype=np.float64)
    y_train = np.asarray(data.train_target, dtype=np.int64).ravel()
    X_test = np.asarray(data.test_features, dtype=np.float64)
    y_test = np.asarray(data.test_target, dtype=np.int64).ravel()

    seed = int(data.random_seed)

    # Три разнородных базовых алгоритма: бэггинг, бустинг и гистограммный бустинг.
    estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            n_jobs=-1, random_state=seed,
        )),
        ("gb", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=seed,
        )),
        ("hgb", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=None,
            random_state=seed,
        )),
    ]

    model = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

    # fit ТОЛЬКО на train — test_target сюда не попадает
    model.fit(X_train, y_train)

    # Вероятности положительного класса для ROC-AUC
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)

    # Метрики считаются post-hoc на test
    result = EnsembleResult(
        model_name="VotingClassifier(RandomForest + GradientBoosting + HistGradientBoosting)",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0)),
    )
    return result
