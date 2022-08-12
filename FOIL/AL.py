from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling

import numpy as np

from FoilModel import FoilImageClassifier

from data import get_data_from_file

model = FoilImageClassifier()

X, y = get_data_from_file()

X_train = np.array(X[:10])
y_train = np.array(y[:10])
X_test = np.array(X[10:])
y_test = np.array(y[10:])
initial_idx = np.array([0, 5])
X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]
X_pool = np.delete(X_train, initial_idx, axis=0)
y_pool = np.delete(y_train, initial_idx, axis=0)


def random_query_strategy(classifier, X, n_instances=1):
    query_idx = np.random.choice(range(len(X)), size=n_instances, replace=False)
    return query_idx, X[query_idx]

def uncertainty_sampling_strategy(classifier, X, n_instances=1):
    probs = np.array(classifier.predict_proba(X))
    print(probs)
    query_idx = np.argsort(probs)[:n_instances]
    return query_idx, X[query_idx]

learner = ActiveLearner(
    estimator=model,
    query_strategy=uncertainty_sampling_strategy,
    X_training=X_initial, y_training=y_initial
)

n_queries = 5
for i in range(n_queries):
    print(f"Round {i}: {learner.score(X_test, y_test)}")
    print(f"Current training set length: {len(learner.X_training)}")
    query_idx, query_item = learner.query(X_pool, n_instances=1)
    learner.teach(X_pool[query_idx], y_pool[query_idx])
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)