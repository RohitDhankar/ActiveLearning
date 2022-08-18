from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling

import numpy as np

from FoilModel import FoilImageClassifier
from data import ClassificationDataManager
from strategies.diversity import diversity_sampling_strategy_global

class ActiveLearningManager:
    def __init__(self, learner: ActiveLearner, X_pool, y_pool) -> None:
        self.learner = learner
        self.X_pool = X_pool
        self.y_pool = y_pool

    def run_active_learning(self, n_queries: int, n_instances: int, plot=True, **query_args) -> None:
        query_lst = []
        acc_lst = []
        for i in range(n_queries):
            acc = self.learner.score(X_test, y_test)
            print(f"Round {i}: Accuracy = {acc}")
            acc_lst.append(acc)
            print(f"X training set size: {len(learner.X_training)}, X pool size: {len(self.X_pool)}")
            query_idx, query_item = learner.query(self.X_pool, n_instances=n_instances, **query_args)
            query_lst.append((query_idx, query_item))
            learner.teach(self.X_pool[query_idx], self.y_pool[query_idx])
            self.X_pool = np.delete(self.X_pool, query_idx, axis=0)
            self.y_pool = np.delete(self.y_pool, query_idx, axis=0)
        
        if plot:
            import matplotlib.pyplot as plt
            with plt.style.context('seaborn-whitegrid'):
                plt.plot(np.arange(n_queries, dtype=int), acc_lst, label='Accuracy')
                plt.xlabel(f"Query(+{n_instances} instances each round)")
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()


model = FoilImageClassifier()

data_parser = ClassificationDataManager()

X, X_unlabeled, y = data_parser.get_data_from_file()

X_train = np.array(X[:110])
y_train = np.array(y[:110])
X_test = np.array(X[-10:])
y_test = np.array(y[-10:])
initial_idx = np.array([0, 11, 24])
X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]
X_pool = np.delete(X_train, initial_idx, axis=0)
y_pool = np.delete(y_train, initial_idx, axis=0)

def random_query_strategy(classifier, X, n_instances=1):
    query_idx = np.random.choice(range(len(X)), size=n_instances, replace=False)
    return query_idx, X[query_idx]

def uncertainty_sampling_strategy(classifier, X, n_instances=1):
    probs = np.array(classifier.predict_proba(X))
    print(f"Uncertianty: {probs}")
    query_idx = np.argsort(probs)[:n_instances]
    print(f"Seleted to label: {query_idx}")
    return query_idx, X[query_idx]

learner = ActiveLearner(
    estimator=model,
    # query_strategy=uncertainty_sampling_strategy,
    query_strategy=random_query_strategy,
    X_training=X_initial, y_training=y_initial
)

active_learner = ActiveLearningManager(learner, X_pool, y_pool)
active_learner.run_active_learning(n_queries=10, n_instances=2)

# print(similarity_sample(X_train[2], X_train[1]))
# print(diversity_sampling_strategy_global(None, X, 5))