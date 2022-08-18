from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling

import numpy as np
import copy

from FoilModel import FoilImageClassifier
from data import ClassificationDataManager
from strategies.diversity import diversity_sampling_strategy_global
from strategies.representativeness import representativeness_query_strategy

class ActiveLearningManager:
    def __init__(self, learner: ActiveLearner, X_pool, y_pool, X_test, y_test) -> None:
        self.learner = learner
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.X_test = X_test
        self.y_test = y_test

    def run_active_learning(self, n_queries: int, n_instances: int, plot=True, **query_args):
        query_lst = []
        acc_lst = []
        for i in range(n_queries):
            acc = self.learner.score(self.X_test, self.y_test)
            print(f"Round {i}: Accuracy = {acc}")
            acc_lst.append(acc)
            print(f"X training set size: {len(self.learner.X_training)}, X pool size: {len(self.X_pool)}")
            query_idx, query_item = self.learner.query(self.X_pool, n_instances=n_instances, **query_args)
            query_lst.append((query_idx, query_item))
            self.learner.teach(self.X_pool[query_idx], self.y_pool[query_idx])
            self.X_pool = np.delete(self.X_pool, query_idx, axis=0)
            self.y_pool = np.delete(self.y_pool, query_idx, axis=0)
        
        if plot:
            import matplotlib.pyplot as plt
            with plt.style.context('seaborn-whitegrid'):
                plt.plot(np.arange(n_queries, dtype=int), acc_lst, label='Accuracy')
                plt.xlabel(f"Query(+{n_instances} instance(s) per round)")
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()
        return query_lst, acc_lst

    def run_against_random(self, n_queries: int, n_instances: int, plot=True, **query_args) -> None:
        # For random
        random_X_pool = copy.deepcopy(self.X_pool)
        random_y_pool = copy.deepcopy(self.y_pool)
        random_leaner = copy.deepcopy(self.learner)
        random_leaner.query_strategy = random_query_strategy
        random_query_lst = []
        random_acc_lst = []
        for i in range(n_queries):
            acc = random_leaner.score(self.X_test, self.y_test)
            print(f"Round {i}: Accuracy = {acc}")
            random_acc_lst.append(acc)
            print(f"X training set size: {len(random_leaner.X_training)}, X pool size: {len(random_X_pool)}")
            query_idx, query_item = random_leaner.query(random_X_pool, n_instances=n_instances, **query_args)
            random_query_lst.append((query_idx, query_item))
            random_leaner.teach(random_X_pool[query_idx], random_y_pool[query_idx])
            random_X_pool = np.delete(random_X_pool, query_idx, axis=0)
            random_y_pool = np.delete(random_y_pool, query_idx, axis=0)

        # For active learning
        query_lst, acc_lst = self.run_active_learning(n_queries, n_instances, plot=False, **query_args)

        if plot:
            import matplotlib.pyplot as plt
            with plt.style.context('seaborn-whitegrid'):
                plt.plot(np.arange(n_queries, dtype=int), random_acc_lst, label='Random')
                plt.plot(np.arange(n_queries, dtype=int), acc_lst, label='Active Learning')
                plt.xlabel(f"Query(+{n_instances} instance(s) per round)")
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()

model = FoilImageClassifier()

data_parser = ClassificationDataManager()

X, X_unlabeled, y = data_parser.get_data_from_file()

X_train = np.array(X[:90])
y_train = np.array(y[:90])
X_test = np.array(X[-30:])
y_test = np.array(y[-30:])
initial_idx = np.array([0, 11, 24])
X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]
X_pool = np.delete(X_train, initial_idx, axis=0)
y_pool = np.delete(y_train, initial_idx, axis=0)

def random_query_strategy(classifier, X, n_instances=1):
    query_idx = np.random.choice(range(len(X)), size=n_instances, replace=False)
    print(f"Seleted idx by random: {query_idx}")
    return query_idx, X[query_idx]

def uncertainty_sampling_strategy(classifier, X, n_instances=1):
    probs = np.array(classifier.predict_proba(X))
    print(f"Uncertianty: {probs}")
    query_idx = np.argsort(probs)[-n_instances:]
    print(f"Seleted to label: {query_idx}")
    return query_idx, X[query_idx]

learner = ActiveLearner(
    estimator=model,
    # query_strategy=uncertainty_sampling_strategy,
    query_strategy=representativeness_query_strategy,
    X_training=X_initial, y_training=y_initial
)

active_learner = ActiveLearningManager(learner, X_pool, y_pool, X_test, y_test)
active_learner.run_against_random(n_queries=50, n_instances=1)

# print(similarity_sample(X_train[2], X_train[1]))
# print(diversity_sampling_strategy_global(None, X, 5))