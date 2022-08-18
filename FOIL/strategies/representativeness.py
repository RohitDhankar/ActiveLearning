from numbers import Number
import numpy as np
from modAL.models import ActiveLearner
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from FoilModel import FoilImageClassifier



def representativeness_query_strategy(classifier: ActiveLearner, X, n_instances=1):
    model: FoilImageClassifier = classifier.estimator
    X_vec = featurelize(X, model.get_object_list())
    X_density = compute_density(X_vec)
    X_density_sorted_idx = np.argsort(X_density)
    print(f"Selected idx by representative: {X_density_sorted_idx[-n_instances:]}")
    return X_density_sorted_idx[-n_instances:], X[-n_instances:]


# Helpers

def compute_density(X):
    """
    Compute the density of the given data.
    formula: sum(similarity(x1, x2)) / n-1
    """
    n = len(X)
    result = []
    for i in range(n):
        density = 0
        for j in range(n):
            if i == j:
                continue
            density += compute_similarity_measure(X[i], X[j])
        result.append(density / (n - 1))
    return np.array(result)

def compute_similarity_measure(x1, x2):
    """
    Compute the similarity measure of the given data.
    formula: (x1 * x2) / ||x1|| ||x2||
    """
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def featurelize(X, object_lst: list[str]):
    """
    Featurelize the given data X by mapping objects to one-hot vectors.

    @param X: The data to featurelize.
    @param object_lst: The list of objects for the data.
    @return: The featurelized data.
    """
    X_objs = get_objects(X)
    X_vectors = []
    for x in X_objs:
        vec = [0 for _ in range(len(object_lst))]
        for obj in x:
            if obj in object_lst:
                vec[object_lst.index(obj)] = 1
            # else:
            #     print(f"Object not found: {obj}")
        X_vectors.append(vec)
    return np.array(X_vectors)

def get_objects(X) -> list[list[str]]:
    """
    Parse each input into a list of objects it contains.

    @param X: The data to get the list of objects from.
    @return: The 2d list of objects.
    """
    X_objs = []
    for x in X:
        objs = []
        objs_raw: dict[str, object] = x['object_detect']['object']
        for idx, value in objs_raw.items():
            objs.append(value['name'])
        X_objs.append(objs)
    return X_objs

if __name__ == '__main__':
    from data import ClassificationDataManager
    dataM = ClassificationDataManager()
    X, X_unlabeled, y = dataM.get_data_from_file()
    model = FoilImageClassifier()
    model.fit(X, y)
    # print(model.print_object_list())
    print('====================')
    result = featurelize(X, model.get_object_list())
    print(compute_similarity_measure(result[0], result[2]))
    pass