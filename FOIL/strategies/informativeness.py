from numbers import Number
import numpy as np
from modAL.models import ActiveLearner
import sys
import os
from FoilModel import FoilImageClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))


def informativeness_query_strategy(classifier: ActiveLearner, X, n_instances=1):
    X_measurement = measure_informativeness_all(X)
    X__sorted_index = np.argsort(X_measurement)
    print("Selected index by informativeness: ", X__sorted_index[-n_instances:])
    return X__sorted_index[-n_instances:], X[-n_instances:]


# Helpers
def get_objects(X):
    X_objs = []
    for x in X:
        objs = []
        objs_raw: dict[str, object] = x['object_detect']['object']
        for idx, value in objs_raw.items():
            objs.append(value['name'])
        X_objs.append(objs)
    return X_objs


def get_panel_obj(X_objs):
    common_objs = [el for sublist in X_objs for el in sublist]
    return max(common_objs, key=common_objs.count)


def get_panel(X_objs, y):
    results = {}
    for i in range(0,len(X_objs)):
        if y[i] not in results:
            results[y[i]] = []
        results[y[i]].append(i)
    output = {}
    for key, value in results.items():
        temp = []
        for item in value:
            temp.append(X_objs[item])
        output[key] = get_panel_obj(temp)
    return output


def min_max_norm(results):
    norm_list = []
    min_value = min(results)
    max_value = max(results)
    for value in results:
        tmp = (value - min_value) / (max_value - min_value)
        norm_list.append(tmp)
    return norm_list


def measure_informativeness_all(X):
    results = np.zeros(len(X), int)
    X_objs = get_objects(X)
    panel = get_panel_obj(X_objs)
    for i in range(0, len(X)):
        objs = X_objs[i]
        unique_objts = np.unique(objs)
        results[i] += len(objs)
        if panel in unique_objts:
            results[i] -= objs.count(panel)
    return min_max_norm(results)


def measure_informativeness_certain(X, sample):
    results = np.zeros(len(X), int)
    X_objs = get_objects(X)
    panel = get_panel_obj(X_objs)
    sample_objs = get_objects(sample)
    for i in range(0, len(X)):
        objs = X_objs[i]
        results[i] += len(objs)
        results[i] += objs.count(panel)
    np.append(results, len(sample_objs) - sample_objs.count(panel))
    return min_max_norm(results)[-1]


if __name__ == '__main__':
    from data import ClassificationDataManager
    dataM = ClassificationDataManager()
    X, X_unlabeled, y = dataM.get_data_from_file()
    model = FoilImageClassifier()
    # print(measure_informativeness_all(X))
    print(y)

