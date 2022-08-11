from abc import abstractmethod
from typing import Any

from foil import FOIL

from label import label

import numpy as np

from data import Foily, FoilX, FoilXItem, get_data

FoilRules = dict[str, list[Any]]

class FoilBase:
    def __init__(self) -> None:
        self._rules: FoilRules = {}
        self._object_list = []
        self._initialized = False

    def get_rules(self) -> FoilRules:
        if not self._initialized:
            print("Please fit the model first.")
            return
        return self._rules

    def get_object_list(self) -> list[str]:
        if not self._initialized:
            print("Please fit the model first.")
            return
        return self._object_list

    def print_rules(self):
        if not self._initialized:
            print("Please fit the model first.")
            return
        print("Current Rules:")
        for idx, (k, v) in enumerate(self._rules.items()):
            print(f"Rule {idx}({k.split('(')[0]}): {v}")
        print("End")
    
    def print_object_list(self):
        if not self._initialized:
            print("Please fit the model first.")
            return
        print("Current Object List:")
        for idx, obj in enumerate(self._object_list):
            print(f"{idx}: {obj}")
        print("End")

    @abstractmethod
    def fit(self, X: FoilX, y: Foily, d, l) -> None:
        raise NotImplementedError("Method not implemented or directly call in FoilBase.")

    @abstractmethod
    def predict(self, X: FoilX) -> list[list[str]]:
        raise NotImplementedError("Method not implemented or directly call in FoilBase.")

    @abstractmethod
    def score(self, X: FoilX, y: Foily):
        raise NotImplementedError("Method not implemented or directly call in FoilBase.")

    @abstractmethod
    def predict_proba(self, X: FoilX):
        raise NotImplementedError("Method not implemented or directly call in FoilBase.")
    

class FoilImageClassifier(FoilBase):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def parse_manual_data(image_meta_data, interpretation) -> FoilXItem:
        X: FoilXItem = {
            "imageId": image_meta_data['imageId'],
        }
        for key, value in interpretation.items():
            X[key] = value
        y = image_meta_data['labels'][0]['name'][0]
        return X, y

    def fit(self, X: FoilX, y: Foily, d={}, l={}) -> None:
        combined_input = X.copy()
        for i in range(len(combined_input)):
            combined_input[i]['type'] = y[i]
        result = FOIL(input_list=combined_input, deleted=d, locked=l)
        self._rules = result[0]
        self._object_list = result[2]
        self._initialized = True
        return

    def predict(self, X: FoilX) -> list[list[str]]:
        if not self._initialized:
            print("Please fit the model first.")
            return
        result = label(dict_list=X, rules=self._rules)
        # drop '(X)'
        for l in result:
            for i in range(len(l)):
                l[i] = l[i].split('(')[0]
        return result

    def score(self, X: FoilX, y: Foily):
        if not self._initialized:
            print("Please fit the model first.")
            return
        result = self.predict(X)
        correct = 0
        total = 0
        for idx, l in enumerate(result):
            total += 1
            if len(l) == 1 and l[0] == y[idx]:
                correct += 1
        return correct / total

    def predict_proba(self, X: FoilX):
        if not self._initialized:
            print("Please fit the model first.")
            return
        label_result = label(dict_list=X, rules=self._rules)
        # calculate probability based on list length
        result = []
        for l in label_result: 
            list_len = len(l)
            prob = 1 / list_len
            result.append(prob)
        return result


if __name__ == '__main__':
    # print(FoilImageClassifier.parse_manual_data({'imageId': '1', 'name': 'lol', 'labels': [{'name': ['motorcyclist']}]}, {'object_detect': {'object': 'person'}, 'panoptic_segmentation': {'1': 'person'}, 'semantic_segmentation': {'1': '2'}}))
    X, y = get_data()
    model = FoilImageClassifier()
    model.fit(X, y)
    # model.print_rules()
    # model.print_object_list()
    # print(model.predict_proba([X[4]]))
    # print(model.score([X[0], X[1], X[2], X[3], X[4]], [y[0], y[1], y[2], y[3], y[4]]))
    pass