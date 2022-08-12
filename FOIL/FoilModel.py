from abc import abstractmethod

from foil_types import *

from foil import FOIL

from label import label

import numpy as np

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
    def parse_data(image_meta_data, interpretation, isManual=True) -> FoilXItem:
        X: FoilXItem = {
            "imageId": image_meta_data['imageId'],
        }
        for key, value in interpretation.items():
            X[key] = value
        if(isManual):
            y = image_meta_data['labels'][0]['name'][0]
        else:
            y = None
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