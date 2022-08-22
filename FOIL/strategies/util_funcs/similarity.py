import sys
import os
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from data import ClassificationDataManager
from foil_types import *

def compute_similarity(x1: FoilXItem, x2: FoilXItem):
    """Computes the similarity between two data entries.
    """
    pass

# Helpers

def get_obj_and_overlap(x: FoilXItem):
    """Get the object list and overlap list of the given data.
    """
    obj_lst = []
    overlap_lst = []
    objects = x['object_detect']['object']
    overlaps = x['object_detect']['overlap']

    for key, obj in objects.items():
        obj_lst.append(obj['name'])
        
    for key, obj in overlaps.items():
        overlap_lst.append((obj_lst[obj['idA']], obj_lst[obj['idB']]))

    obj_lst = list(dict.fromkeys(obj_lst))
    overlap_lst = list(dict.fromkeys(overlap_lst))
    return obj_lst, overlap_lst


if __name__ == '__main__':
    from jycm.jycm import YouchamaJsonDiffer
    from jycm.helper import make_ignore_order_func

    dataM = ClassificationDataManager()
    X, X_unlabeled, y = dataM.get_data_from_file()

    # ycm = YouchamaJsonDiffer({'a': {'b': {'c': 1}, 'd': 2}}, {'a': {'b': {'c': 3}, 'd': 2}})
    obj_lst1, overlap_lst1 = get_obj_and_overlap(X[1])
    obj_lst2, overlap_lst2 = get_obj_and_overlap(X[20])

    left = {
        'value': obj_lst1 + overlap_lst1
    }

    right = {
        'value': obj_lst2 + overlap_lst2
    }

    ycm = YouchamaJsonDiffer(left, right , ignore_order_func=make_ignore_order_func(['value']))
    print(left['value'], '\n\n',right['value'])
    print("====================================")
    result = ycm.get_diff()
    for key in result:
        print(key, len(result[key]))
        # print(result[key])

    pass