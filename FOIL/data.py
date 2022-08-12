from pprint import pprint
from FoilModel import FoilImageClassifier

from foil_types import *

CONNECTING_STRING = "mongodb://localhost:27017"

def get_classification_data_from_db():
    from pymongo import MongoClient
    client = MongoClient(CONNECTING_STRING)
    db = client['NSIL']
    images_collection = db['images']
    workspaces_collection = db['workspaces']

    # Get all images
    images: list[object] = images_collection.find({})
    # Get all imageMetaDatas
    ws = workspaces_collection.find_one({})
    image_meta_datas = ws['collections'][0]['images']
    
    # Get X, y
    X = []
    X_unlabeled = []
    y = []
    for img_md in image_meta_datas:
        img_id = img_md['imageId']
        target_img = next(x for x in images if str(x['_id']) == str(img_id))
        # process labeled images
        if img_md['labels'] and len(img_md['labels']) == 1 and img_md['labels'][0]['name'] and len(img_md['labels'][0]['name']) == 1 and img_md['labels'][0]['confirmed']:                
            x_single, y_single = FoilImageClassifier.parse_data(img_md, target_img['interpretation'])
            X.append(x_single)
            y.append(y_single)
        else:
            # process unlabeled images
            x_single, _ = FoilImageClassifier.parse_data(img_md, target_img['interpretation'], isManual=False)
            X_unlabeled.append(x_single)
    return X, X_unlabeled, y

def save_data_to_file(X=[], X_unlabeled=[], y=[], filename='data', from_db=False):
    import os
    if(os.path.isfile(filename)):
        print("File already exists. Stop saving.")
        return
    import pickle
    if from_db:
        X, X_unlabeled, y = get_classification_data_from_db()
    # save X, X_unlabeled, y as a list
    with open(filename, 'wb') as f:
        pickle.dump([X, X_unlabeled, y], f)

def get_data_from_file(filename='data'):
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    save_data_to_file(from_db=True)
    X, X_unlabeled, y = get_data_from_file()
    print('X[0]: ', X[0])
    print('y[0]: ', y[0])
    # print(len(X_unlabeled))
    # print(X_unlabeled[0])
    pass