from pprint import pprint
from FoilModel import FoilImageClassifier

from foil_types import *

class ClassificationDataManager:
    """Provide utility methods for data preprocessing in classification mode."""
    
    def get_data_from_db(self, url="mongodb://localhost:27017"):
        """ 
        Retrive data from MongoDB and parse data into X, X_unlabeled and y.

        @param url: url to db
        @return: X, X_unlabeled, y
        """
        from pymongo import MongoClient
        client = MongoClient(url)
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

        client.close()
        return X, X_unlabeled, y

    def save_data_to_file(self, X=[], X_unlabeled=[], y=[], filename='data', from_db=False):
        """ 
        Save X, X_unlabeled and y to local file.

        @param X: data X
        @param X_unlabeled: data X_unlabeled
        @param y: data y
        @param filename: filename to save
        @param from_db: if True, data is retrived from db and no need to provide X, X_unlabeled and y
        """
        import os
        if(os.path.isfile(filename)):
            print("File already exists. Stop saving.")
            return
        import pickle
        if from_db:
            X, X_unlabeled, y = self.get_data_from_db()
        # save X, X_unlabeled, y as a list
        with open(filename, 'wb') as f:
            pickle.dump([X, X_unlabeled, y], f)

    def get_data_from_file(self, filename='data'):
        """
        Load X, X_unlabeled and y from local file.

        @param filename: filename to load
        @return: X, X_unlabeled, y
        """
        import pickle
        with open(filename, 'rb') as f:
            return pickle.load(f)

if __name__ == '__main__':
    data_parser = ClassificationDataManager()
    # data_parser.save_data_to_file(from_db=True)
    X, X_unlabeled, y = data_parser.get_data_from_file()
    # print('X[0]: ', X[0])
    # print('y[0]: ', y[0])
    for idx, v in enumerate(y):
        print(f"{idx}: {v}")
    # print(X_unlabeled[0])
    pass