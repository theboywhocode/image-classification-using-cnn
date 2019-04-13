
from keras.models import Sequential
from keras.models import model_from_json

def predict(model, X):
    img_class = model.predict_classes(X)
    prediction = img_class[0]
    classname = img_class[0]
    Y = classname
    return Y

if __name__ == '__main__':
    import sys
    img_dir = sys.argv[1]
    from get_dataset import get_img
    img = get_img(img_dir)
    import numpy as np
    X = np.zeros((1, 32, 32, 3), dtype='float32')
    X[0] = img
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    Y = predict(model, X)
    print('It is a ' + Y + ' !')
