#!/usr/bin/env python
from PIL import Image, ImageTk
import argparse
import numpy
from keras.models import model_from_json
def create_parser():
    parser = argparse.ArgumentParser(description='pridect digit image')    
    parser.add_argument('file', action="store", type=str)
    return parser

def create_model():
    with open('model.txt', 'r') as config_file:
        config = config_file.read()
        model = model_from_json(config)
        model.load_weights('weight.txt')
        return model

if __name__ == "__main__":
    parser=create_parser()
    paras=parser.parse_args()
    img=Image.open(paras.file)
    data=numpy.array(list(img.getdata()),dtype=float).reshape((1,1,28,28))
    data=(255-data)/255
    print data
    model=create_model()
    print model.predict(data)

