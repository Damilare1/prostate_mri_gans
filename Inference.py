import sys
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import TrainingUtils

class Inference:

    _model_path = '/content/drive/MyDrive/Dissertation/Dissertation/model-files'
    _img_dir = './AugmentedImages'
    _image_type = 'ADC'
    _model = None
    _model_name = 'gen-model-*'
    _seed = None
    _utils = None
    _id = None

    def __init__(self) -> None:
        self._utils = TrainingUtils()
        self.set_id(self._utils.get_random_id())

    def get_id(self):
        return self._id
    def set_id(self, id):
        self._id = id
    def get_image_type(self):
        return self._image_type
    def set_image_type(self, type):
        self._image_type = type

    def get_model_name(self):
        return self._model_name
    
    def set_model_name(self, name):
        self._model_name = name

    def get_model_path(self):
        return self._model_path
    
    def set_model_path(self, path):
        self._model_path = path

    def _get_top_model(self):
        path = '{}/{}/{}'.format(self.get_model_path(), self.get_image_type(), self.get_model_name())
        print(path)
        for file in glob(path):
            return file
        raise Exception('Empty model location')
    
    def _get_model_details(self):
        model_name = self._get_top_model()
        model_id = model_name[-12:-6]
        model_date = model_name[-23:-13]
        seed = self._get_model_seed(model_id=model_id, model_date=model_date)
        return model_name, model_id, model_date, seed
        
    def _get_model_seed(self, model_id, model_date):
        path = '{}/{}/seed_{}_{}_*'.format(self.get_model_path(), self.get_image_type(),model_date, model_id)
        for file in glob(path):
            return file
        raise Exception('Seed not found for model')

    def _load_model_and_seed(self):
        model_name, _,_, seed = self._get_model_details()
        return tf.keras.models.load_model(model_name), np.load(seed)
    
    def get_model_summary(self):
        model = self._load_model_and_seed()
        model.summary()
    
    def start(self):
        model, seed = self._load_model_and_seed()
        self._utils.set_seed(seed)
        self._generate_and_save_augmented_images(model)

    def _generate_and_save_augmented_images(self, generator):
        img_dir = '{}/{}'.format(self._img_dir, self.get_id())
        self._utils.save_images(generator, img_dir)

sys.modules[__name__] = Inference