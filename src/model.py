import dlib, cv2, os
from imutils import face_utils
from tqdm.auto import tqdm
import datetime
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
# %matplotlib inline
# %load_ext tensorboard
 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from dog_face_detector import DogFaceDetector

dfd = DogFaceDetector()

def create_face_dataset(path_to_dataset, outpath):
    filepaths = [x for x, y, z in os.walk(path_to_dataset)]
    categories = [y for x, y, z in os.walk(path_to_dataset)][0]
    img_lists = [z for x, y, z in os.walk(path_to_dataset)]
    for fpath, cat, img_lst in zip(filepaths[1:], categories, img_lists[1:]):
        pbar = tqdm(img_lst)
        if cat == 'Young':
            tqdm.write('Skipping Young')
            continue
        for fname in pbar:
            try:
                dfd.get_dogface(img_path=fpath+'/'+fname, 
                                outpath=f"{outpath}/{cat}/", verbose=False)
            except:
                tqdm.write(str(fname))
                continue
        print(f"{cat} done.")

class DogAgeModel:
    def __init__(self, preload_model: (str or bool or None) = False) -> None:
        if isinstance(preload_model, str):
            from keras.models import load_model
            self.model = load_model(preload_model)
        else:
            self.model = self.generate_model()
    
    def generate_model(self):
        METRICS = [
                        keras.metrics.TruePositives(name='tp'),
                        keras.metrics.FalsePositives(name='fp'),
                        keras.metrics.TrueNegatives(name='tn'),
                        keras.metrics.FalseNegatives(name='fn'), 
                        keras.metrics.CategoricalAccuracy(name='acc'),
                        keras.metrics.Precision(name='pcn'),
                        keras.metrics.Recall(name='rcl'),
                        keras.metrics.AUC(name='auc'),
                        keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
                        ]
        IMG_LEN = 224
        IMG_SHAPE = (IMG_LEN,IMG_LEN,3)
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights='imagenet', classes=3)
        base_model.trainable = False
        
        model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.75),
        tf.keras.layers.Dense(3, activation='softmax'), 
        ])

        model.compile(optimizer=tf.keras.optimizers.Adamax(0.001),
              loss='categorical_crossentropy',
              metrics=METRICS)
        
        return model

    def generate_generators(self, path_to_dataset, train_test_val=True):
        img_width = 224
        img_height = 224
        batch_size = 128

        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                            shear_range=0.1,
                                            zoom_range=0.1,
                                            # rotation_range=30,
                                            horizontal_flip=True)

        val_datagen = ImageDataGenerator(rescale=1. / 255)

            train_generator = train_datagen.flow_from_directory(
                '/content/ALL3/train',
                target_size=(img_width, img_height),
                # batch_size=batch_size,
                class_mode='categorical')
            validation_generator = val_datagen.flow_from_directory(
                '/content/ALL3/val',
                target_size=(img_width, img_height),
                # batch_size=batch_size,
                class_mode='categorical')
            test_generator = val_datagen.flow_from_directory(
                '/content/ALL3/test',
                target_size=(img_width, img_height),
                # batch_size=batch_size,
                class_mode='categorical')

        
