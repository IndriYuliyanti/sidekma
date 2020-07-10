import os, shutil
import numpy as np

original_dataset_dir = ('D://hamatomat/dataa')
base_dir = 'data'

categories = ['sehat', 'hama']

str_train_val = ['train', 'validation']

if not os.path.exists(base_dir):
    os.mkdir(base_dir)
    print('Created directory: ', base_dir)

for dir_type in str_train_val:
    train_test_val_dir = os.path.join(base_dir, dir_type)

    if not os.path.exists(train_test_val_dir):
        os.mkdir(train_test_val_dir)

    for category in categories:
        dir_type_category = os.path.join(train_test_val_dir, category)

        if not os.path.exists(dir_type_category):
            os.mkdir(dir_type_category)
            print('Created directory: ', dir_type_category)

directories_dict = {}

np.random.seed(12)
for cat in categories:
    list_of_images = np.array(os.listdir(os.path.join(original_dataset_dir,cat)))
    print("{}: {} files".format(cat, len(list_of_images)))
    indexes = dict()
    # indexes['validation'] = sorted(np.random.choice(len(list_of_images), size=100, replace=False))
    # YUDI: hapus kode di bawah dan uncomment di atas
    indexes['validation'] = sorted(np.random.choice(len(list_of_images), size=1, replace=False))
    indexes['train'] = list(set(range(len(list_of_images))) - set(indexes['validation']))
    for phase in str_train_val:
        for i, fname in enumerate(list_of_images[indexes[phase]]):
            source = os.path.join(  original_dataset_dir, cat, fname)
            destination = os.path.join(base_dir, phase, cat, str(i)+".jpg")
            shutil.copyfile(source, destination)
        print("{}, {}: {} files copied".format(cat, phase, len(indexes[phase])))
        directories_dict[phase + "_" + cat + "_dir"] = os.path.join(base_dir, phase, cat)

directories_dict

print('Total training gambar daun sehat:', len(os.listdir(directories_dict['train_sehat_dir'])))
print('Total training gambar daun dengan hama:', len(os.listdir(directories_dict['train_hama_dir'])))
print("-"*32)
print('Total validation gambar daun sehat:', len(os.listdir(directories_dict['validation_sehat_dir'])))
print('Total validation gambar daun dengan hama:', len(os.listdir(directories_dict['validation_hama_dir'])))

"""Pembangunan Model"""

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json

train_datagen = ImageDataGenerator(
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    batch_size=32,
    class_mode='binary',
    target_size=(224,224))

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

validation_generator = validation_datagen.flow_from_directory(
    'data/validation',
    shuffle=False,
    class_mode='binary',
    target_size=(224,224))

conv_base = ResNet50(
    include_top=False,
    weights='imagenet')

for layer in conv_base.layers:
    layer.trainable = False

x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x) 
predictions = layers.Dense(2, activation='softmax')(x)
model = Model(conv_base.input, predictions)

optimizer = keras.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

history = model.fit_generator(generator=train_generator,
                              epochs=6,
                              validation_data=validation_generator)

model.save('keras/models/my_model.h5')
model.save_weights('keras/models/weights.h5')
with open('keras/models/keras.json', 'w') as f:
        f.write(model.to_json())

model = load_model('keras/models/my_model.h5')
with open('keras/models/keras.json') as f:
    model = model_from_json(f.read())
model.load_weights('keras/models/weights.h5')

"""Proses Validasi dan Prediksi"""

from PIL import Image
validation_img_paths = ["data/validation/sehat/1.jpg",
                        "data/validation/sehat/2.jpg",
                        "data/validation/hama/11.jpg",
                        "data/validation/hama/22.jpg"]
img_list = [Image.open(img_path) for img_path in validation_img_paths]

validation_batch = np.stack([preprocess_input(np.array(img.resize((224,224))))
                             for img in img_list])

pred_probs = model.predict(validation_batch)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
for i, img in enumerate(img_list):
    ax = axs[i]
    
    ax.axis('off')
    ax.set_title("{:.0f}% Dengan Hama, {:.0f}% Daun Sehat".format(100*pred_probs[i,0],
                                                            100*pred_probs[i,1]))
    ax.imshow(img)

"""# Bagian Baru"""
