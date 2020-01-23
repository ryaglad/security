import keras

# import keras
import matplotlib.pyplot as plt
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())


model_path = os.path.join('..', 'snapshots', 'resnet50_csv_09.h5')
model = models.load_model(model_path, backbone_name='resnet50')
print(model.summary())

model = models.convert_model(model)

# load label to names mapping for visualization purposes
labels_to_names = {0: 'bullet', 1: 'knife', 2: 'container', 3: 'gun', 4: 'screwdriver'}


# load image
#image = read_image_bgr('000000008021.jpg')
#image = read_image_bgr('/Users/ScottsPC/Desktop/images/bear.jpg')


def predict(image_path):
    #image = read_image_bgr('/
    image = read_image_bgr(image_path)
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    #boxes, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    #print(labels)

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break
            
        color = label_color(label)
        
        b = box.astype(int)
        draw_box(draw, b, color=color)
        
        caption = "{} {:.2f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
    print(label)
    print(score)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()

image = '/Users/ryans/Desktop/images.jpg'

predict(image)


