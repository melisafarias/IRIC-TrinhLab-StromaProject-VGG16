from email.mime import application
import csv
import pandas as pd
import tensorflow.compat.v2 as tf
import cv2
import numpy as np
import os
import shutil
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import tkinter as TK
from tkinter.filedialog import askdirectory, askopenfile
import pathlib as pl
from pathlib import Path
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, Dropout, MaxPool2D
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import Model, layers, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend
from keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
#from google.colab.patches import cv2_imshow


# Set some Variables for the Encoding process 
num_classes=4 
img_size=224
img_height = 224
img_width = 224
batch_size = 2
saved_model = load_model("VGG16_Stromal_Training.h5")
# Set the path to the folder containing the image tiles
folder_path = askdirectory(title = "Select a File")
#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pl.Path(folder_path).with_suffix('')
print(data_dir)
#train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split = 0.2, subset = "training", seed = 123, image_size = (img_height, img_width), batch_size = batch_size)
#print(train_ds)
image_input = Input(shape = (224,224,3))
print(image_input)
#image_input = Input(shape = )
#print(image_input)
print("VGG16\n")
Top = False
folder_path2 = askdirectory(title = "Select a File")
for i in range(2):
    if i >= 1:
        Top = True
    #model = VGG16(input_tensor = image_input, include_top = Top, weights= "imagenet", classifier_activation = "softmax")
    print("Summary\n")
    #model.summary()
    saved_model.summary()
    pixels_features = []

    for each in os.listdir(folder_path):
        path = os.path.join(folder_path,each)
        img = image.load_img(path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        feature = saved_model.predict(img_data)
        pixels_features.append(feature)

    Tiles_df = pd.DataFrame(np.array(pixels_features).reshape(-1,len(pixels_features)))
    print(Tiles_df)

    #Saves the csv to the specified path (Change name when required until I can make a for loop to check if the name already exists)
    if i < 1:
        Tiles_df.to_csv(folder_path2 + "/" + "Test_tiles-FCL-10.csv")
    else:
        Tiles_df.to_csv(folder_path2 + "/" + "Test_tiles-FLI-10.csv")

WEIGHTS_PATH = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
)
WEIGHTS_PATH_NO_TOP = (
    "https://storage.googleapis.com/tensorflow/"
    "keras-applications/vgg16/"
    "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
)

layers = VersionAwareLayers()


@keras_export("keras.applications.vgg16.VGG16", "keras.applications.VGG16")
def VGG16(
    include_top = True,
    weights = "imagenet",
    input_tensor = None,
    input_shape = None,
    pooling = None,
    classes = 2,
    classifier_activation = "softmax",
):
    """Instantiates the VGG16 model.

    Reference:
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)

    For image classification use cases, see
    [this page for detailed examples](
      https://keras.io/api/applications/#usage-examples-for-image-classification-models).

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](
      https://keras.io/guides/transfer_learning/).

    The default input size for this model is 224x224.

    Note: each Keras Application expects a specific kind of input preprocessing.
    For VGG16, call `tf.keras.applications.vgg16.preprocess_input` on your
    inputs before passing them to the model.
    `vgg16.preprocess_input` will convert the input images from RGB to BGR,
    then will zero-center each color channel with respect to the ImageNet
    dataset, without scaling.

    Args:
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer.  When loading pretrained weights, `classifier_activation` can
            only be `None` or `"softmax"`.

    Returns:
      A `keras.Model` instance.
    """
    if not (weights in {"imagenet", None} or tf.io.gfile.exists(weights)):
        raise ValueError(
            "The `weights` argument should be either "
            "`None` (random initialization), `imagenet` "
            "(pre-training on ImageNet), "
            "or the path to the weights file to be loaded.  Received: "
            f"weights={weights}"
        )

    if weights == "imagenet" and include_top and classes != 1000:
        raise ValueError(
            'If using `weights` as `"imagenet"` with `include_top` '
            "as true, `classes` should be 1000.  "
            f"Received `classes={classes}`"
        )
    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights,
    )

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    if i >=0 :
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(img_input)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

        # Block 2
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(x)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

        # Block 3
        x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
        x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
        x = layers.Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

        # Block 4
        x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
        x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(x)
        x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

        # Block 5
        x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(x)
        x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(x)
        x = layers.Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

        if include_top:
            # Classification block
            x = layers.Flatten(name="flatten")(x)
            x = layers.Dense(4096, activation="relu", name="fc1")(x)
            x = layers.Dense(4096, activation="relu", name="fc2")(x)

            imagenet_utils.validate_activation(classifier_activation, weights)
            x = layers.Dense(
                classes, activation=classifier_activation, name="predictions"
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D()(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = training.Model(inputs, x, name="vgg16")

    # Load weights.
    if weights == "imagenet":
        if include_top:
            weights_path = data_utils.get_file(
                "vgg16_weights_tf_dim_ordering_tf_kernels.h5",
                WEIGHTS_PATH,
                cache_subdir="models",
                file_hash="64373286793e3c8b2b4e3219cbf3544b",
            )
        else:
            weights_path = data_utils.get_file(
                "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
                WEIGHTS_PATH_NO_TOP,
                cache_subdir="models",
                file_hash="6d6bbae143d832006294945121d1f1fc",
            )
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


@keras_export("keras.applications.vgg16.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="caffe"
    )

def get_labels(file_path):
    import os 
    return tf.strings.split(file_path , os.path.sep)[-2]

@keras_export("keras.applications.vgg16.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)

def process_img(file_path):
    label = get_labels(file_path)
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [224,224])
    return img , label

"""preprocess_input.__doc__ = imagenet_utils.PREPROCESS_INPUT_DOC.format(
    mode="",
    ret=imagenet_utils.PREPROCESS_INPUT_RET_DOC_CAFFE,
    error=imagenet_utils.PREPROCESS_INPUT_ERROR_DOC,
)
decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__"""