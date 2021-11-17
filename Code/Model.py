# //model//

# //packages
import tensorflow as tf
from tensorflow.keras.models import Sequential

# //parameters
LR = 1e-3
DROPOUT = 0.5

def model_definition(pretrained = True):
    if pretrained = True:
        # // pretrained model - vgg16
        model = tf.keras.applications.vgg16.VGG16(
            include_top=True, weights='imagenet', input_tensor=None,
            input_shape=None, pooling=None, classes=1000,
            classifier_activation='softmax'
        )
        model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])
    else:
        # // CNN model
        model = Sequential([
            Conv2D(16, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation="relu"),
            BatchNormalization(),
            AveragePooling2D((2, 2)),
            Flatten(),
            Dense(400, activation="tanh"),
            Dropout(DROPOUT),
            BatchNormalization(),
            Dense(10, activation="softmax")
        ])
        model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

