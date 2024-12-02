import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from keras.callbacks import EarlyStopping
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2, EfficientNetV2L, Xception
import numpy as np
import random
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix

# Set random seed for reproducibility
def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

seed = 101
set_seed(seed)

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Directories
train_dir = "/home/kaga/Desktop/mingyang/dataset/archive (2)/RiceLeafsv3/train"
val_dir = "/home/kaga/Desktop/mingyang/dataset/archive (2)/RiceLeafsv3/validation"
test_dir = "/home/kaga/Desktop/mingyang/dataset/archive (2)/RiceLeafsv3/validation"

labels = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot']

# Data augmentation
datagen_train = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    horizontal_flip=True
)
datagen_val = ImageDataGenerator(rescale=1./255)
datagen_test = ImageDataGenerator(rescale=1./255)

batch_size = 4
input_shape = (299, 299, 3)

generator_train = datagen_train.flow_from_directory(
    directory=train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    shuffle=True
)
generator_val = datagen_val.flow_from_directory(
    directory=val_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    shuffle=False
)
generator_test = datagen_test.flow_from_directory(
    directory=test_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    shuffle=False
)

# Define base models
def build_model(base_model, num_classes):
    base_model.trainable = False
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.6)(x)
    x = Dense(512, activation="PReLU", kernel_regularizer=regularizers.l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="PReLU", kernel_regularizer=regularizers.l2(0.00001))(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation="PReLU", kernel_regularizer=regularizers.l2(0.00001))(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model

num_classes = len(labels)

modelA = build_model(InceptionResNetV2(include_top=False, input_shape=input_shape, weights='imagenet'), num_classes)
modelB = build_model(EfficientNetV2L(include_top=False, input_shape=input_shape, weights='imagenet'), num_classes)
modelC = build_model(Xception(include_top=False, input_shape=input_shape, weights='imagenet'), num_classes)

# Ensemble model
models = [modelA, modelB, modelC]
model_input = tf.keras.Input(shape=input_shape)
model_outputs = [model(model_input) for model in models]
ensemble_output = tf.keras.layers.Average()(model_outputs)  # Simple averaging
ensemble_model = tf.keras.models.Model(inputs=model_input, outputs=ensemble_output)

# Metrics
def Recall(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

def Precision(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())

def F1(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

def Specificity(y_true, y_pred):
    true_negatives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())

# Compile the ensemble model
ensemble_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision, Recall, F1, Specificity]
)

# Training
earlystopping = EarlyStopping(monitor="val_loss", mode="min", patience=8, restore_best_weights=True)
steps_per_epoch = generator_train.n // batch_size
validation_steps = generator_val.n // batch_size

history = ensemble_model.fit(
    generator_train,
    validation_data=generator_val,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=50,
    callbacks=[earlystopping]
)

# Visualization: Confusion Matrix
Y_pred = ensemble_model.predict(generator_test)
y_pred = np.argmax(Y_pred, axis=1)
conf_matrix = confusion_matrix(generator_test.classes, y_pred)
df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
plt.figure(figsize=(10, 10))
sn.heatmap(df_cm, annot=True, cmap="Blues", fmt="d")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

# Metrics visualization
metrics = ['accuracy', 'loss', 'Precision', 'Recall', 'F1']
for metric in metrics:
    plt.plot(history.history[metric], label=f'Train {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
    plt.title(metric.capitalize())
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    plt.show()
