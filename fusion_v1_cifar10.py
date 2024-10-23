import tensorflow as tf
from tensorflow.keras import layers, Model
import cv2  
import numpy as np
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Define a maximum number of keypoints for ORB descriptors
MAX_KEYPOINTS = 500  

# MobileNetV2 backbone (pre-trained on ImageNet)
backbone = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                             include_top=False, 
                                             weights='imagenet')


class MobileDetFusionModel(Model):
    def __init__(self):
        super(MobileDetFusionModel, self).__init__()
        self.backbone = backbone
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense_1 = layers.Dense(128, activation='relu')
        self.class_output = layers.Dense(10, activation='softmax', name='class_output') 

    def classification_head(self, input_features):
        x = self.global_avg_pool(input_features)
        x = self.dense_1(x)
        class_output = self.class_output(x)
        return class_output

    def call(self, inputs):

        neural_features = self.backbone(inputs)

        orb_features_batch = tf.map_fn(lambda img: extract_orb_tf(img), inputs, dtype=tf.float32)

        # Fuse neural and non-neural features
        fused_features = fusion_layer(neural_features, orb_features_batch)

        # Pass fused features to the classification head
        output = self.classification_head(fused_features)
        return output


def extract_orb_features(image):

    image = image.numpy()  

    # ORB feature extraction
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)

    if descriptors is None:
        descriptors = np.zeros((1, 32))  

    # Pad or truncate descriptors to the maximum number of keypoints
    if descriptors.shape[0] > MAX_KEYPOINTS:
        descriptors = descriptors[:MAX_KEYPOINTS, :]  
    else:
        # Pad with zeros to match the max keypoints
        pad_size = MAX_KEYPOINTS - descriptors.shape[0]
        descriptors = np.pad(descriptors, ((0, pad_size), (0, 0)), 'constant')


    descriptors = descriptors.reshape((1, 1, MAX_KEYPOINTS, 32))  
    return descriptors


def extract_orb_tf(image):
    orb_features = tf.py_function(func=extract_orb_features, inp=[image], Tout=tf.float32)
    orb_features.set_shape([1, 1, MAX_KEYPOINTS, 32])  
    return orb_features


def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  
    image = image / 255.0  
    return image


def fusion_layer(neural_features, orb_features):

    orb_features = tf.squeeze(orb_features, axis=1)  
    

    orb_features_resized = tf.image.resize(orb_features, 
                                           (tf.shape(neural_features)[1], tf.shape(neural_features)[2])) 
    
    # Now concatenate along the channel dimension
    fused_features = tf.concat([neural_features, orb_features_resized], axis=-1)  
    return fused_features


def load_cifar10_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Resize images to 224x224 and normalize
    x_train_resized = np.array([preprocess_image(image) for image in x_train])
    x_test_resized = np.array([preprocess_image(image) for image in x_test])

    return (x_train_resized, y_train), (x_test_resized, y_test)

# Compile and train the model
if __name__ == "__main__":
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = load_cifar10_dataset()

    # Initialize the model
    model = MobileDetFusionModel()

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc}")
