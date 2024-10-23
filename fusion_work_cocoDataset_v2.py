import tensorflow as tf
from tensorflow.keras import layers, Model
import cv2
import numpy as np
from pycocotools.coco import COCO
import os


MAX_KEYPOINTS = 500  

# Path to COCO dataset annotations and images
COCO_ANNOTATIONS_PATH = '/home/aimslab/Documents/cocoDataset/coco/annotations/instances_train2017.json'
COCO_IMAGES_PATH = '/home/aimslab/Documents/cocoDataset/coco/images/train2017/train2017/'

def preprocess_labels(labels, num_predictions=49):


    preprocessed_labels = np.zeros((len(labels), num_predictions), dtype=np.int32)
    
    for i, label in enumerate(labels):

        valid_labels = np.clip(label, 0, 90)
        preprocessed_labels[i, :min(num_predictions, len(valid_labels))] = valid_labels[:num_predictions] 
    
    return preprocessed_labels

def load_coco_dataset(coco_annotations_path, coco_images_path, num_samples=1000):
    coco = COCO(coco_annotations_path)
    img_ids = coco.getImgIds()
    img_ids = img_ids[:num_samples]  
    
    images = []
    labels = []
    bboxes = []
    
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(coco_images_path, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            continue

        image = preprocess_image(image)
        images.append(image)
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        

        image_bboxes = []
        image_labels = []
        for ann in anns
            bbox = ann['bbox'] 
            image_bboxes.append(bbox)
            image_labels.append(ann['category_id'])  
        
        # Resize bounding boxes to match SSD Lite output shape
        bboxes.append(preprocess_bounding_boxes(np.array(image_bboxes)))
        labels.append(image_labels)
    
    # Convert to numpy arrays
    images = np.array(images)
    bboxes = np.array(bboxes)
    labels = preprocess_labels(labels)  

    return images, labels, bboxes

# MobileNetV2 backbone (pre-trained on ImageNet)
backbone = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                             include_top=False,
                                             weights='imagenet')
#########checking feature map##########
# Create a dummy input image
dummy_input = tf.random.normal([1, 224, 224, 3])

# Pass through the backbone
feature_map = backbone(dummy_input)
print("Feature map shape:", feature_map.shape)

##################end##################


class SSDLiteHead(layers.Layer):
    def __init__(self, num_classes=80):  # COCO has 80 classes
        super(SSDLiteHead, self).__init__()
        self.num_classes = num_classes

        # Depthwise separable convolutions for SSD Lite
        self.conv1 = layers.SeparableConv2D(256, kernel_size=3, padding='same', activation='relu')
        self.conv2 = layers.SeparableConv2D(256, kernel_size=3, padding='same', activation='relu')
        self.conv3 = layers.SeparableConv2D(256, kernel_size=3, padding='same', activation='relu')


        self.classification_head = layers.Conv2D(num_classes, kernel_size=1, activation='softmax', name='class_output')
        self.box_head = layers.Conv2D(4, kernel_size=1, name='box_output')

    def call(self, features):
        x = self.conv1(features)
        x = self.conv2(x)
        x = self.conv3(x)

        class_predictions = self.classification_head(x)
        box_predictions = self.box_head(x)

        # Flatten the predictions for compatibility with loss functions
        class_predictions = tf.reshape(class_predictions, [tf.shape(class_predictions)[0], -1, self.num_classes], name='class_output')
        box_predictions = tf.reshape(box_predictions, [tf.shape(box_predictions)[0], -1, 4], name='box_output')
        return {'class_output': class_predictions, 'box_output': box_predictions}


class MobileDetFusionModel(Model):
    def __init__(self, num_classes=80):
        super(MobileDetFusionModel, self).__init__()
        self.backbone = backbone
        self.ssdlite_head = SSDLiteHead(num_classes)
        
    def call(self, inputs):
        neural_features = self.backbone(inputs)
        orb_features_batch = tf.map_fn(lambda img: extract_orb_tf(img), inputs, dtype=tf.float32)
        fused_features = fusion_layer(neural_features, orb_features_batch)
        return self.ssdlite_head(fused_features)

# ORB feature extractor function (using OpenCV)
def extract_orb_features(image):
    image = image.numpy()
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)

    if descriptors is None:
        descriptors = np.zeros((1, 32))

    if descriptors.shape[0] > MAX_KEYPOINTS:
        descriptors = descriptors[:MAX_KEYPOINTS, :]
    else:
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
    orb_features_resized = tf.image.resize(orb_features, (tf.shape(neural_features)[1], tf.shape(neural_features)[2]))
    fused_features = tf.concat([neural_features, orb_features_resized], axis=-1)
    return fused_features

def preprocess_bounding_boxes(boxes, num_predictions=49):
    # Check if there are no bounding boxes for this image
    if len(boxes) == 0:
        # If no bounding boxes are present, fill with zeros or a default value
        reshaped_boxes = np.zeros((num_predictions, 4))
    else:
        reshaped_boxes = np.zeros((num_predictions, 4))
        reshaped_boxes[:min(len(boxes), num_predictions), :] = boxes[:min(len(boxes), num_predictions)]
    return reshaped_boxes


if __name__ == "__main__":
    # Load the COCO dataset
    images, labels, bboxes = load_coco_dataset(COCO_ANNOTATIONS_PATH, COCO_IMAGES_PATH)
    
    ####### labels check ###########
    
    coco = COCO(COCO_ANNOTATIONS_PATH)
    category_ids = [ann['category_id'] for ann in coco.loadAnns(coco.getAnnIds())]
    print(f"Unique category IDs: {set(category_ids)}")

    ########   end  ################

    # Initialize the model
    model = MobileDetFusionModel(num_classes=91)  # Adjusted to 81 to include background class

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={
            'class_output': 'sparse_categorical_crossentropy',
            'box_output': 'mean_squared_error'
        },
        metrics={'class_output': 'accuracy'}
    )

    # Train the model
    model.fit(
        images,
        {'class_output': labels, 'box_output': bboxes},
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(images, {'class_output': labels, 'box_output': bboxes})
    print(f"Test Accuracy: {test_acc}")
