'''
Tool to augment labeled data images for Model 2.
Takes an input of labeled .png files and outputs modified labeled .png files.
Author: Aneesha Ramaswamy
Usage: python model2_data_augmentation.py [input directory] [output directory]
'''
import os
import os.path as PATH
import sys
import random
import cv2
import numpy as np
import pandas as pd

def rotation(image):
    '''
    Function rotates image by +/- 1 degree.
    param image: scaled image to be rotated
    recommended rotation angle from dhSegment: 0.2
    '''
    # Randomly choose a rotation angle
    rotation_angle = 0
    while(rotation_angle == 0):
        rotation_angle = random.uniform(-1, 1)
    # grab the dimensions of the image
    (h, w) = image.shape
    # if the center is None, initialize it as the center of
    # the image
    center = (w // 2, h // 2)
    # perform the rotation
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    # return the rotated image
    return rotated

def scaling(image):
    '''
    Function scales image by +/- 0.2.
    param image: image to be resized
    scaling value recommended from dhSegment = 0.2
    '''
    # Randomly choose a scaling factor
    scaling_percent = 0
    while(scaling_percent == 0):
        scaling_percent = random.uniform(-0.2, 0.2)
    # If scaling_percent is negative, make image smaller
    if (scaling_percent < 0):
        width = int(image.shape[1] * (1 - (scaling_percent * -1)))
        height = int(image.shape[0] * (1 - (scaling_percent * -1)))
    # Otherwise make bigger
    else:
        width = int(image.shape[1] * (1 + scaling_percent))
        height = int(image.shape[0] * (1 + scaling_percent))
    dsize = (width, height)
    return cv2.resize(image, dsize)

def flip(src):
    '''
    Function flips images horizontally.
    param src: source image to be flipped
    '''
    return cv2.flip(src, 1)

# def main(argv):
# Pipeline 2: Read in Images + Labels
(train_labels, train_images) = np.load(file="model_data/train_labels.npy", allow_pickle=True), np.load(file="model_data/train_images.npy", allow_pickle=True)
(test_labels, test_images) = np.load(file="model_data/test_labels.npy", allow_pickle=True), np.load(file="model_data/test_images.npy", allow_pickle=True)
(val_labels, val_images) = np.load(file="model_data/val_labels.npy", allow_pickle=True), np.load(file="model_data/val_images.npy", allow_pickle=True)

train_aug = []
test_aug = []
val_aug = []

for image in train_images:
    augmented_img = flip(image)
    augmented_img = rotation(augmented_img.astype(np.uint8))
    train_aug.append(augmented_img)

for image in test_images:
    augmented_img = flip(image)
    augmented_img = rotation(augmented_img.astype(np.uint8))
    test_aug.append(augmented_img)

for image in val_images:
    augmented_img = flip(image)
    augmented_img = rotation(augmented_img.astype(np.uint8))
    val_aug.append(augmented_img)


train_aug = np.asarray(train_aug)
test_aug = np.asarray(test_aug)
val_aug = np.asarray(val_aug)

# Generate Labels

# Append Original and Augmented Arrays together

train_labels_combined = np.concatenate((train_labels, train_labels))
test_labels_combined = np.concatenate((test_labels, test_labels))
val_labels_combined = np.concatenate((val_labels, val_labels))

# Shuffle Indices
train_images_combined = np.concatenate((np.arange(train_images.shape[0]), np.arange(train_aug.shape[0], train_aug.shape[0]*2)))
test_images_combined = np.concatenate((np.arange(test_images.shape[0]), np.arange(test_aug.shape[0], test_aug.shape[0]*2)))
val_images_combined = np.concatenate((np.arange(val_images.shape[0]), np.arange(val_aug.shape[0], val_aug.shape[0]*2)))

train_df = pd.DataFrame({"Labels":train_labels_combined, "Images":train_images_combined})
test_df = pd.DataFrame({"Labels":test_labels_combined, "Images":test_images_combined})
val_df = pd.DataFrame({"Labels":val_labels_combined, "Images":val_images_combined})

train_df = train_df.reindex(np.random.permutation(train_df.index))
test_df = test_df.reindex(np.random.permutation(test_df.index))
val_df = val_df.reindex(np.random.permutation(val_df.index))

train_labels = train_df[["Labels"]].to_numpy().flatten()
test_labels = test_df[["Labels"]].to_numpy().flatten()
val_labels = val_df[["Labels"]].to_numpy().flatten()

train_images_index = train_df[["Images"]].to_numpy().flatten()
test_images_index = test_df[["Images"]].to_numpy().flatten()
val_images_index = val_df[["Images"]].to_numpy().flatten()

random_train_images = []
random_test_images = []
random_val_images = []

for index in train_images_index:
    print("TRAIN IMAGE INDEX:", index)
    if index > train_images.shape[0] - 1:
        random_train_images.append(train_images[index-train_images.shape[0]])
    else:
        random_train_images.append(train_images[index])

for index in test_images_index:
    print("TEST IMAGE INDEX:", index)
    if index > test_images.shape[0] - 1:
        random_test_images.append(test_images[index-test_images.shape[0]])
    else:
        random_test_images.append(test_images[index])

for index in val_images_index:
    print("VAL IMAGE INDEX:", index)
    if index > val_images.shape[0] - 1:
        random_val_images.append(val_images[index-val_images.shape[0]])
    else:
        random_val_images.append(val_images[index])


random_train_images = np.asarray(random_train_images)
random_test_images = np.asarray(random_test_images)
random_val_images = np.asarray(random_val_images)

print("TRAIN IMAGES:", random_train_images.shape)
print("TEST IMAGES:", random_test_images.shape)
print("VAL IMAGES:", random_val_images.shape)

print("TRAIN LABELS:", train_labels.shape)
print("TEST LABELS:", test_labels.shape)
print("VAL LABELS:", val_labels.shape)


np.save(file="model_data/train_images.npy", arr=random_train_images) # Model 2 Training
np.save(file="model_data/test_images.npy", arr=random_test_images) # Model 2 Testing
np.save(file="model_data/val_images.npy", arr=random_val_images) # Inference

np.save(file="model_data/train_labels.npy", arr=train_labels) # Model 2 Training
np.save(file="model_data/test_labels.npy", arr=test_labels) # Model 2 Testing
np.save(file="model_data/val_labels.npy", arr=val_labels) # Inference

'''    
# Pipeline 1: Read in Images from Directory
# Get all files to be augmented
for file in os.listdir(sys.argv[1]):
# Found file
name, ext = PATH.splitext(file)
print("Found " + file)
# Apply data augmentation
input_file_path = PATH.join(sys.argv[1], file)

src = cv2.imread(input_file_path, cv2.IMREAD_UNCHANGED)
new_image = flip(src)
new_image = rotation(new_image)
new_image = scaling(new_image)
# Write new image to file 
output_file_path = PATH.join(sys.argv[2], name + "_augmented.png")
cv2.imwrite(output_file_path, new_image)
'''

'''
if __name__ == "__main__":
  main(sys.argv[1:])
'''