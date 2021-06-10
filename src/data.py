import os
import os.path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from PIL import Image, ImageOps

######################################################
##################### LABELS #########################
######################################################

def generate_labels(curr_dir, label_list, file_list):
    curr_dir_files = np.asarray(os.listdir(curr_dir))
    curr_dir_length = curr_dir_files.size

    for file in curr_dir_files:
        file_list.append(file)

    if "good" in curr_dir:
        label_list.append(np.ones(shape=(curr_dir_length)))
    elif "bad" in curr_dir:
        label_list.append(np.zeros(shape=(curr_dir_length)))

train_good_dir = os.getcwd() + "/merged_dataset/train/labeled_good/"
test_good_dir = os.getcwd() + "/merged_dataset/test/labeled_good/"
val_good_dir = os.getcwd() + "/merged_dataset/val/labeled_good/"

train_bad_dir = os.getcwd() + "/merged_dataset/train/labeled_bad/"
test_bad_dir = os.getcwd() + "/merged_dataset/test/labeled_bad/"
val_bad_dir = os.getcwd() + "/merged_dataset/val/labeled_bad/"

label_dirs = [train_bad_dir, train_good_dir, test_bad_dir, test_good_dir, val_bad_dir, val_good_dir]
image_dirs = [train_bad_dir, train_good_dir, test_bad_dir, test_good_dir, val_bad_dir, val_good_dir]
train_labels = []
train_files = []
test_labels = []
test_files = []
val_labels = []
val_files = []

for dir in label_dirs:
    if "train" in dir:
        generate_labels(dir, train_labels, train_files)
    elif "test" in dir:
        generate_labels(dir, test_labels, test_files)
    elif "val" in dir:
        generate_labels(dir, val_labels, val_files)
    else:
        print("No directory with 'train', 'test', or 'val' images to label")

# Convert lists to NumPy Arrays
train_labels = np.asarray(train_labels)
train_labels = np.reshape(train_labels, train_labels.size)
train_files = np.asarray(train_files)

test_labels = np.asarray(test_labels)
test_labels = np.reshape(test_labels, test_labels.size)
test_files = np.asarray(test_files)

val_labels = np.asarray(val_labels)
val_labels = np.reshape(val_labels, val_labels.size)
val_files = np.asarray(val_files)

# Create (file, label) pairs in NumPy Array

train_labels = np.stack((train_files, train_labels), axis=-1)
test_labels = np.stack((test_files, test_labels), axis=-1)
val_labels = np.stack((val_files, val_labels), axis=-1)

# new_labels contain only the label columns and omit the file columns
new_train_labels = train_labels[:,1].astype("float64")
new_test_labels = test_labels[:,1].astype("float64")
new_val_labels = val_labels[:,1].astype("float64")

np.save(file="model_data/train_labels.npy", arr=new_train_labels)
np.save(file="model_data/test_labels.npy", arr=new_test_labels)
np.save(file="model_data/val_labels.npy", arr=new_val_labels)

np.save(file="model_data/train_files.npy", arr=train_labels)
np.save(file="model_data/test_files.npy", arr=test_labels)
np.save(file="model_data/val_files.npy", arr=val_labels)


######################################################
##################### IMAGES #########################
######################################################

def read_image(image_list, image_path):
    print("CURRENT IMAGE:", image_path)
    curr_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    curr_image = curr_image / curr_image.max()
    curr_image = curr_image.astype("int")
    
    '''
    print("ORIGINAL SHAPE:", curr_image.shape)
    curr_image = Image.fromarray((curr_image * 255).astype(np.uint8))
    orig_size = curr_image.size
    ratio = float(1000)/max(orig_size)
    new_size = tuple([int(x*ratio) for x in orig_size])
    curr_image = curr_image.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (1000, 1000))
    new_im.paste(curr_image, ((1000-new_size[0])//2, (1000-new_size[1])//2))
    
    new_im = np.array(new_im)
    # Convert RGB to BGR 
    new_im = new_im[:, :, ::-1].copy()
    new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY)
    new_im = new_im / new_im.max()
    new_im = new_im.astype("int")
    '''
    print("NEW SHAPE:", curr_image.shape)
    print("UNIQUE VALS:", np.unique(curr_image))

    image_list.append(curr_image)

train_images = []
test_images = []
val_images = []

# TODO: Randomize the Order

for dir in image_dirs:
    for image in os.listdir(dir):
        if ("train" in dir and "bad" in dir):
            read_image(train_images, (train_bad_dir + image))
        elif ("train" in dir and "good" in dir):
            read_image(train_images, (train_good_dir + image))
        elif ("test" in dir and "bad" in dir):
            read_image(test_images, (test_bad_dir + image))
        elif ("test" in dir and "good" in dir):
            read_image(test_images, (test_good_dir + image))
        elif ("val" in dir and "bad" in dir):
            read_image(val_images, (val_bad_dir + image))
        elif ("val" in dir and "good" in dir):
            read_image(val_images, (val_good_dir + image))
        else:
            print("No Directory with 'train', 'test', or 'val' was Found")

# Convert lists to NumPy Arrays for train_images, test_images, and val_images
train_images = np.asarray(train_images)
test_images = np.asarray(test_images)
val_images = np.asarray(val_images)

####################################
# Resize and Preserve Aspect Ratio #
####################################
'''
def resize(im, desired_size):
    print("ORIGINAL SHAPE:", im.shape)
    im = Image.fromarray((im * 255).astype(np.uint8))
    orig_size = im.size
    ratio = float(desired_size)/max(orig_size)
    new_size = tuple([int(x*ratio) for x in orig_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))

    new_im = np.array(new_im)
    # Convert RGB to BGR 
    new_im = new_im[:, :, ::-1].copy()
    new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY)
    new_im = new_im / new_im.max()
    new_im = new_im.astype("int")
    print("NEW SHAPE:", new_im.shape)
    print("UNIQUE VALS:", np.unique(new_im))
    return new_im
'''
'''
resized_train = []
resized_test = []
resized_val = []

for image in train_images:
    resize(image, 1000, resized_train)

for image in test_images:
    resize(image, 1000, resized_test)

for image in val_images:
    resize(image, 1000, resized_val)

resized_train = np.asarray(resized_train)
resized_test = np.asarray(resized_test)
resized_val = np.asarray(resized_val)

print('RESIZED TRAIN', resized_train.shape)
print('RESIZED TEST', resized_test.shape)
print('RESIZED VAL', resized_val.shape)
'''
# Save each NumPy Array as its own file
np.save(file="model_data/train_images.npy", arr=train_images) # Model 2 Training
np.save(file="model_data/test_images.npy", arr=test_images) # Model 2 Testing
np.save(file="model_data/val_images.npy", arr=val_images) # Inference