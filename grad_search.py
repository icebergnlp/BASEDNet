# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from glob import glob
from imageio import imread, imsave
import os
from tqdm import tqdm

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from dh_segment.post_processing import find_lines
from dh_segment.inference import LoadedModel
from dh_segment.post_processing import binarization, line_vectorization
from dh_segment.io import PAGE
from tensorflow.python.keras.backend import update

# tf.compat.v1.disable_eager_execution() # need to disable eager in TF2.x (i.e. if you are in TF2.x, uncomment this)

test_labels, test_images = np.load(file="model_data/test_labels.npy"), np.load(file="model_data/test_images.npy")
test_files = np.load(file="model_data/test_files.npy")

print("****************************************")
print("TEST IMAGES SHAPE:", test_images.shape)
print("TEST LABELS SHAPE:", test_labels.shape)
# print("TEST[2]:", test_files[2])
print("****************************************")

def page_make_binary_mask(probs: np.ndarray, threshold: float=-1) -> np.ndarray:
    """
    Computes the binary mask of the detected Page from the probabilities outputed by network
    :param probs: array with values in range [0, 1]
    :param threshold: threshold between [0 and 1], if negative Otsu's adaptive threshold will be used
    :return: binary mask
    """

    mask = binarization.cleaning_probs(probs, 1.5)
    mask = binarization.hysteresis_thresholding(mask, 0.2, 0.4)
    return mask


def modelOneScore(matrix, probs):
    # matrix: binary matrix
    # probs: matrix of probabilities from model 1
    #print("MATRIX:", matrix)
    #print("TYPE:", type(matrix))
    #print("MATRIX SHAPE:", matrix.shape)
    #print("PROBS SHAPE:", probs.shape)
    # print("UNIQUE VALS:", np.unique(matrix))

    '''
    sum (loggedProbs @ 1 indices) + (1 - logged probs @ 0 indices)

    theta_ij = probability @ i, j
    y_ij = {0, 1} binary pixel value @ i, j

    log(theta_ij)**y_ij  *  (1 - theta_ij)**(1 - y_ij)

    sum for all i, j ^^^
    '''

    score_sum = 0

    if type(matrix) is np.ndarray:
        for i in range(1000):
            for j in range(1000):
                score_sum += (np.log(probs[i][j])**matrix[i][j]) * ((1 - probs[i][j])**(1 - matrix[i][j]))
    else:
        score_sum = tf.reduce_sum(tf.math.multiply(tf.math.pow(tf.math.log(probs), matrix), tf.math.pow(tf.subtract(probs,1.0),tf.subtract(1.0, matrix))))

    return score_sum
    
    # return np.sum((np.log(probs)**matrix) * ((1 - probs)**(1 - matrix)))
    


with tf.compat.v1.Session() as sess: # For use with TF2.x
    based_model = keras.models.load_model("saved_models/based_cnn_v1.h5")

    '''
    based_model.summary()
    
    test_loss, test_acc = based_model.evaluate(test_images, test_labels, batch_size=1, verbose=1)

    print("***********************************")
    print("TEST ACCURACY:",test_acc)
    print("TEST LOSS:", test_loss)
    print("***********************************")
    exit()
    '''

    '''
    # Execute when trying to reach benchmark test_acc value (ex: test_acc >= 0.98)
    satisfied = False
    
    while not satisfied:
        if test_acc >= 0.97:
            satisfied = True
            break
        else:
            exec(open("train.py").read())
            exec(open("inference.py").read())

    print("HIGHEST TEST ACC:", test_acc)
    '''

    '''
    print()
    print("***********************************")
    print("INCORRECT MODEL PREDICTIONS")
    print("***********************************")

    bad_y_misclass_count = 0
    good_y_misclass_count = 0
    misclassified_indices = []

    for index, image in enumerate(test_images):
        prediction = based_model.predict(np.expand_dims(image, axis=0), batch_size=1)[0][0]
        
        if prediction < 0.5:
            prediction = math.floor(prediction)
        else:
            prediction = math.ceil(prediction)
        
        if prediction != test_labels[index]:
            print("MISCLASSIFIED IMAGE: test_images[" + str(index) + "]")
            print("MISCLASSIFIED LABEL:", prediction)
            print("ACTUAL LABEL:", test_labels[index])

            misclassified_indices.append(index)
            
            if prediction == 0:
                good_y_misclass_count += 1
            else:
                bad_y_misclass_count += 1
    
    print("***********************************")
    print("MISCLASSIFICATION SUMMARY")
    print("***********************************")
    print("BAD Y MISCLASSIFIED COUNT:", bad_y_misclass_count)
    print("GOOD Y MISCLASSIFIED COUNT:", good_y_misclass_count)
    '''
    dh_model = LoadedModel("page_model4/export", predict_mode='image')

    filename = "/home/jonzamora/Desktop/based/input/0001 103.jpg"
    orig_image = cv2.imread(filename, cv2.IMREAD_COLOR)
    orig_shape = orig_image.shape[:2]
    curr_image = cv2.resize(orig_image, dsize=(1000, 1000), interpolation=cv2.INTER_AREA)
    prediction_outputs = dh_model.predict(input_tensor=curr_image)
    probs = prediction_outputs['probs'][0]
    original_shape = prediction_outputs['original_shape']
    probs = probs[:, :, 1]  # Take only class '1' (class 0 is the background, class 1 is the page)
    probs = probs / np.max(probs)  # Normalize to be in [0, 1]
    page_bin = page_make_binary_mask(probs)
    updated_matrix = np.copy(page_bin)
    # print("Y MATRIX SHAPE:", updated_matrix.shape)
    # print("PROBS MATRIX SHAPE:", probs.shape)    
    plt.imsave("updated_matrix4.png", updated_matrix, cmap="binary")
    
    m1 = modelOneScore(updated_matrix, probs)
    m1_orig = m1
    print("M1:", m1)
    
    Y = np.expand_dims(updated_matrix, axis=0) # SHAPE: (1, 1000, 1000)
    inp = tf.compat.v1.placeholder(dtype=tf.float32, shape=Y.shape, name="inp")
    p = tf.compat.v1.placeholder(dtype=tf.float32, shape=probs.shape, name="p")

    with tf.GradientTape() as tape:
        tape.watch(inp)
        x = modelOneScore(tf.reshape(inp, shape=[1000, 1000]), p)
        y = based_model(inp, training=False)[0][0]
        z = tf.math.add(tf.math.multiply(tf.constant(0.975), x), tf.math.multiply(tf.constant(0.025), tf.math.log(y)))

    grad = tape.gradient(target=z, sources=inp)

    print("MAX GRAD VALUE:", sess.run(tf.reduce_max(grad), feed_dict={inp: Y, p: probs}))
    print("MIN GRAD VALUE:", sess.run(tf.reduce_min(grad), feed_dict={inp: Y, p: probs}))
    print("***********************************")

    grad_val = sess.run(grad, feed_dict={inp: Y, p: probs})
    print("x:", sess.run(x, feed_dict={inp: Y, p: probs}))
    print("y:", sess.run(y, feed_dict={inp: Y, p: probs}))
    print("z:", sess.run(z, feed_dict={inp: Y, p: probs}))
    
    in_before = plt.imsave("search_output/in_before.png", np.squeeze(Y, axis=0), cmap="binary")
    map_before = plt.imsave("search_output/map_before.png", np.squeeze(grad_val, axis=0), cmap="bwr")

    initial_prediction = based_model.predict(Y, batch_size=1)[0][0]


    epsilon = 13

    for iter in range(10):
        print("ITER:", iter + 1)
        grad_val = sess.run(grad, feed_dict={inp: Y, p: probs})
        # Y = epsilon * np.sign(grad_val)
        # Y = np.clip((Y + epsilon * np.sign(grad_val)), 0, 1)
        Y = np.clip((Y + epsilon * grad_val), 0, 1)
        
        
        #lines = find_lines(lines_mask=np.squeeze(np.around(Y)))
        #image = np.full((1000, 1000, 3), (255, 255, 255)).astype(np.uint8)
        #Y = cv2.polylines(image, lines, isClosed=False, color=(0, 0, 0), thickness=3)
        #Y = cv2.cvtColor(Y, cv2.COLOR_BGR2GRAY)
        #Y = cv2.threshold(Y,127,1,cv2.THRESH_BINARY_INV)[1]
       
        Y = np.squeeze(Y, axis=0)
        Y = cv2.threshold(Y, 0.5, 1, cv2.THRESH_BINARY)[1]
        #print("Y SHAPE 1:", Y.shape) # (1000, 1000)
        #print("UNIQUE:", np.unique(Y))
        m1 = modelOneScore(Y, probs)
        Y = np.expand_dims(Y, axis=0)
        #print("Y SHAPE 2:", Y.shape) # (1, 1000, 1000)
        m2 = based_model.predict(Y, batch_size=1)[0][0]
        curr_score = sess.run(z, feed_dict={inp: Y, p: probs})
        print("z:", curr_score, "epsilon:", epsilon)

        print()
    


    Y = np.around(Y)
    new_prediction = based_model.predict(Y, batch_size=1)[0][0]
    print("INITIAL M2 SCORE:", initial_prediction)
    print("CURRENT M2 SCORE (EXACT):", new_prediction)

    if new_prediction < 0.5:
        new_prediction = math.floor(new_prediction)
    else:
        new_prediction = math.ceil(new_prediction)

    print("CURRENT M2 SCORE (ROUNDED):", new_prediction)
    print()
    print("INITIAL M1 SCORE:", m1_orig)
    print("CURRENT M1 SCORE:", m1)

    in_after = plt.imsave("search_output/in_after.png", np.squeeze(Y), cmap="binary")
    map_after = plt.imsave("search_output/map_after.png", np.squeeze(grad_val, axis=0), cmap="bwr")


    '''
    PAGE XML POST PROCESSING
    '''

    
    PAGE_XML_DIR = './page_xml'
    output_dir = 'search/output'
    os.makedirs(output_dir, exist_ok=True)
    # PAGE XML format output
    output_pagexml_dir = os.path.join(output_dir, PAGE_XML_DIR)
    os.makedirs(output_pagexml_dir, exist_ok=True)

    # POST PROCESSING
    # Upscale to have full resolution image (cv2 uses (w,h) and not (h,w) for giving shapes)
    bin_upscaled = cv2.resize(page_bin.astype(np.uint8, copy=False),
                                      tuple(orig_shape[::-1]), interpolation=cv2.INTER_NEAREST)
    updated_upscaled = cv2.resize(np.squeeze(Y).astype(np.uint8, copy=False),
                                      tuple(orig_shape[::-1]), interpolation=cv2.INTER_NEAREST)

    # Find the longest central line for each connected component in the given binary mask
    pred_page_coords = line_vectorization.find_lines(bin_upscaled.astype(np.uint8, copy=False))
    updated_pred_page_coords = line_vectorization.find_lines(updated_upscaled.astype(np.uint8, copy=False))

    # Draw page box on original image and export it. Add also box coordinates to the txt file
    if pred_page_coords is not None:
        for line in pred_page_coords:
            cv2.polylines(orig_image, line, True, (0, 0, 255), thickness=5)

        # Create textlines and XML file
        text_region = PAGE.TextRegion()
        text_region.text_lines = [PAGE.TextLine.from_array(baseline_coords=line) for line in pred_page_coords]
    else:
        print('No box found')
        text_region = PAGE.TextRegion()
    
    basename = os.path.basename(filename).split('.')[0]
    imsave(os.path.join(output_dir, '{}_boxes.jpg'.format(basename)), orig_image)

    # for new matrix
    if updated_pred_page_coords is not None:
        for line in updated_pred_page_coords:
            cv2.polylines(Y, line, True, (0, 0, 255), thickness=5)
        # Write corners points into a .txt file
        # txt_coordinates += '{},{}\n'.format(filename, format_quad_to_string(pred_page_coords))

        # Create page region and XML file
        # page_border = PAGE.Border(coords=PAGE.Point.cv2_to_point_list(pred_page_coords[:, None, :]))

        # Create textlines and XML file
        text_region = PAGE.TextRegion()
        text_region.text_lines = [PAGE.TextLine.from_array(baseline_coords=line) for line in updated_pred_page_coords]
    else:
        print('No box found in {}'.format(filename))
        text_region = PAGE.TextRegion()
        #page_border = PAGE.Border()
    
    basename = os.path.basename(filename).split('.')[0]
    #imsave(os.path.join(output_dir, '{}_updated_boxes.jpg'.format(basename)), np.squeeze(Y))

    page_xml = PAGE.Page(image_filename=filename, image_width=original_shape[1], image_height=original_shape[0] , text_regions=[text_region])
    xml_filename = os.path.join(output_pagexml_dir, '{}.xml'.format(basename))
    page_xml.write_to_file(xml_filename, creator_name='PageExtractor')