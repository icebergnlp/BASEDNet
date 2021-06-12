'''
Tool for producing colored masks of ground truth data stored in the PAGE XML 
format. Each mask contains the top and bottom lines of each text line bounding 
box in the document image. Overlapping lines are colored as the combination of 
the top and bottom line colors.

Author: Jason Vega
Email: jasonvega14@yahoo.com
'''

import os
import sys
import argparse
import cv2
import numpy as np
import xml.etree.ElementTree as et
from math import atan2

PAGE_TAG = "ns0:Page"
TEXTREGION_TAG = "ns0:TextRegion"
TEXTLINE_TAG = "ns0:TextLine"
BASELINE_TAG = "ns0:Baseline"
COORDS_TAG = "ns0:Coords"
POINTS_ATTR = "points"
WIDTH_TAG = "imageWidth"
HEIGHT_TAG = "imageHeight"

NAMESPACE_GT = { # for https
    "ns0": "https://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
    "xsi": "https://www.w3.org/2001/XMLSchema-instance"
}

NAMESPACE_GT_2 = { # for http
    "ns0": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance"
}

COORD_DELIM = ","

THICKNESS = 3
BASELINE_COLOR = (255, 0, 0)
TOP_LINE_COLOR = (0, 255, 0)
BOTTOM_LINE_COLOR = (0, 0, 255)
LINES_CLOSED = True

CHANNELS = 3

QUAD_CORNERS = 4
LINE_POINTS = 2

TOP_KEY = "TOP"
BOTTOM_KEY = "BOTTOM"
BASELINE_KEY = "BASELINE"
NO_IMAGE = "NONE"
ORIGINAL_IMAGE = "ORIGINAL"
REDSCALE_IMAGE = "REDSCALE"

LINE_COLORS = {
    BASELINE_KEY: BASELINE_COLOR,
    TOP_KEY: TOP_LINE_COLOR,
    BOTTOM_KEY: BOTTOM_LINE_COLOR
}

'''
Returns a list of lines represented as lists of (x, y) coordinates defining 
their bounding boxes.

page: the XML page element to search for lines in.
namespace: the namespace used in the XML file.
'''
def get_line_coords(page, namespace):
    line_list = []

    for region in page.findall(TEXTREGION_TAG, namespace):
        for line in region.findall(TEXTLINE_TAG, namespace):
            coordsElement = line.find(COORDS_TAG, namespace)
            coords = coordsElement.get(POINTS_ATTR).split()

            # Convert each coordinate value from strings to integers
            for i in range(len(coords)):
                xy_coords = coords[i].split(COORD_DELIM)

                for j in range(len(xy_coords)):
                    xy_coords[j] = int(xy_coords[j])

                coords[i] = xy_coords

            line_list.append(coords)

    return line_list

'''
Returns a list of baselines, each represented as lists of (x, y) coordinates.

page: the XML page element to search for baselines in.
namespace: the namespace used in the XML file.
'''
def get_baseline_coords(page, namespace):
    baseline_list = []

    for region in page.findall(TEXTREGION_TAG, namespace):
        for line in region.findall(TEXTLINE_TAG, namespace):
            baseline_element = line.find(BASELINE_TAG, namespace)

            if baseline_element is None:
                continue

            coords = baseline_element.get(POINTS_ATTR).split()

            # Convert each coordinate value from strings to integers
            for i in range(len(coords)):
                xy_coords = coords[i].split(COORD_DELIM)

                for j in range(len(xy_coords)):
                    xy_coords[j] = int(xy_coords[j])

                coords[i] = xy_coords
        
            coords = sorted(coords, key=lambda k: [k[0], k[1]])
            baseline_list.append(coords)

    return np.array([np.array(xi) for xi in baseline_list])

'''
Given a list of coordinates defining a bounding box, return the four corners
in clockwise orientation.

line_coords: the list of coordiantes defining the bounding box.
centroid: the average (x, y) point of the coordinates.
'''
def get_corners(line_coords, centroid):
    if len(line_coords) == QUAD_CORNERS:
        x_sorted = line_coords[line_coords[:,0].argsort()]
        left_y = x_sorted[:LINE_POINTS]
        right_y = x_sorted[LINE_POINTS:]
        left_y_sorted = left_y[left_y[:,1].argsort()]
        right_y_sorted = right_y[right_y[:,1].argsort()]

        top_left_corner = left_y_sorted[0]
        bottom_left_corner = left_y_sorted[1]
        top_right_corner = right_y_sorted[0]
        bottom_right_corner = right_y_sorted[1]

        return (top_left_corner, top_right_corner, bottom_right_corner, 
                bottom_left_corner)

    first_quad = []
    second_quad = []
    third_quad = []
    fourth_quad = []

    # Group coordinates by quadrant
    for line_coord in line_coords:
        if line_coord[0] >= centroid[0]:
            if line_coord[1] <= centroid[1]:
                first_quad.append(line_coord)
            else:
                fourth_quad.append(line_coord)
        else:
            if line_coord[1] <= centroid[1]:
                second_quad.append(line_coord)
            else:
                third_quad.append(line_coord)

    # Compute distances from centroid for each coordinate
    first_quad_distances = np.array(list(map(np.linalg.norm, 
        first_quad - centroid)))
    second_quad_distances = np.array(list(map(np.linalg.norm, 
        second_quad - centroid)))
    third_quad_distances = np.array(list(map(np.linalg.norm, 
        third_quad - centroid)))
    fourth_quad_distances = np.array(list(map(np.linalg.norm, 
        fourth_quad - centroid)))

    # Find corner as the furthest point in each quadrant
    top_left_corner = second_quad[np.argmax(second_quad_distances)]
    top_right_corner = first_quad[np.argmax(first_quad_distances)]
    bottom_right_corner = fourth_quad[np.argmax(fourth_quad_distances)]
    bottom_left_corner = third_quad[np.argmax(third_quad_distances)]

    return (top_left_corner, top_right_corner, bottom_right_corner, 
            bottom_left_corner)

'''
Returns a dictionary of lists, where each list contains lists of coordinates
representing an element of the textline (e.g. top line, baseline, etc.) for
all text lines.

labels: a set of which elements of the text line to obtain coordinates for.
lines: the coordinates of the line bounding boxes.
'''
def get_label_coords(labels, lines):
    label_coords = {}

    for label in labels:
        label_coords[label] = []

    for line_coords in lines:
        if len(line_coords) < QUAD_CORNERS:
            continue

        line_coords = np.array(line_coords, np.int32)

        # Get centroid of line
        line_moments = cv2.moments(line_coords)
        centroid_x = int(line_moments["m10"] / line_moments["m00"])
        centroid_y = int(line_moments["m01"] / line_moments["m00"])
        centroid = np.array([centroid_x, centroid_y], dtype=np.uint32) 

        top_left_corner, top_right_corner, bottom_right_corner, \
                bottom_left_corner = get_corners(line_coords, centroid)

        center_to_top_left = top_left_corner - centroid
        center_to_top_right = top_right_corner - centroid
        center_to_bottom_right = bottom_right_corner - centroid
        center_to_bottom_left = bottom_left_corner - centroid

        # Calculate range of angles with centroid as origin for each side, 
        # bounded by angles of corners
        top_line_angle_range = [atan2(-center_to_top_right[1], 
            center_to_top_right[0]), atan2(-center_to_top_left[1], 
                center_to_top_left[0])]
        bottom_line_angle_range = [atan2(-center_to_bottom_left[1], 
            center_to_bottom_left[0]), atan2(-center_to_bottom_right[1], 
                center_to_bottom_right[0])]
        '''
        right_line_angle_range = [atan2(-center_to_bottom_right[1], 
            center_to_bottom_right[0]), atan2(-center_to_top_right[1], 
                center_to_top_right[0])]
        '''

        # Create line groups for coordinates initialized with corners
        top_line = [[top_left_corner[0], top_left_corner[1]], 
                [top_right_corner[0], top_right_corner[1]]]
        bottom_line = [[bottom_right_corner[0], bottom_right_corner[1]], 
                [bottom_left_corner[0], bottom_left_corner[1]]]
        '''
        right_line = [[top_right_corner[0], top_right_corner[1]], 
                [bottom_right_corner[0], bottom_right_corner[1]]]
        left_line = [[bottom_left_corner[0], bottom_left_corner[1]], 
                [top_left_corner[0], top_left_corner[1]]]
        '''

        # Add each coordinate to the correct group depending on which range the
        # angle from the centroid origin falls in
        for line_coord in line_coords:
            # Skip if coordinate is a corner
            if (line_coord == top_left_corner).all() or \
                    (line_coord == top_right_corner).all() or \
                    (line_coord == bottom_right_corner).all() or \
                    (line_coord == bottom_left_corner).all():
                continue

            angle = atan2(-(line_coord - centroid)[1], 
                    (line_coord - centroid)[0])

            if TOP_KEY in labels and angle > top_line_angle_range[0] and \
                    angle < top_line_angle_range[1]:
                top_line.append([line_coord[0], line_coord[1]])

            if BOTTOM_KEY in labels and angle > bottom_line_angle_range[0] and \
                    angle < bottom_line_angle_range[1]:
                bottom_line.append([line_coord[0], line_coord[1]])
            
            '''
            if angle > right_line_angle_range[0] and \
                    angle < right_line_angle_range[1]:
                right_line.append([line_coord[0], line_coord[1]])

            if angle < bottom_line_angle_range[0] or \
                    angle > top_line_angle_range[1]:
                left_line.append([line_coord[0], line_coord[1]])
            '''

        if TOP_KEY in labels:
            label_coords[TOP_KEY].append(sorted(top_line, 
                key=lambda k: [k[0], k[1]]))

        if BOTTOM_KEY in labels:
            label_coords[BOTTOM_KEY].append(sorted(bottom_line, 
                key=lambda k: [k[0], k[1]]))

    return label_coords

'''
Produce a mask for (a) specific side(s) of the bounding box of each text line.

labels: a set of which elements of the text line to obtain coordinates for.
page: the XML page element to search for baselines in.
thickness: the thickness of the lines in the mask.
image: the document image.
image_option: the type of image to impose the mask onto.
'''
def get_line_mask(labels, page, thickness, image, image_option):
    # lines = get_line_coords(page, NAMESPACE_GT) was needed for getting bounding box, but don't need this feature for 2020-21 year
    image_height = 1000 
    image_width = 1000
    image_shape = (image_height, image_width, CHANNELS)
    original_height = int(page.get(HEIGHT_TAG))
    original_width = int(page.get(WIDTH_TAG))

    height_ratio = image_height / float(original_height)
    width_ratio = image_width  / float(original_width)

    coords = dict() # Initialize Empty Dictionary
    mask = None if image is not None else np.zeros(image_shape, dtype=np.uint8)
    redscale_mask = None
    
    if image is not None and image_option == REDSCALE_IMAGE:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        redscale_mask = np.zeros(image_shape, dtype=np.uint8)

        # Feed inverted grayscale value to red channel
        for i in range(grayscale.shape[0]):
            for j in range(grayscale.shape[1]):
                redscale_mask[i][j][0] = 255 - grayscale[i][j]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None \
            else None

    if BASELINE_KEY in labels:
        baselines = get_baseline_coords(page, NAMESPACE_GT)
        if (len(baselines) == 0):
            baselines = get_baseline_coords(page, NAMESPACE_GT_2)
        
        for i, line in enumerate(baselines):
            for j, val in enumerate(line):
              #print("coordinate before: ", baselines[i][j])
              baselines[i][j] = np.multiply(baselines[i][j], [width_ratio, height_ratio])
              #print("coordinate after: ", baselines[i][j])
              
            #print("in baselines: ", i.shape)
        #print()
        #print("width ratio:", width_ratio)
        #print("height_ratio:", height_ratio)
        #print("Baseline type: ", baselines[0])
        coords[BASELINE_KEY] = baselines

    for label in labels:
        for line in coords[label]:
            if image is not None:
                if image_option == ORIGINAL_IMAGE:
                    cv2.polylines(image, [np.array(line, np.int32)], 
                            not LINES_CLOSED, LINE_COLORS[label], 
                            thickness=thickness)
                elif image_option == REDSCALE_IMAGE:
                    redscale_mask += cv2.polylines(np.zeros(image_shape, 
                        dtype=np.uint8), [np.array(line, np.int32)], 
                        not LINES_CLOSED, LINE_COLORS[label], 
                        thickness=thickness)

            else:
                mask += cv2.polylines(np.zeros(image_shape, dtype=np.uint8),
                    [np.array(line, np.int32)], not LINES_CLOSED, 
                    LINE_COLORS[label], thickness=thickness)

    if image is not None:
        if image_option == ORIGINAL_IMAGE:
            return image
        elif image_option == REDSCALE_IMAGE:
            return redscale_mask
    
    return mask

'''
Run the labeling tool with the given arguments.

argv: command line arguments.
'''
def main(argv):
    parser = argparse.ArgumentParser(description="A tool to produce text " \
            "line masks for document images.")
    parser.add_argument("gt_dir", help="The ground truth directory.")
    parser.add_argument("img_dir", help="The image directory.")
    parser.add_argument("out_dir", help="The desired output directory.")
    parser.add_argument("labels", help="Elements of text line to produce " \
            "mask for.", type=str, nargs='+', choices=[TOP_KEY, BOTTOM_KEY, 
                BASELINE_KEY])
    parser.add_argument("-t", "--thickness", help="Line thickness for the \
            output mask.", default=THICKNESS, type=int, choices=range(1, 
                THICKNESS + 1))
    parser.add_argument("-i", "--image", help="Impose mask onto image.", 
            type=str, choices=[NO_IMAGE, ORIGINAL_IMAGE, REDSCALE_IMAGE],
            default=NO_IMAGE)
    args = parser.parse_args(argv)
    gt_dir = args.gt_dir
    img_dir = args.img_dir
    out_dir = args.out_dir
    labels = set(args.labels)
    thickness = args.thickness
    image_option = args.image

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for filename in os.listdir(gt_dir):
        print("Labeling " + filename + "...")

        gt_file = open(gt_dir + "/" + filename, 'r')
        image = None if image_option == NO_IMAGE else \
                cv2.imread(img_dir + "/" + filename.replace(".xml", ".png"))

        root = et.parse(gt_file).getroot()
        gt_page = root.find(PAGE_TAG, NAMESPACE_GT)
        if (gt_page == None):
                gt_page = root.find(PAGE_TAG, NAMESPACE_GT_2)

        mask = get_line_mask(labels, gt_page, thickness, image, image_option)

        cv2.imwrite(out_dir + "/"+ filename.replace(".xml", ".png"),
                cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
        gt_file.close()

if __name__ == "__main__":
    main(sys.argv[1:])
