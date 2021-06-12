'''
Tool to create a bad baseline xml file from a good baseline xml file. Mimics the issue in dhSegment where noise on the margins of the pages are incorrectly detected. Takes in a path to a directory of xml files with good baselines and outputs a directory of xml files with bad baselines
Usage: python bad_baseline_generator.py [input directory] [output directory]
Author: Aneesha Ramaswamy & Lynn Dang
'''
# Import Stuff
import numpy as np
import os
import os.path as PATH
import xml.etree.ElementTree as ET
import sys
import random
import math

PAGE_TAG = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Page"
BASELINE_TAG = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Baseline"
TEXTLINE_TAG = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextLine"

# since some pages have https tag
PAGE_TAG_2 = "{https://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Page"
BASELINE_TAG_2 = "{https://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}Baseline"
TEXTLINE_TAG_2 = "{https://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15}TextLine"

def get_baseline_points(tag,tree, root, textline):
    '''
    For a random input baseline in the given xml file, add line segments from the right or left of the page.
    Returns a string with the bad baseline points
    '''
# check if the tag is different
    if (tag == 0):
        page_width = int(root.find(PAGE_TAG).attrib['imageWidth'])
        page_height = int(root.find(PAGE_TAG).attrib['imageHeight'])
        baseline = textline.find(BASELINE_TAG)
    else:
        page_width = int(root.find(PAGE_TAG_2).attrib['imageWidth'])
        page_height = int(root.find(PAGE_TAG_2).attrib['imageHeight'])
        baseline = textline.find(BASELINE_TAG_2)

    baseline_length = random.randint(20, 50)
    baseline_angle = random.randint(-35, 35)
    # Decide which side of the page baseline will be on: left(0) or right(1)
    # NOTE: only applies to two page layouts, will need to change to
    # apply to one page layout
    coord = baseline.attrib['points'].split()[0]
    if (int(coord.split(',')[0]) < int(page_width/3)):
        direction = 0;
    else:
        direction = 1;

    # Find x,y coordinate of first or last coordinate in baseline,
    # and calculate x,y coords of new baseline points
    if (direction == 0):
        coord = baseline.attrib['points'].split()[0]
        start_x_coord = int(int(coord.split(',')[0])/random.randint(2,6))
        y_coord = int(coord.split(',')[1])
        end_x_coord = start_x_coord - baseline_length
        # Make sure end_x_coord doesn't go past beginning of page
        if (end_x_coord < 0):
            end_x_coord = 0

    else:
        coord = baseline.attrib['points'].split()[-1]
        end_baseline_x = int(coord.split(',')[0])
        start_x_coord = int(((1/random.randint(2,6))*(page_width - end_baseline_x))) + end_baseline_x
        y_coord = int(coord.split(',')[1])
        end_x_coord = start_x_coord + baseline_length
        # Make sure end_x_coord doesn't exceed page_width
        if (end_x_coord > page_width):
            end_x_coord = page_width
    # Recalculate y coordinate based on rotation angle
    y_coord_change = int(baseline_length * math.tan(math.radians(baseline_angle)))
    end_y_coord = y_coord + y_coord_change
    if (end_y_coord > page_height):
        end_y_coord = page_height
    elif (end_y_coord < 0):
        end_y_coord = 0
    # start_point is the point closes to the baseline and end_point is the
    # point closest to the end of the page
    start_point = str(start_x_coord) + "," + str(y_coord)
    end_point = str(end_x_coord) + "," + str(end_y_coord)
    return (str(start_point) + " " + str(end_point))

def get_side_points(tag, tree, root, textline, parent_map, spacing):
    '''
    For a random input baseline in the given xml file, adds a slanted line side note and
    adds extra lines above and below to simulate side notes
    returns number of lines changed
    '''

    # check if the tag is different
    if (tag == 0):
        page_width = int(root.find(PAGE_TAG).attrib['imageWidth'])
        page_height = int(root.find(PAGE_TAG).attrib['imageHeight'])
        baseline = textline.find(BASELINE_TAG)
    else:
        page_width = int(root.find(PAGE_TAG_2).attrib['imageWidth'])
        page_height = int(root.find(PAGE_TAG_2).attrib['imageHeight'])
        baseline = textline.find(BASELINE_TAG_2)

    # check if the line is long enough
    coords = baseline.attrib['points'].split()

    start_x_coord = int(coords[0].split(',')[0])
    end_x_coord = int(coords[len(coords) - 1].split(',')[0])

    if (abs(start_x_coord - end_x_coord) < page_width / 5):
        return 0 # no changes made

    # Decide which side of the page baseline will be on: left(0) or right(1)
    # NOTE: only applies to two page layouts, will need to change to
    # apply to one page layout
    
    coord = coords[0]
    if (int(coord.split(',')[0]) < int(page_width/3)):
        direction = 0;
    else:
        direction = 1;

    # Find x,y coordinate of first or last coordinate in baseline,
    # and calculate x,y coords of new baseline points
    if (direction == 0):
        coord = baseline.attrib['points'].split()[0]
        start_baseline_x = int(coord.split(',')[0])
        #start_x_coord = start_baseline_x - int(((1/random.randint(2, 3))*(start_baseline_x)))
        start_x_coord = start_baseline_x - int(int(start_baseline_x/random.randint(6,7)))
        #baseline_length = random.randint(int(abs(start_baseline_x - start_x_coord)/2), abs(start_baseline_x - start_x_coord))
        baseline_length = random.randint(int(1 *start_x_coord / 3), int(2 * start_x_coord / 3))

        y_coord = int(coord.split(',')[1])
        end_x_coord = start_x_coord - baseline_length
        # Make sure end_x_coord doesn't go past beginning of page
        if (end_x_coord < 0):
            end_x_coord = 0
        #calculates the middle point
        range_x_coord = int((end_x_coord + start_x_coord) / 2)
        mid_x_coord = random.randint(range_x_coord, max(start_x_coord, end_x_coord))

    else:
        coord = baseline.attrib['points'].split()[-1]
        end_baseline_x = int(coord.split(',')[0])
        start_x_coord = int(((1/random.randint(6, 10))*(page_width - end_baseline_x))) + end_baseline_x
        y_coord = int(coord.split(',')[1])

        max_length = page_width - start_x_coord
        baseline_length = random.randint(int(1 * max_length / 3), int(2 * max_length/3))
        end_x_coord = start_x_coord + baseline_length
        # Make sure end_x_coord doesn't exceed page_width
        if (end_x_coord > page_width):
            end_x_coord = page_width
        range_x_coord = int((end_x_coord + start_x_coord) / 2)
        #calculates the middle point
        mid_x_coord = random.randint(min(start_x_coord, end_x_coord), range_x_coord)
    
    # Recalculate y coordinate based on rotation angle
    y_coord_change = random.randint(0, spacing)
    end_y_coord = y_coord + y_coord_change
    if (end_y_coord > page_height):
        end_y_coord = page_height
    elif (end_y_coord < 0):
        end_y_coord = 0
    # start_point is the point closes to the baseline and end_point is the
    # point closest to the end of the page
    start_point = str(start_x_coord) + "," + str(y_coord)
    mid_point = str(mid_x_coord) + "," + str(end_y_coord)
    end_point = str(end_x_coord) + "," + str(end_y_coord)

    # extended baseline
    if (direction == 0):
        baseline.attrib['points'] = (str(end_point) + " " + str(mid_point) + " " + str(start_point) + " " + baseline.attrib['points'])
    else:
        baseline.attrib['points'] += (" " + str(start_point) + " " + str(mid_point) + " " + str(end_point))
    
    # add the extra baselines above and below to mimic a side note
    numLinesAbove = random.randint(2,3)
    numLinesBelow = random.randint(2,3)
    distance = 0
    for i in range (0, numLinesAbove):    
        distance += spacing

        new_y_coord = end_y_coord + distance
        # to avoid going off the page
        if (new_y_coord < 0):
            continue
        elif (new_y_coord > page_height):
            continue

        new_start_point = str(mid_x_coord) + "," + str(new_y_coord)
        new_end_point = str(end_x_coord) + "," + str(new_y_coord)

        if (tag == 0):
            new_textline = ET.SubElement(parent_map[textline], TEXTLINE_TAG)
            new_baseline = ET.SubElement(new_textline, BASELINE_TAG)
        else:
            new_textline = ET.SubElement(parent_map[textline], TEXTLINE_TAG_2)
            new_baseline = ET.SubElement(new_textline, BASELINE_TAG_2)

        new_baseline.attrib["points"] = new_start_point + " " + new_end_point
    
    distance = 0
    for i in range (0, numLinesBelow):    
        distance += spacing

        new_y_coord = end_y_coord - distance
        # to avoid going off the page
        if (new_y_coord < 0):
            continue
        elif (new_y_coord > page_height):
            continue

        new_start_point = str(mid_x_coord) + "," + str(new_y_coord)
        new_end_point = str(end_x_coord) + "," + str(new_y_coord)

        if (tag == 0):
            new_textline = ET.SubElement(parent_map[textline], TEXTLINE_TAG)
            new_baseline = ET.SubElement(new_textline, BASELINE_TAG)
        else:
            new_textline = ET.SubElement(parent_map[textline], TEXTLINE_TAG_2)
            new_baseline = ET.SubElement(new_textline, BASELINE_TAG_2)

        new_baseline.attrib["points"] = new_start_point + " " + new_end_point

    return 1 # changes were made

def offset_baseline_points(tag, tree, root, textline, parent_map):
    '''
    For a random input baseline in the given xml file, offsets the baseline above/below the original
    line
    returns number of lines changed
    '''
    if (tag == 0):
        baseline = textline.find(BASELINE_TAG)
    else:
        baseline = textline.find(BASELINE_TAG_2)

    # determine the length to offset
    baseline_x_length = random.randint(50, 60)
    coords = baseline.attrib['points'].split()

    # check if the baseline is long enough
    start_x_coord = int(coords[0].split(',')[0])
    end_x_coord = int(coords[len(coords) - 1].split(',')[0])

    if (len(coords) < 5):
        return 0 # no changes made
    if (abs(start_x_coord - end_x_coord) < baseline_x_length):
        return 0 # no chanes made
    start_index = random.randint(1, len(coords)-2)
    end_index = start_index+1

    # determine the length for the offset line
    new_x_length = int(coords[end_index].split(',')[0]) - int(coords[start_index].split(',')[0])
    while ((new_x_length < baseline_x_length) and (end_index < (len(coords)-2))):
        end_index += 1
        new_x_length = int(coords[end_index].split(',')[0]) - int(coords[start_index].split(',')[0])
    
   
    # cut the old baseline and create the new baseline
    baseline.attrib['points'] = coords[0]
    for i in range(0, start_index + 1):
        baseline.attrib['points'] += (" " + coords[i])

    if (tag == 0):
        new_textline = ET.SubElement(parent_map[textline], TEXTLINE_TAG)
        new_baseline = ET.SubElement(new_textline, BASELINE_TAG)
    else:
        new_textline = ET.SubElement(parent_map[textline], TEXTLINE_TAG_2)
        new_baseline = ET.SubElement(new_textline, BASELINE_TAG_2)
    new_baseline.attrib['points'] = coords[end_index]
    for i in range(end_index+1, len(coords)):
        new_baseline.attrib['points'] += (" " + coords[i])

    # offset the line
    for i in range(start_index, end_index+1):
      new_y = int(coords[i].split(',')[1]) + 15
      new_coord = str(coords[i].split(',')[0]) + "," + str(new_y)
      coords[i] = new_coord
      
    if (tag == 0):
        new_textline = ET.SubElement(parent_map[textline], TEXTLINE_TAG)
        new_baseline = ET.SubElement(new_textline, BASELINE_TAG)
    else:
        new_textline = ET.SubElement(parent_map[textline], TEXTLINE_TAG_2)
        new_baseline = ET.SubElement(new_textline, BASELINE_TAG_2)
    new_baseline.attrib['points'] = coords[start_index]
    for i in range(start_index+1, end_index + 1):
        new_baseline.attrib['points'] += (" " + coords[i])

    return 1 # changes were made  

def erase_baseline_points(tag, tree, root, textline, parent_map):
    '''
    For a random input baseline in the given xml file, erase a section of the baseline
    returns number of lines changed
    '''
    if (tag == 0):
        baseline = textline.find(BASELINE_TAG)
    else:
        baseline = textline.find(BASELINE_TAG_2)

    coords = baseline.attrib['points'].split()

    # check if the baseline is long enough
    start_x_coord = int(coords[0].split(',')[0])
    end_x_coord = int(coords[len(coords) - 1].split(',')[0])

    if (len(coords) < 5):
        return 0 # no changes made
    if (abs(start_x_coord - end_x_coord) < 60):
        return 0 # no changes made
    start_index = random.randint(1, len(coords)-2)
    end_index = start_index+1

    # determine the length to offset
    baseline_x_length = random.randint(int(abs(start_x_coord - end_x_coord)/8), int(abs(start_x_coord - end_x_coord)/5))

    # determine the length for the offset line
    new_x_length = int(coords[end_index].split(',')[0]) - int(coords[start_index].split(',')[0])
    while ((new_x_length < baseline_x_length) and (end_index < (len(coords)-2))):
        end_index += 1
        new_x_length = int(coords[end_index].split(',')[0]) - int(coords[start_index].split(',')[0])
    
   
    # split the baseline in two
    baseline.attrib['points'] = coords[0]
    for i in range(0, start_index + 1):
        baseline.attrib['points'] += (" " + coords[i])

    if (tag == 0):
        new_textline = ET.SubElement(parent_map[textline], TEXTLINE_TAG)
        new_baseline = ET.SubElement(new_textline, BASELINE_TAG)
    else:
        new_textline = ET.SubElement(parent_map[textline], TEXTLINE_TAG_2)
        new_baseline = ET.SubElement(new_textline, BASELINE_TAG_2)
    new_baseline.attrib['points'] = coords[end_index]
    for i in range(end_index+1, len(coords)):
        new_baseline.attrib['points'] += (" " + coords[i])

    return 1 # changes were made  


def create_textlines(tree, root):
    '''
    Randomly choose textlines to make a new textline that contains the
    bad baseline points
    '''
    num_changes = 0

    num_bad_textlines = random.randint(4, 10)
    textline_list = []

    # Find what textlines to base the bad baselines off of
    tag = 0
    for textline in root.iter(TEXTLINE_TAG):
        textline_list.append(textline)
    if (len(textline_list) == 0):
        tag = 1
        for textline in root.iter(TEXTLINE_TAG_2):
            textline_list.append(textline)

    if(num_bad_textlines > len(textline_list)):
        bad_textline_list = textline_list
    else:
        bad_textline_list = random.sample(textline_list, num_bad_textlines)

    # Map out all parent-child relationships
    parent_map = {c: p for p in tree.iter() for c in p}
    
    # Create new texlines with bad baseline points
    for textline in bad_textline_list:
        if (tag == 0):
            new_textline = ET.SubElement(parent_map[textline], TEXTLINE_TAG)
            new_baseline = ET.SubElement(new_textline, BASELINE_TAG)
        else:
            new_textline = ET.SubElement(parent_map[textline], TEXTLINE_TAG_2)
            new_baseline = ET.SubElement(new_textline, BASELINE_TAG_2)
        new_baseline.attrib["points"] = get_baseline_points(tag, tree, root, textline)
        num_changes += (len(new_baseline.attrib["points"]) > 0)
    
    # Create new side notes with bad baseline points
    spacing = findLineSpacing(root, textline_list, tag)
    num_bad_textlines = random.randint(6, 8)
    if (num_bad_textlines > len(textline_list)):  #if there aren't enough textlines, skip
        return num_changes

    bad_textline_list = random.sample(textline_list, num_bad_textlines)
    prevtextline = None
    for textline in bad_textline_list:
        if (textline is None):
            continue
        if (prevtextline is not None): # make sure the side notes don't overlap
            if getSpacingDiff(textline, prevtextline, tag) < (spacing * 3):
                continue
        num_changes += get_side_points(tag, tree, root, textline, parent_map, spacing)
        prevtextline = textline

    # create the offset baselines
    num_bad_textlines = random.randint(4, 6)
    if (num_bad_textlines > len(textline_list)):  #if there aren't enough textlines, skip
        return num_changes

    bad_textline_list = random.sample(textline_list, num_bad_textlines)

    for textline in bad_textline_list:
        if (textline is None):
            continue
        num_changes += offset_baseline_points(tag, tree, root, textline, parent_map)

    # erase parts of the baselines

    num_tries = 0
    num_erased = 0
    num_bad_offsets = random.randint(10, 13)

    if(num_bad_offsets > len(textline_list)):
      return num_changes  # if there arent' enough textlines, skip
  
    while (num_erased  < num_bad_offsets):
        if (num_tries > 4):
          break
        bad_textline_list = random.sample(textline_list, num_bad_textlines)
        for textline in bad_textline_list:
            if (textline is None):
                continue
            num_erased += erase_baseline_points(tag, tree, root, textline, parent_map)
        num_tries += 1

    return num_changes

def findLineSpacing(root, textline_list, tag):
    '''
    Finds minimum spacing between the lines to augment the side notes
    '''
    if (tag == 0):
        page_height = int(root.find(PAGE_TAG).attrib['imageHeight'])
    if (tag == 1):
        page_height = int(root.find(PAGE_TAG_2).attrib['imageHeight'])

    prevY = -1
    nextY = -1
    minSpacing = -1
    for textline in textline_list:
        if (tag == 0):
            baseline = textline.find(BASELINE_TAG)
        else:
            baseline = textline.find(BASELINE_TAG_2)
        if (baseline == None):
            continue

        y_points = []
        for points in baseline.attrib['points'].split():
            y_points.append(int(points.split(',')[1]))
        avg_y_point = int(sum(y_points) / len(y_points))
        
        if (prevY == -1):
            if (tag == 0):
                prevY = avg_y_point
            if (tag == 1):
                prevY = avg_y_point
        elif (nextY == -1):
            if (abs(avg_y_point - prevY) < page_height / 100):
                continue
            nextY = avg_y_point
            minSpacing = abs(nextY - prevY)
            prevY = nextY
        else:
            if (abs(avg_y_point - prevY) < page_height / 100):
                continue
            nextY = avg_y_point

            minSpacing = min(minSpacing, abs(nextY - prevY))
            prevY = nextY
        prevPoints = baseline.attrib['points']
    print("Min Space: ", minSpacing)
    return minSpacing

def getSpacingDiff(textline1, textline2, tag):
    '''
    returns spacing between the of lines
    '''
    if (tag == 0):
        baseline1 = textline1.find(BASELINE_TAG)
        baseline2 = textline2.find(BASELINE_TAG)
    else:
        baseline1 = textline1.find(BASELINE_TAG_2)
        baseline2 = textline2.find(BASELINE_TAG_2)

    if (baseline1 == None):
        return 0
    if (baseline2 == None):
        return 0

    y_points = []
    for points in baseline1.attrib['points'].split():
        y_points.append(int(points.split(',')[1]))
    avg_y_point1 = int(sum(y_points) / len(y_points))

    y_points = []
    for points in baseline1.attrib['points'].split():
        y_points.append(int(points.split(',')[1]))
    avg_y_point2 = int(sum(y_points) / len(y_points))

    return abs(avg_y_point2 - avg_y_point1)

def main(argv):
    # Get all .xml files in directory
    for file in os.listdir(sys.argv[1]):
        # Parse XML file
        tree = ET.parse(PATH.join(sys.argv[1], file))
        root = tree.getroot()
        # Add bad baselines, if no changes were made, then don't make a new xml
        if (create_textlines(tree, root) < 1):
            print("Skipping " + name)
            continue
        # Write tree to file
        name, ext = PATH.splitext(file)
        output_file_path = PATH.join(sys.argv[2], name + "_bad.xml")
        tree.write(output_file_path)
        print("Generating " + name + "_bad.xml...")

if __name__ == "__main__":
  main(sys.argv[1:])
