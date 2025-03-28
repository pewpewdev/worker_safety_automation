import argparse
import os
import cv2
import logging
import statistics
import datetime
def get_class_mapping(class_file_path):
    
    """Genereates the class mapping from classes.txt. 
    Please note since we don't have class index (0,1,2...) mentioned in classes.txt, we take first entry as class 0, second as class 1 and so on
    eg: if your classes.txt have following classes, you class mapping will be like "CLASS MAPPING" 
    
    |     Person     |   CLASS MAPPING   
|--------------------|--------------------
|   hard-hat         |      0 
|   gloves           |      1 
|   mask             |      2
|   glasses          |      3
|   boots            |      4
|   vest             |      5
|   ppe-suit         |      6 
|   ear-protector    |      7
|   safety-harness   |      8

    

    Args:
        class_file_path (str): Path to classes.txt
    return: 
        dict
    """
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    class_mapping = {}

    # Read classes.txt
    try:
        with open(class_file_path, "r") as file:
            classes = file.readlines()
        
        #classes.txt is empty
        if not classes:
            raise ValueError("classes.txt is empty.")
    except FileNotFoundError:
        logging.error("classes.txt not found.")
        exit()
        
        
    # Generate class mapping
    for index, class_name in enumerate(classes):
        class_mapping[index] = class_name.strip()
        
        
    #Print the class mapping
    logging.info("Class Mapping:")
    for index, class_name in class_mapping.items():
        logging.info(f"{index}: {class_name}")
    return class_mapping
def read_annotation_file(annotation_file_path):
    
    annotations_list = []
    try:
        with open(annotation_file_path, "r") as file:
            annotations = file.readlines()
        
        for line in annotations:
            line = line.strip().split()

            if len(line) < 5:
                continue
            
            class_index = int(line[0])
            x_center, y_center, width, height = map(float, line[1:5])
            
            try:
                annotations_list.append({
                    'class_index': class_index,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
            except:
                continue
        return annotations_list
        #Annotation file  is empty
        if not annotations:
            raise ValueError("annotations file is empty")
            return None
    except:
        logging.info("{annotation_file_path} not found.")
        return None
    
def get_image_coords(yolo_annotations,image_width,image_height):
    
    x_center = yolo_annotations["x_center"]
    y_center = yolo_annotations["y_center"]
    width = yolo_annotations["width"]
    height = yolo_annotations["height"]
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height
    
    xmin = int(x_center - (width / 2))
    ymin = int(y_center - (height / 2))
    xmax = int(x_center + (width / 2))
    ymax = int(y_center + (height / 2))
    
    return (xmin, ymin, xmax, ymax)


def calculate_overlap_percentage(rect1, rect2):
    """Calculate the overlap percentage between person bounding bbox rectangle and ppe bbox rectangle

    Args:
        rect1 (tuple): BBox co-ords for person
        rect2 (tuple): BBox co-ords for ppe
    Returns:
        int: Value between 0-100 which will be the overlap percentage
    """
    # Calculate intersection coordinates
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])

    # Calculate area of intersection
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate total area of rect2
    rect2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

    # Calculate percentage overlap
    overlap_percentage = (intersection_area / rect2_area) * 100
    
    return overlap_percentage
def calculate_offset(bbox_person, bbox_ppe,cropped_image_width,cropped_image_height,class_index):
    """After cropping the person bbox from image, we have to change the offset of ppe annotations.
    This function calculates the offset and normalize the annotations intoi the yolo format

    Args:
        bbox_person (tuple): xmin, ymin, xmax and ymax of the person annotation wrt to full image
        bbox_ppe (tuple):  xmin, ymin, xmax and ymax of the ppe annotation wrt to full image
        cropped_image_width (int): cropped person image width
        cropped_image_height (int): cropped person image height
        class_index (int): class index of ppe annotations

    Returns:
        tuple: returns tuple contatining annotations of ppe in yolo format
    """
    x_offset, y_offset =bbox_person[0],bbox_person[1]
    new_xmin =  bbox_ppe[0]-x_offset
    new_ymin = bbox_ppe[1]-y_offset
    new_xmax = bbox_ppe[2]-x_offset
    new_ymax = bbox_ppe[3]-y_offset
    
    if new_xmin<0:
        new_xmin = 0
    if new_ymin<0:
        new_ymin=0
    if new_xmax>cropped_image_width:
        new_xmax=cropped_image_width
    if new_ymax>cropped_image_height:
        new_ymax=cropped_image_height
        
    centerX = (new_xmin + new_xmax) / 2
    centerY = (new_ymin + new_ymax) / 2

    # Calculate width and height of bounding box
    boxWidth = new_xmax-new_xmin
    boxHeight =  new_ymax - new_ymin 
    
    # Normalize coordinates and dimensions
    normalizedCenterX = centerX / cropped_image_width
    normalizedCenterY = centerY / cropped_image_height
    normalizedBoxWidth = boxWidth / cropped_image_width
    normalizedBoxHeight = boxHeight / cropped_image_height
    #print("new_ymax",boxWidth,boxHeight,centerX,centerY,cropped_image_width,cropped_image_height,normalizedCenterX)
    #print(class_index,normalizedCenterX,normalizedCenterY,normalizedBoxWidth,normalizedBoxHeight )
    return (class_index,normalizedCenterX,normalizedCenterY,normalizedBoxWidth,normalizedBoxHeight )
def crop_and_save_annotations(image_file_path, annotation_file_path, class_mapping, save_dir,person_class_index,overlap_percentage,image_name,):
    """This function will crop and save images for every person annotation.
        This will readjust the annotation from image level to person level

    Args:
        image_file_path (str): path to image file
        annotation_file_path (str): path to annotation file
        class_mapping (dict): dictonary for class mapping 
        save_dir (str): Path to save crops and annotations
    """
    global each_class_count 
    global person_width_list 
    global person_height_list
    #read annotations from annotation file
    annotations = read_annotation_file(annotation_file_path)
    
    #If some issue with annotation file
    if not annotations:
        return None
    #read and get  height and width of image
    read_image = cv2.imread(image_file_path)
    height, width, shape = read_image.shape
    
    #Iterate through annotations
    for one_anno in annotations:
        ppe_annotations_list = []
        #If annotation belong to person class
        if one_anno["class_index"]==person_class_index:
            
            #Get annotation of person in image co-ords
            rect_person = get_image_coords(one_anno,width,height)
            
            #crop and save person image 
            cropped_image = read_image[rect_person[1]:rect_person[3], rect_person[0]:rect_person[2]]
            cropped_image_height = rect_person[3]-rect_person[1]
            cropped_image_width = rect_person[2]-rect_person[0]
            person_width_list.append(cropped_image_width)
            person_height_list.append(cropped_image_height)
            save_filename_name = image_name+str(datetime.datetime.now())
            cv2.imwrite(os.path.join(save_dir, "images",save_filename_name+".jpg"), cropped_image)
            
            #iterate through other annotations to see which other annotations lying inside person
            for other_anno in annotations:
                #for class index other than person, check what bbox are lying inside the person
                if int(other_anno["class_index"])!=person_class_index:
                    
                    #Get annotation of ppe in image co-ords
                    rect_other= get_image_coords(other_anno,width,height)
                    
                    #calculate overlap between person 
                    overlap = calculate_overlap_percentage(rect_person,rect_other) 
                    if overlap>= overlap_percentage:
                        if other_anno["class_index"] not in each_class_count:
                            each_class_count[other_anno["class_index"]]=1
                        else:
                            each_class_count[other_anno["class_index"]]+=1
                        bbox_ppe_yolo = calculate_offset(rect_person,rect_other,cropped_image_width,cropped_image_height,other_anno["class_index"])
                        ppe_annotations_list.append(bbox_ppe_yolo)
            with open(os.path.join(save_dir, "labels",save_filename_name+".txt"), "w") as file:
                # Write YOLO format for one bounding box
                print("items", ppe_annotations_list)
                for items in ppe_annotations_list:
                    file.write(f"{items[0]} {items[1]} {items[2]} {items[3]} {items[4]}\n")
                
def calculate_statistics(data):
    # Mean
    mean = statistics.mean(data)
    
    # Median
    median = statistics.median(data)
    
    # Mode
    mode = statistics.mode(data)
    
    return  mean, median, mode


                
    
ap = argparse.ArgumentParser()
ap.add_argument("--img_dir", required=True,type = str, help="Path to images dir")
ap.add_argument("--anno_dir", required=True,type = str, help="Path to annotations dir")
ap.add_argument("--class_file", required=True, type=str, help = "Path to classes file as used in yolo")
ap.add_argument("--save_dir", required=True, type = str, help = "Path where ppe annotation data will be stored")
ap.add_argument("--person_class_index", required=False, type = int, default=0, help = "Path where ppe annotation data will be stored")
ap.add_argument("--overlap_percentage", required=False, type = int, default=70, help = "how much overlap must be there between person annotation and ppe annotations")

args = ap.parse_args()

#Make save directory
os.makedirs(os.path.join(args.save_dir, "images"),exist_ok=True)
os.makedirs(os.path.join(args.save_dir, "labels"),exist_ok=True)
#Global variables
each_class_count = {}
person_width_list = []
person_height_list = []

class_mapping = get_class_mapping(args.class_file)
for images in os.listdir(args.img_dir):
    image_path = os.path.join(args.img_dir,images)
    labels_path = os.path.join(args.anno_dir,os.path.splitext(images)[0]+".txt")
    crop_and_save_annotations(image_path, labels_path, class_mapping, args.save_dir, args.person_class_index,args.overlap_percentage, os.path.splitext(images)[0])
    continue


print("Height mean, median, mode",calculate_statistics(person_height_list))
print("each class count dict", each_class_count)
print("width mean, median, mode",calculate_statistics(person_width_list))





















