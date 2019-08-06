import os, sys, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from image_processing_tools import *

img_directory = "../../Soma_Draw/img_save/data/" ## override directory path
excluded = [".ipynb_checkpoints", "unprocessed", "cyan", "red", "scaled"]
names = []

# list of colour boundaries
boundaries = [
	([70, 0, 100, 0], [100, 255,255, 255]),
	([0, 50, 0, 0], [70, 255,255, 255])
]

for path, subdirs, files, in os.walk(img_directory):
    for subdir in subdirs:
        if any(ex in subdir for ex in excluded):
            continue
        names.append(subdir)
        print("Reading from foler: " + subdir)
        new_folder_names = ["", "/red", "/cyan", "/red/scaled", "/cyan/scaled"]
        target_dir = img_directory + subdir
        for dir_name in new_folder_names:
            try:
                os.mkdir(target_dir + dir_name)
            ##if directory exists, continue
            except Exception as e:
#                 print(e)
                continue
        for path, subdirs, files, in os.walk(img_directory+subdir):
            for file in files:
                try:
                    drawings = []
                    file_path = os.path.join(path,file)
                    img = cv2.imread(file_path)
                    ## convert to hsv
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    # loop over the boundaries
                    for lower_bound, upper_bound in boundaries:
                        drawings.append(extract_drawing(img, hsv, tuple(lower_bound), tuple(upper_bound)))
                    ## separate masking of cyan and red + ## filter bodily outline
                    file_path = os.path.normpath(os.path.join(file_path, os.pardir))
                    cyan_drawing, red_drawing = drawings
                    cv2.imwrite(file_path+"/cyan/scaled/"+file, compress(cyan_drawing, (224, 224)))
                    cv2.imwrite(file_path+"/red/scaled/"+file, compress(red_drawing, (224, 224)))
                    print ("finished processing new files")
                except:
                    continue
