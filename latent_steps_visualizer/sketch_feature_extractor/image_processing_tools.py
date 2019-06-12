#!/usr/bin/env python
# coding: utf-8

# In[42]:


import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pdf2image import convert_from_path

pdf_directory = "data/pdf/"
img_directory = "data/img/"

def base_name(path):
    return os.path.splitext(os.path.basename(path))[0]

def new_performer_data(name):
    ##make new directory for performer
    new_folder_names = ["", "/unprocessed", "/red", "/cyan", "/red/scaled", "/cyan/scaled"]
    target_dir = img_directory + name
    for dir_name in new_folder_names:
        try:
            os.mkdir(target_dir + dir_name)
        ##if directory exists, continue
        except Exception as e:
            print(e)
            continue

##display image in notebook
def image(img):
    fig = plt.figure(figsize=(16,12))
    plt.imshow(img)
    plt.show()

## cover up uneccesarry information from image
def add_whitespace(img,x,y,w,h):
    result = img.copy()
    cv2.rectangle(result, (x,y),(0+w,0+h),(255,255,255),-1)
    return result
    
##openCV commands for cropping PAF output
def get_outlines(inputImg, thresh):
    gray = cv2.cvtColor(inputImg, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    # invert image
    binary = cv2.bitwise_not(binary)
    # morph coins by eroding and dilating to remove noise
    morph_kernel = np.ones((15,15),np.uint8)
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel)
    return morphed
    
def crop_heatmap(inputImg, padding):
    #get origional image dimentions
    crop_amt = 10
    input_height, input_width = inputImg.shape[:2]
    min_area = [(input_width/crop_amt), (input_width/crop_amt)]
    max_area = [input_width - (input_width/crop_amt), input_height - (input_width/crop_amt)]
    
    # create copy of image to draw bounding boxes
    bounding_img = np.copy(inputImg)
    #filter image to get shapes
    source_morphed = get_outlines(inputImg, 145)
    outline = np.copy(source_morphed)
    # find contours
    all_contours,hierarchy = cv2.findContours(source_morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print('found %d contor(s)' % len(all_contours))

    # Find biggest bounding box within range
    largest = 0
    main_cnt = all_contours[0]
    for contour in all_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if(w > min_area[0] and h > min_area[1] and w < max_area[0] and h < max_area[1]):
            if(cv2.contourArea(contour) > largest):
                largest = cv2.contourArea(contour)
                main_cnt = contour

    x, y, w, h = cv2.boundingRect(main_cnt)
    #cv2.rectangle(bounding_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    crop_x = int(round(input_width/w*padding))
    crop_img = bounding_img[y-padding:y+h+padding, x-crop_x:x+w+crop_x] 
    result = crop_img
    return result

def update_progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
    return count + 1

def transparent_background(img):
    grey = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    alpha = output[:,:,3] # Channel 3
    trans = np.dstack([bgr, alpha]) # Add the alpha channel
#     return result

def compress(img, scale):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(grey, scale) 
    return scaled

def extract_drawing(img, hsv_img, lower, upper):
    mask_color = cv2.inRange(hsv_img, lower, upper)
    mask_outline = cv2.inRange(hsv_img, (0, 50, 0, 0), (255, 255,255, 255))
    mask = cv2.bitwise_and(mask_color, mask_outline)
    output = cv2.bitwise_and(img,img, mask=mask)
    return output
