import cv2
import numpy as np
import os


for subdir, dirs, files in os.walk('./dataset'):
    for file in files:
        filepath = subdir+os.sep+file
        
        # load image
        img = cv2.imread(filepath) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale

        # threshold to get just the signature
        retval, thresh_gray = cv2.threshold(gray, thresh=205, maxval=255, type=cv2.THRESH_BINARY)

        # find where the signature is and make a cropped region
        points = np.argwhere(thresh_gray==0) # find where the black pixels are
        points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
        x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
        x, y, w, h = max(0,x-20), max(0,y-20), w+20, h+20 # make the box a little bigger
        crop = img[y:y+h, x:x+w] # create a cropped region of the gray image
        print(file)

        cv2.imwrite('./dataset_cropped/'+os.path.splitext(file)[0] + '.png', crop)

        # get the thresholded crop
        #retval, thresh_crop = cv2.threshold(crop, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

        #display
        #cv2.imshow("Cropped and thresholded image", thresh_crop) 
        #cv2.waitKey(0)
