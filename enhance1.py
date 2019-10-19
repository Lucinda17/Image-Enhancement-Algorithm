"""
imageEnhance.py

YOUR WORKING FUNCTION

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image,ImageEnhance
input_dir = 'input/input'
output_dir = 'output/output'

# you are allowed to import other Python packages above
##########################
def enhanceImage(img):
   # inputImg: Input image, a 3D numpy array of row*col*3 in BGR format
    # outputImg: Enhanced image
    outputImg=img

    # --- CLACHE----#
    lab= cv2.cvtColor(outputImg, cv2.COLOR_BGR2LAB)
    # --- Splitting the LAB image to different channels ----
    l, a, b = cv2.split(lab)
    # --- Applying CLAHE to L-channel----
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    cl = clahe.apply(l)
    # --- Merge the CLAHE enhanced L-channel with the a and b channel ---
    outputImg = cv2.merge((cl,a,b))
    #-----Converting image from LAB Color model to RGB model----
    outputImg = cv2.cvtColor(outputImg, cv2.COLOR_LAB2BGR) 


    # # --- Pillow enhance function ---#
    # #-- enhance brightness,color,contrast and saturation
    outputImg=Image.fromarray(outputImg.astype('uint8'))
    outputImg=ImageEnhance.Brightness(outputImg).enhance(0.95)
    outputImg=ImageEnhance.Color(outputImg).enhance(2)
    outputImg=ImageEnhance.Contrast(outputImg).enhance(1.02)
    outputImg=ImageEnhance.Sharpness(outputImg).enhance(1.67)
    outputImg=np.array(outputImg)

   

    return outputImg

#Visualization Code:     
input_dir = 'input/input'
output_dir = 'output/output'
#visualize image no.18
i = 26
    
input   = cv2.imread(input_dir + str(i) + '.jpg')
gt      = cv2.imread('groundtruth/gt'+str(i)+'.jpg')
output  = enhanceImage(input)

gt      = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
input   = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
output  = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(50,50))
plt.subplot(121)
plt.imshow(gt),plt.xticks([]),plt.yticks([]),plt.title('groundtruth')
plt.subplot(122)
plt.imshow(output),plt.xticks([]),plt.yticks([]),plt.title('output')
plt.show()