"""
evaluate.py

DO NOT MODIFY ANY CODES IN THIS FILE EXCEPT THE DEFAULT PARAMETERS
OTHERWISE YOUR RESULTS MAY BE INCORRECTLY EVALUATED! 

@author: Wong Lai Kuan, 2019
For questions or bug reporting, please send an email to lkwong@mmu.edu.my

"""

import cv2
import numpy as np
import importlib
import sys, getopt
from prettytable import PrettyTable
import enhance1 as eh

# Default parameters (the only code you can change)
verbose = False
input_dir = 'input/input'
output_dir = 'output/output'
groundtruth_dir = 'groundtruth/gt'
numImages = 30

def histogram_intersection(x, y):
    n = x.shape[0]
    intersection = 0;
    for i in range(1,n):
        intersection = intersection + min(x[i], y[i])
    return intersection/np.sum(y);
    
def hsv_Histogram(img):
    binsize=255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
    h, s, v = cv2.split(img)
    h_hist, h_bin = np.histogram(h, binsize, density=True)
    s_hist, s_bin = np.histogram(s, binsize, density=True)
    v_hist, v_bin = np.histogram(v, binsize, density=True)
    hsv_hist = np.concatenate((h_hist, s_hist, v_hist))
    
    return hsv_hist

# Read command linehargs
myopts, args = getopt.getopt(sys.argv[1:],"i:vh")

# Reload module
importlib.reload(eh)

################################################
# o == option    a == argument passed to the o #
################################################

# parsing command line args
for o, a in myopts:
    #print(o)
    #print(a)
    if o == '-v':
        verbose = True
    elif o == '-h':
        print("\nUsage: %s -v               for extra verbosity" % sys.argv[0])
        sys.exit()
    else:
        print(' ')

simScore = np.zeros(numImages,)
error = np.zeros(numImages,)

# Evaluate each image and compare with ground-truth
for i in range(numImages):
    input = cv2.imread(input_dir + str(i+1) + '.jpg')
    output = eh.enhanceImage(input)
    cv2.imwrite(output_dir + str(i+1) +'.jpg', output)
    gt = cv2.imread(groundtruth_dir + str(i+1) + '.jpg')
    
    output_hist = hsv_Histogram(output)
    gt_hist = hsv_Histogram(gt)
    
    simScore[i]=histogram_intersection(output_hist, gt_hist)


# Print TOP 15 out of 30 performance scores (dataset1)
# uncomment below code to print TOP 15
'''
# Print performance scores   
top = 15
img_score = []   
for i in range(numImages):
    img_score.append([i+1, round(simScore[i],4)])
    img_score.sort(key=lambda x: x[1], reverse=True)
    img_score = img_score[:top]   
simScore.sort()
simScore = simScore[::-1]

if verbose:
    print('\n####  DETAILED TOP 10 RESULTS  ####')
    t = PrettyTable(['Image', 'Similarity Score'])
    for i,j in img_score:
        t.add_row([i, j])
    t.add_row([' ',' '])
    t.add_row(['All', str(round(np.sum(simScore[:top])/top,4)) ])
    print(t)

else:
    print('Similarity Score: %d%%' % (np.sum(simScore[:top])/top*100)) 
'''
# Print all 30 performance scores (dataset1)       
if verbose:
    print('####  DETAILED RESULTS  ####')
    t = PrettyTable(['Image', 'Similarity Score'])
    for i in range(numImages):
        t.add_row([i+1, str(round(simScore[i],4))])
    t.add_row([' ',' '])
    t.add_row(['All', str(round(np.sum(simScore)/numImages,4)) ])
    print(t)
else:
    print('Similarity Score: %d%%' % (np.sum(simScore)/numImages*100)) 
            
# END OF EVALUATION CODE####################################################