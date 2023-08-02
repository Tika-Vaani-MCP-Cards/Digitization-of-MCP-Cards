import os
import cv2
import pandas as pd 

# Set Paths Here

# Enter path of the templates here
TEMPLATE_PATHS = [
    os.path.join("./", "templates", "designPdf_templateImage_upper.jpg"),
    os.path.join("./", "templates", "designPdf_templateImage_lower.jpg")
]

# Enter coordinate CSV file path here, corresponding to the template 
TEMPLATE_COORDINATES = {
    TEMPLATE_PATHS[0].split('/')[-1]: os.path.join("./", "templateCoordinates.csv"),
    TEMPLATE_PATHS[1].split('/')[-1]: os.path.join("./", "templateCoordinates.csv")
}

# Enter query image folder path here, corresponding to the template 
IMAGE_SOURCE = {
    TEMPLATE_PATHS[0].split('/')[-1]: os.path.join("./", "dm_mcp_cards"),
    TEMPLATE_PATHS[1].split('/')[-1]: os.path.join("./", "dm_mcp_cards")
}

# Create paths for writing files, folders will be created based on the template names
GLOBALLY_ALIGNED_IMAGES = {} # A dict of template_name -> corresponding globally aligned images folder path
GLOBAL_HOMOGRAPHY_MATRIX = {} # A dict of template_name -> corresponding global homography matrices folder path
GLOBALLY_ALIGNED_ROIS =  {} # A dict of template_name -> corresponding folder of rois generated from global alignment path
LOCAL_HOMOGRAPHY_MATRIX = {} # A dict of template_name -> corresponding local homography matrices folder path
LOCALLY_ALIGNED_ROIS =  {} # A dict of template_name -> corresponding folder of rois generated from local alignment

TEMPLATE_SIZES = {} # A dict of TEMPLATE_SIZES

# Loop through the templates
for i in TEMPLATE_PATHS:
    template_name = i.split('/')[-1]

for i in TEMPLATE_PATHS:
    template_name = i.split('/')[-1]

    GLOBALLY_ALIGNED_IMAGES[template_name]  =  os.path.join("./new", template_name + '__' + "globally_alined_images")
    GLOBAL_HOMOGRAPHY_MATRIX[template_name] =  os.path.join("./new", template_name + '__' + "global_alignment_mat")
    GLOBALLY_ALIGNED_ROIS[template_name]    =  os.path.join("./new", template_name + '__' + "globally_aligned_rois")
    LOCAL_HOMOGRAPHY_MATRIX[template_name]  =  os.path.join("./new", template_name + '__' + "local_alignment_mat")
    LOCALLY_ALIGNED_ROIS[template_name]     =  os.path.join("./new", template_name + '__' + "locally_aligned_rois")

    if not os.path.exists(GLOBALLY_ALIGNED_IMAGES[template_name]):
        os.makedirs(GLOBALLY_ALIGNED_IMAGES[template_name])
    
    if not os.path.exists(GLOBAL_HOMOGRAPHY_MATRIX[template_name]):
        os.makedirs(GLOBAL_HOMOGRAPHY_MATRIX[template_name])
    
    if not os.path.exists(GLOBALLY_ALIGNED_ROIS[template_name]):
        os.makedirs(GLOBALLY_ALIGNED_ROIS[template_name])
    
    if not os.path.exists(LOCAL_HOMOGRAPHY_MATRIX[template_name]):
        os.makedirs(LOCAL_HOMOGRAPHY_MATRIX[template_name])
    
    if not os.path.exists(LOCALLY_ALIGNED_ROIS[template_name]):
        os.makedirs(LOCALLY_ALIGNED_ROIS[template_name])


    # A dict of template names -> template shape
        img = cv2.imread(i)
        h,w,_ = img.shape
        TEMPLATE_SIZES[template_name] = (h,w)

# For local alginment enter custom coordinates manually- (x1,y1),(x2,y2) 
# Run mark_xy.py program and click on the template image to get the marked coordinate list, which you can paste here
# The local_alignment program will crop img[x1:x2, y1:y2] for both template and query image and perform local alignment for each region
# [ 
#    [[left_x1, right_x1], [top_y1, bottom_y1]], .. ]
# ]
crop_coordinates_dict = {
    'designPdf_templateImage_upper.jpg': [[[11, 12], [512, 591]], [[21, 10], [755, 590]], [[257, 10], [998, 585]], [[499, 11], [1245, 587]], [[744, 8], [1653, 584]]],
    'designPdf_templateImage_lower.jpg': [[[0, 500], [0, 600]], [[0, 750], [0, 600]], [[250, 1000], [0, 600]], [[500, 1250], [0, 600]], [[750, 1500], [0, 600]]]
}

# The index of the template to process
TEMPLATE_INDEX = 0

#Computer Vision Params
LOWE_RATIO_TEST_VAL = 0.70 # play with different values and see the changes in no. of matches (optimal is 0.70)
KNN_NEIGHBOURS = 2
FLANN_TREE_INITIAL_INDEX = 0
FLANN_MAX_TREES = 5
FLANN_SEARCH_CHECKS = 50
MIN_MATCH_COUNT = 10

