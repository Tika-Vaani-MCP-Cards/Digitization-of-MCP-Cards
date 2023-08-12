# This file will crop the same ROI multiple times, w.r.t different local crops provided
# We will process all the ROIs through the ocr and keep the one with the highest confidence score



from itertools import count
import os
from sys import path
from threading import local
import cv2 as cv
import cv2
import pandas as pd
from datetime import date
import numpy as np
import traceback
import shutil
from random import choice
from tqdm  import tqdm
import pickle
from local_variables import *



def get_cv_image(image_path):
	return cv.imread(image_path, 1)


def save_ROIs(templatePath, coordinates, alignedImage, filename, x1,y1,x2,y2):
	prefixImage = filename.split('.')[0]
	temp = alignedImage.copy()
	for index, row in coordinates.iterrows():
		if row[0] == templatePath.split("/")[-1]:
			if not ((x1 <= row[2] and x2 >= row[4]) and (y1 <= row[3] and y2 >= row[5])):
				continue

			xx1,yy1 = row[2],row[3]
			xx2,yy2 = row[4],row[5]
			
			currentROI = alignedImage[yy1:yy2, xx1-10:xx2+8]

			# Save the ROI to a file
			key = os.path.join(LOCALLY_ALIGNED_ROIS[template_name], '_'.join([str(prefixImage), row[1]+".jpg"]))
			if key in path_counter:
				path_counter[key] += 1
			else:
				path_counter[key] = 0

			cv.imwrite(os.path.join(LOCALLY_ALIGNED_ROIS[template_name], '_'.join([str(prefixImage), row[1]+"_"+str(path_counter[key])+".jpg"])), currentROI)
			temp = cv.rectangle(temp,(xx1,yy1), (xx2,yy2), (0,255,0),5)

	# cv2.imshow('temp', temp)
	# cv2.waitKey(0)


def align_query_image(template, queryImage, method):
	# Check which matching method is selected
	if method == "SIFT":
		# Create a SIFT object
		sift = cv2.SIFT_create()
		# Detect keypoints and compute feature vectors for the template and query images
		kp1, des1 = sift.detectAndCompute(template, None)
		kp2, des2 = sift.detectAndCompute(queryImage, None)

		# Set up the FLANN matcher parameters
		FLANN_INDEX_KDTREE = FLANN_TREE_INITIAL_INDEX
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=FLANN_MAX_TREES)
		search_params = dict(checks=FLANN_SEARCH_CHECKS)
		flann = cv.FlannBasedMatcher(index_params, search_params)

		# Match the feature vectors using the kNN algorithm
		matches = flann.knnMatch(des1, des2, k=KNN_NEIGHBOURS)

		# Set up the match mask
		matchesMask = [[0, 0] for i in range(len(matches))]

		# Use Lowe's Ratio Test to filter out bad matches
		good = []
		for i, (m, n) in enumerate(matches):
			if m.distance < LOWE_RATIO_TEST_VAL * n.distance:
				good.append(m)
				matchesMask[i] = [1, 0]

		# Draw the matches
		draw_params = dict(matchColor=(0, 255, 0),
						   singlePointColor=(0, 0, 255),
						   matchesMask=matchesMask,
						   flags=cv.DrawMatchesFlags_DEFAULT)
		img2 = cv.drawMatchesKnn(template, kp1, queryImage, kp2, matches, None, **draw_params)
		# cv2.imwrite('img2.png', img2)

		# If there are enough good matches, perform Homography and Warping
		if len(good) > MIN_MATCH_COUNT:
			src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
			dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
			
			M, mask = cv.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 15.0)
			matchesMask = mask.ravel().tolist()
			
			num_of_inliers = 0
			for i in range(len(matchesMask)):
				if matchesMask[i] == 1:
					num_of_inliers += 1

			# Warp the query image into the template image
			warpedImage = cv.warpPerspective(queryImage, np.linalg.inv(M), (template.shape[1], template.shape[0]))
			return warpedImage, M, num_of_inliers 

def get_cropped_template_list(img):
	h,w,c = img.shape
	img_list = []

	for i in range(num_of_regions):
		img1    = img.copy()
		left_x  = crop_coordinates_list[i][0][0]
		right_x = crop_coordinates_list[i][1][0]
		top_y  = crop_coordinates_list[i][0][1]
		bottom_y = crop_coordinates_list[i][1][1]

		img1[:top_y,:] = 0
		img1[:,:left_x] = 0
		img1[:,right_x:] = 0
		img1[bottom_y:,:] = 0

		# if i == 0:
		# 	img1[:,(right_x-left_x)//2:] = img1[:,:(w-(right_x-left_x)//2)]
		# 	img1[:,:(right_x-left_x)//2] = 0

		img_list.append(img1)
		
	
	return img_list

def get_cropped_img_list(img, imgPath):
	h,w,c = img.shape
	# print(h,w)
	img_list = []
	margin_px = 75 # for query image we don't perform tight cropping but keep some pixels left and right
	for i in range(num_of_regions):
		img1    = img.copy()
		left_x  = crop_coordinates_list[i][0][0]
		right_x = crop_coordinates_list[i][1][0]
		top_y  = crop_coordinates_list[i][0][1]
		bottom_y = crop_coordinates_list[i][1][1]

		# print(left_x, right_x, top_y, bottom_y)
		img1[:max(0,top_y-margin_px),:] = 0
		img1[:,:max(0,left_x-margin_px)] = 0
		img1[:,min(w,right_x+margin_px):] = 0
		img1[min(w,bottom_y+margin_px):, :] = 0

		img_list.append(img1)

	return img_list



def recur(query_image, template_image, feaureDetector, filename):
	 
	h,w,c = query_image.shape
	warped_list = []
	template_list = get_cropped_template_list(template_image)
	img_list = get_cropped_img_list(query_image, filename)

	for i in range(num_of_regions):
		img1 = img_list[i]
		aligned_image, M, num_of_inliers = align_query_image(template_list[i], img1, feaureDetector)

		# Write matrix
		# Save homography matrix
		with open(os.path.join(LOCAL_HOMOGRAPHY_MATRIX[template_name], filename.split('.')[0]+'_'+ str(i)+'.pkl'), 'wb') as f:
			pickle.dump(M, f)

		x1,y1 = crop_coordinates_list[i][0]
		x2,y2 = crop_coordinates_list[i][1]

		save_ROIs(current_template, template_coordinates, aligned_image, filename, x1, y1, x2, y2)

	
def Iou(a,b):
	x1 = max(a[0], b[0])
	x2 = min(a[2], b[2])
	y1 = max(a[1], b[1])
	y2 = min(a[3], b[3])

	area = (x2-x1)*(y2-y1)

	return area


current_template = TEMPLATE_PATHS[TEMPLATE_INDEX] # get the current template
template_name = current_template.split('/')[-1] 
source = GLOBALLY_ALIGNED_IMAGES[template_name]	# get the source folder of query images aligned using global homography
template_coordinates = pd.read_csv(TEMPLATE_COORDINATES[template_name]) # get the ROI coordinates for the chosen template

# get the crop coordinates for the current template
crop_coordinates_list = []
local_regions = pd.read_csv(TEMPLATE_LOCAL_REGIONS[template_name])
for rows in local_regions.iterrows():
	crop_coordinates_list.append([[rows[1][0], rows[1][1]], [rows[1][2], rows[1][3]]])
num_of_regions  = len(crop_coordinates_list)

template_image = get_cv_image(current_template)
path_counter = {} # we will use this dictionary to save multiple images for the same roi
print(source)
for file in os.scandir(source):
	try:
		filename = file.name
		print(filename)
		query_image = cv.imread(os.path.join(source, filename), 1)
		
		if not any(ext in filename for ext in ('.jpg', '.png', 'jpeg')):
			continue
	
		recur(query_image, template_image, 'SIFT', filename)

	except Exception as e:
		print(traceback.format_exc())
		exit()
		pass


	


