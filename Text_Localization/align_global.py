import os
from sys import path
import cv2 as cv
import cv2
import pandas as pd
from datetime import date
import numpy as np
import shutil
from random import choice
import pickle
from tqdm import tqdm
from local_variables import *
import traceback


def get_cv_image(image_path):
	return cv.imread(image_path, 1)

def get_pandas(csv_path):
	return pd.read_csv(csv_path)

#TODO : Method can be versatile in the future such as ORB, SIFT, SURF, BRIEF. As of now our codebase only processes SIFT features
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



def save_ROIs(templatePath, coordinates, alignedImage, filename):
	prefixImage = filename.split('.')[0]
	for index, row in coordinates.iterrows():
		# print(row, templatePath.split("/")[-1])
		if row[0] == templatePath.split("/")[-1]:
			# Extract the ROI from the aligned image
			currentROI = alignedImage[row[3]:row[5], row[2]:row[4]]
			# Save the ROI to a file
			cv.imwrite(os.path.join(GLOBALLY_ALIGNED_ROIS[template_name], '_'.join([str(prefixImage), row[1]+".jpg"])), currentROI)

current_template = TEMPLATE_PATHS[TEMPLATE_INDEX]
template_name = current_template.split('/')[-1]
template_image = get_cv_image(current_template)
source = IMAGE_SOURCE[template_name]
# read csv file of coordinates
template_coordinates = pd.read_csv(TEMPLATE_COORDINATES[template_name])

for file in os.scandir(source):
	try:
		filename = file.name
		if not any(ext in filename for ext in ('.jpg', '.png')):
			continue
		print(filename)
		query_image = cv.imread(os.path.join(source, filename), 1)

		aligned_image, M, num_of_inliers = align_query_image(template_image, query_image, "SIFT")

		cv2.imwrite(os.path.join(
			GLOBALLY_ALIGNED_IMAGES[template_name], 
			filename), 
			aligned_image)


	   # Save homography matrix
		with open(os.path.join(GLOBAL_HOMOGRAPHY_MATRIX[template_name], filename.split('.')[0]+'.pkl'), 'wb') as f:
			pickle.dump(M, f)

		# Save ROIs
		save_ROIs(current_template, template_coordinates, aligned_image, filename)

		print("Image aligned successfully")
		
	except Exception as e:
		print(traceback.format_exc())
		pass
