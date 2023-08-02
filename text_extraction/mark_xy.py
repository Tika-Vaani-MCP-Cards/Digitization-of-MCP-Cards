# importing the module
import cv2
from local_variables import *

# function to display the coordinates of
# of the points clicked on the image
point_list = []
def click_event(event, x, y, flags, params):

	# checking for left mouse clicks
	if event == cv2.EVENT_LBUTTONDOWN:

		# displaying the coordinates
		# on the Shell
		print(x, ' ', y)
		point_list.append([x,y])

		# displaying the coordinates
		# on the image window
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img, str(x) + ',' +
					str(y), (x,y), font,
					1, (255, 0, 0), 2)
		cv2.imshow('image', img)

	# checking for right mouse clicks	
	if event==cv2.EVENT_RBUTTONDOWN:

		# displaying the coordinates
		# on the Shell
		print(x, ' ', y)
		point_list.append([x,y])
		# displaying the coordinates
		# on the image window
		font = cv2.FONT_HERSHEY_SIMPLEX
		b = img[y, x, 0]
		g = img[y, x, 1]
		r = img[y, x, 2]
		cv2.putText(img, str(b) + ',' +
					str(g) + ',' + str(r),
					(x,y), font, 1,
					(255, 255, 0), 2)
		cv2.imshow('image', img)

# driver function
if __name__=="__main__":

	# mention template path here
	template_path = TEMPLATE_PATHS[TEMPLATE_INDEX]

	img = cv2.imread(template_path, 1)

	# displaying the image
	cv2.imshow('image', img)

	# setting mouse handler for the image
	# and calling the click_event() function
	cv2.setMouseCallback('image', click_event)

	# wait for a key to be pressed to exit
	cv2.waitKey(0)

	new_point_list = []
	print(point_list)
	i = 0
	while i < len(point_list):
		print(i, i+1, len(point_list))
		x1 = point_list[i][0]; y1 = point_list[i][1]
		x2 = point_list[i+1][0]; y2 = point_list[i+1][1]

		new_point_list.append([[x1,y1], [x2,y2]])
		i += 2
	

	print(new_point_list)
		
	# close the window
	cv2.destroyAllWindows()
