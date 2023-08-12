import cv2
import tkinter as tk
import csv
import time  # Import the time module
from tkinter import Entry, Button, Label
from local_variables import *  # Make sure TEMPLATE_PATHS and TEMPLATE_INDEX are defined

# Variables to store the coordinates and text
x1, y1, x2, y2 = 0, 0, 0, 0
text_input = ""
marked_coordinates = []

def end_program():
	save_to_csv()
	root.destroy()  # Use root.destroy() to close the GUI window and end the program

# ...

def click_event(event, x, y, flags, params):
	global x1, y1, x2, y2
	if event == cv2.EVENT_LBUTTONDOWN:
		if x1 == 0 and y1 == 0:  # First coordinate
			x1, y1 = x, y
			cv2.circle(img_with_dots, (x1, y1), 5, (0, 255, 0), -1)  # Green dot for first point
			update_image()
			
		elif x2 == 0 and y2 == 0:  # Second coordinate
			x2, y2 = x, y
			cv2.circle(img_with_dots, (x2, y2), 5, (0, 0, 255), -1)  # Red dot for second point
			update_image()
			update_labels()

def update_labels():
	x1_label.config(text=f"UL_X: {x1}, UL_Y: {y1}")
	x2_label.config(text=f"BR_X: {x2}, BR_Y: {y2}")

def update_image():
	cv2.imshow('image', img_with_dots)

def submit_coordinates():
	global text_input, x1, y1, x2, y2
	text_input = text_entry.get()
	marked_coordinates.append([x1, y1, x2, y2, text_input])  # Include name_input
	x1, y1, x2, y2 = 0, 0, 0, 0
	update_labels()
	text_entry.delete(0, 'end')  # Clear text entry
	save_to_csv()

def save_to_csv():
	global template_name
	file_path = "ROI_info/rois.csv"
	with open(file_path, 'w', newline='') as f:
		csv_writer = csv.writer(f)
		csv_writer.writerow(["Attribute", "UL_X", "UL_Y", "BR_X", "BR_Y"])
		for coord in marked_coordinates:
			csv_writer.writerow([template_name, coord[4], coord[0], coord[1], coord[2], coord[3]])
	print("Data saved to:", file_path)

# Create the main GUI window
root = tk.Tk()
root.title("Coordinate Marker")

# Load the image
template_path = TEMPLATE_PATHS[TEMPLATE_INDEX]
template_name = TEMPLATE_PATHS[TEMPLATE_INDEX].split("/")[-1]
img = cv2.imread(template_path, 1)


# Create an image copy to add dots
img_with_dots = img.copy()

# Display the image
update_image()
cv2.setMouseCallback('image', click_event)

# Create labels for coordinates
x1_label = Label(root, text="UL_X: 0, UL_Y: 0", padx=10)
x1_label.pack()
x2_label = Label(root, text="BR_X: 0, BR_Y: 0", padx=10)
x2_label.pack()

# Create text entry
text_label = Label(root, text="Attribute:")
text_label.pack()
text_entry = Entry(root, width=30)
text_entry.pack()


# Create submit button
submit_button = Button(root, text="Submit", command=submit_coordinates)
submit_button.pack()

# Create end button
end_button = Button(root, text="End", command=end_program)
end_button.pack()

# Get screen width and height for centering the GUI
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate the position to center the GUI
x_position = (screen_width - root.winfo_reqwidth()) // 2
y_position = (screen_height - root.winfo_reqheight()) // 2 + 10

# Position the GUI window in the center of the screen
root.geometry(f"+{x_position}+{y_position}")

# Start the GUI main loop
root.mainloop()

# Close OpenCV windows
cv2.destroyAllWindows()
