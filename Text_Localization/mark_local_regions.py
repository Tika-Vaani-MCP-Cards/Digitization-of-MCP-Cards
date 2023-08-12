import cv2
import tkinter as tk
import csv
from itertools import cycle
from tkinter import Entry, Button, Label
from local_variables import *

# Load the image, change TEMPLATE_INDEX to change the image
TEMPLATE_INDEX = 1
template_path = TEMPLATE_PATHS[TEMPLATE_INDEX]
template_name = TEMPLATE_PATHS[TEMPLATE_INDEX].split("/")[-1]
img = cv2.imread(template_path, 1)

# Variables to store the coordinates, text, and colors
x1, y1, x2, y2 = 0, 0, 0, 0
marked_coordinates = []
dot_colors = cycle([(0, 255, 0), (0, 0, 255), (255, 0, 0)])  # Define a cycle of colors

def end_program():
    save_to_csv()
    root.destroy()

def click_event(event, x, y, flags, params):
    global img, x1, y1, x2, y2, dot_colors

    if event == cv2.EVENT_LBUTTONDOWN:
        if x1 == 0 and y1 == 0:  # First coordinate
            x1, y1 = x, y
            color = next(dot_colors)
            cv2.circle(img, (x1, y1), 5, color, -1)  # Change dot color
            update_image(img)

        elif x2 == 0 and y2 == 0:  # Second coordinate
            x2, y2 = x, y
            color = next(dot_colors)
            cv2.circle(img, (x2, y2), 5, color, -1)  # Change dot color
            update_image(img)
            update_labels()

def update_labels():
    x1_label.config(text=f"UL_X: {x1}, UL_Y: {y1}")
    x2_label.config(text=f"BR_X: {x2}, BR_Y: {y2}")

def update_image(img):
    cv2.imshow('image', img)

def submit_coordinates():
    global x1, y1, x2, y2
    marked_coordinates.append([x1, y1, x2, y2])
    x1, y1, x2, y2 = 0, 0, 0, 0
    update_labels()
    save_to_csv()
    update_image(img)

def save_to_csv():
    global template_name
    file_path = f"ROI_info/{template_name}_local_regions.csv"
    with open(file_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["UL_X", "UL_Y", "BR_X", "BR_Y"])
        for coord in marked_coordinates:
            csv_writer.writerow([coord[0], coord[1], coord[2], coord[3]])
    print("Data saved to:", file_path)

# Create the main GUI window
root = tk.Tk()
root.title("Coordinate Marker")

# Display the image
update_image(img)
cv2.setMouseCallback('image', click_event)

# Create labels for coordinates
x1_label = Label(root, text="UL_X: 0, UL_Y: 0", padx=10)
x1_label.pack()
x2_label = Label(root, text="BR_X: 0, BR_Y: 0", padx=10)
x2_label.pack()

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
