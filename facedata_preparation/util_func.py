
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd

def get_bounding_box_from_landmarks(landmarks, image, scale=2):
    image_shape = image.shape
    # Convert landmarks to a numpy array if it's not already
    landmarks = np.array(landmarks)
    
    # Calculate the initial bounding box
    min_x = np.min(landmarks[:, 0])
    max_x = np.max(landmarks[:, 0])
    min_y = np.min(landmarks[:, 1])
    max_y = np.max(landmarks[:, 1])
    
    # Calculate the center and size of the bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y
    
    # Scale the bounding box
    new_width = width * scale
    new_height = height * scale
    
    # Calculate the new bounding box coordinates
    new_min_x = int(center_x - new_width / 2)
    new_max_x = int(center_x + new_width / 2)
    new_min_y = int(center_y - new_height / 2)
    new_max_y = int(center_y + new_height / 2)
    
    # Handle cases where the bounding box goes out of the image bounds
    img_height, img_width = image_shape[:2]
    
    new_min_x = max(0, new_min_x)
    new_min_y = max(0, new_min_y)
    new_max_x = min(img_width, new_max_x)
    new_max_y = min(img_height, new_max_y)
    
    return (new_min_x, new_min_y, new_max_x, new_max_y)




def rad_to_degree(head_pose):
	return head_pose * 180/np.pi


def draw_sns(distribution, name):
	plt.figure()
	df_head = pd.DataFrame({"Yaw [degree]": distribution[:,1], "Pitch [degree]":distribution[:,0]})
	h = sns.JointGrid(x="Yaw [degree]", y="Pitch [degree]", data=df_head, xlim=(-150,150), ylim=(-150,150))  
	h.ax_joint.set_aspect('equal')         
	h.plot_joint(sns.histplot)                         
	h.ax_marg_x.set_axis_off()
	h.ax_marg_y.set_axis_off()
	h.ax_joint.set_yticks([-80, -40, 0, 40, 80])
	h.ax_joint.set_xticks([-80, -40, 0, 40, 80])
	plt.savefig(name+'.jpg',bbox_inches='tight')


def grid_images(image_list, grid_shape):
	if len(image_list) != grid_shape[0] * grid_shape[1]:
		raise ValueError("Grid shape incompatible with number of images")

	# Split list of images into sublists for each grid row
	rows = [image_list[i:i+grid_shape[1]] for i in range(0, len(image_list), grid_shape[1])]

	# Concatenate images horizontally to make grid rows
	rows = [cv2.hconcat(row) for row in rows]

	# Concatenate images vertically to make grid
	grid = cv2.vconcat(rows)

	return grid



def write_error(text, out_txt_path):
	with open(out_txt_path, 'a') as f:
		f.write( text + '\n')



def set_dummy_camera_model(image=None):
	h, w = image.shape[:2]
	focal_length = w
	center = (w//2, h//2)
	camera_matrix = np.array(
		[[focal_length, 0, center[0]],
		[0, focal_length, center[1]],
		[0, 0, 1]], dtype = "double"
	)
	camera_distortion = np.zeros((1, 5)) # Assuming no lens distortion
	return np.array(camera_matrix), np.array(camera_distortion)


def get_largest_face(preds):
    largest_face = None
    max_area = 0

    for pred in preds:
        # Calculate bounding box for the current face
        x_min = np.min(pred[:, 0])
        y_min = np.min(pred[:, 1])
        x_max = np.max(pred[:, 0])
        y_max = np.max(pred[:, 1])
        
        # Calculate area of the bounding box
        area = (x_max - x_min) * (y_max - y_min)
        
        # Update largest face if the current face has a larger area
        if area > max_area:
            max_area = area
            largest_face = pred

    return largest_face