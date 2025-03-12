import h5py
import imageio
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cm as cm
from matplotlib import rc
import matplotlib.font_manager
import pandas as pd
import seaborn as sns
from tqdm import tqdm
sns.set_theme(font_scale=1.5)


font = {'family' : 'sans-serif',
		'serif': 'Helvetica'}
matplotlib.rc('font', **font)

def myplot(x,y,s,bins=128):
	# rg = np.array([[-80, 80], [-80, 80]])s
	rg = np.array([[-120, 120], [-120, 120]])
	heatmap, xedges, yedges = np.histogram2d(x,y,bins=bins,density=True,range=rg)
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
	return heatmap.T, extent

def plot_heatmap(save_path, to_plot, s):
	x = to_plot[:,1]
	y = to_plot[:,0]
	img, extent = myplot(x,y,s)
	# extent=[-80,80,-80,80]
	# plt.figure()
	plt.clf()
	plt.imshow(img,extent=extent,origin='lower', cmap=cm.jet)
	plt.savefig(save_path)
	
def rad_to_degree(head_pose):
	return head_pose * 180/np.pi



def draw_two_fig(samples, name):

	plot_heatmap(name+'_hm.jpg',samples,16)

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