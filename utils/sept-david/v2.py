from skimage.transform import rescale, resize, downscale_local_mean
from skimage import io

import numpy as np
import torch
from matplotlib import pyplot as plt
from itertools import product

#nothing changed
def bilinear_interpolation(input_array, x_indices, y_indices):
	# Convert indices from [-1, 1] to [0, width-1] or [0, height-1]
	x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
	y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
	# Get the four surrounding pixel coordinates
	x0 = torch.floor(x_indices).to(torch.long)
	y0 = torch.floor(y_indices).to(torch.long)
	x1 = x0 + 1
	y1 = y0 + 1
	# Clamp the coordinates to be within the image bounds
	x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
	y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
	x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
	y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
	# Calculate the interpolation weights
	x = x_indices - x0
	y = y_indices - y0
	# Perform bilinear interpolation
	output = (
		input_array[x0, y0] * (1 - x) * (1 - y) +
		input_array[x1, y0] * x * (1 - y) +
		input_array[x0, y1] * (1 - x) * y +
		input_array[x1, y1] * x * y
	)
	return output



def convert(input):
	return (input +1) * (1500 - 1) * 0.5



fixed = resize(io.imread("./A01_1.jpg", as_gray=True),(1500,1500))
data = open("./control_points_A01_1_2.txt", "r")

og_points = []
mov_points = []
for i in data:
	choped = i.split(" ")
	og_points.append((float(choped[0]),float(choped[1])))
	mov_points.append((float(choped[2]),float(choped[3])))


#create synthetic dfv
df_simple = np.zeros((1500,1500,2))

#should be bound between -1 and 1
df_simple[:,:,0] = 0.1
df_simple[:,:,1] = 0.1


x_indices = df_simple[:,:,0]
y_indices = df_simple[:,:,1]

#convert from -1 to 1 to 0.. width and height
x_indices = (x_indices +1) * (1500 - 1) * 0.5
y_indices = (y_indices+ 1) * (1500 - 1) * 0.5


mapx_base, mapy_base = np.meshgrid(np.arange(-1,1,2/1500), np.arange(-1,1,2/1500))

mapx = mapx_base - 0.1
mapy = mapy_base - 0.1



dfs = np.stack([mapy, mapx], axis=2).reshape((2250000, 2)) # for the interp function
dfsimple = np.stack([mapx, mapy], axis=2)#for addressing, easier for me



moved = bilinear_interpolation( torch.from_numpy(fixed), torch.from_numpy(dfs[:, 0]), torch.from_numpy(dfs[:, 1])).reshape(1500,1500)

fig, axs = plt.subplots(2, 2)

axs[0,0].imshow(fixed,cmap="gray")
axs[0,1].imshow(moved,cmap="gray")


factor = 1500/2912
plt.imshow(fixed)

#we are going to plot  the moved points. Typically, these would be in the ground truth
#file. However, since this transformation is synth we will create them as we
#know the precise translations used
moved_points = []

axs[1,1].imshow(moved,cmap="gray")

for i in og_points:

	#why 75? the range is -1 to 1, meaning it has size of 2.
	#we can imagine the image divided in 4 quadrans with the 0,0 in 
	#the image center, which if its 1500x1500 would be at 750,750.
	#All of this means that "0.1" in one direction is equivalent to
	#(0.1*1500)/2 pixels, which is 75 for 1500
	moved_points.append((i[0]*factor+75, i[1]*factor+75))
	axs[1,1].scatter(i[0]*factor+75, i[1]*factor+75,s=1)




arr1 = [-2,-1,0,1,2]
arr2 = [-5,-4,-3,-2,-1,0,1,2,3,4,5,6]

k = list(product(arr2, arr2)) #diferentes tamaños para que se vexan mellor
#plot the original fixed points
for i in og_points:
	for kp in k:
		fixed[int(np.round(i[1]*factor))-kp[0]][int(np.round(i[0]*factor))-kp[1]] = 1

k = list(product(arr1, arr1))
for i in moved_points:

	xn, yn = dfsimple[int(np.round(i[0])), int(np.round(i[1])),:]
	xn = convert(xn)
	yn = convert(yn)

	for kp in k:
		fixed[int(np.round(xn))-kp[0]][int(np.round(yn))-kp[1]] = 0




axs[1,0].imshow(fixed,cmap="gray")





plt.savefig("./test_scatter2.png", dpi=300)
plt.close()


