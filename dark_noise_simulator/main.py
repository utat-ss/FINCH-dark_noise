# still need the actual numbers for photons per pixel & the number of pixels
import scipy
import numpy as np
from numpy.fft import fft2, fftshift, ifftshift
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.widgets import Slider
import matplotlib.animation as animation
#import datacube
data = np.load('.npy_files/cuprite512.npy')

#visualizing the datacube
data_img = data[:,:,100]

#step 1
#computing photon shot noise
num_photons = 500 #assumption
#num_pixels = 256 #original
num_pixels  = 512 #changed to fit datacube 

mu_p = num_photons * np.ones((num_pixels, num_pixels))

fig, ax = plt.subplots()
img = ax.imshow(mu_p, vmin=400, vmax=600)
image = ax.imshow(data_img)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('No noise')

cb = plt.colorbar(img)
cb.set_label('Photons')

plt.show()

#add random poissonian process 
#set seed so that we can reproducibly generate the same random numbers each time
seed = 42
#associate a randomState instance with that seed
rs = np.random.RandomState(seed)
#Using this RandomState instance, call the poisson method with a mean of num_photons and a size (num_pixels, num_pixels)
shot_noise = rs.poisson(num_photons, (num_pixels, num_pixels))

affected_data = np.copy(data[:,:,100])
dim = np.shape(affected_data)
for i in range(0, dim[0]):
  for j in range(0, dim[1]):
    if shot_noise[i,j] > 512:
      affected_data[i,j] = 0

#plot the image and compare it to the one where no shot noise is present
fig, (ax0, ax1) = plt.subplots(ncols=2)
img0 = ax0.imshow(mu_p, vmin=400, vmax=600)
image0 = ax0.imshow(data_img) #added
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_title('No shot noise')

divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb0 = plt.colorbar(img0, cax=cax)
cb0.set_ticks([])

img1 = ax1.imshow(shot_noise, vmin=400, vmax=600)
image1 = ax1.imshow(affected_data) #added
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Shot noise')

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb1 = plt.colorbar(img1, cax=cax)
cb1.set_label('Photons')

plt.show()

# distribution of photons hitting each pixel.
plt.hist(shot_noise.ravel(), bins = np.arange(350, 650))
plt.xlabel('Number of photons per pixel')
plt.ylabel('Frequency')
plt.show()

#step2
#again, dont have certain values aside from quantum efficiency, assuming the rest
#quantum_efficiency = 0.69
quantum_efficiency = 0.6 #given = >60 ???

# Round the result to ensure that we have a discrete number of electrons
electrons = np.round(quantum_efficiency * shot_noise)

affected_data_2 = np.copy(affected_data)
dim = np.shape(affected_data_2)
for i in range(0, dim[0]):
  for j in range(0, dim[1]):
    if electrons[i,j] > 512:
      affected_data_2[i,j] = 0

fig, (ax0, ax1) = plt.subplots(ncols=2)
img0 = ax0.imshow(shot_noise, vmin=200, vmax=600)
image0 = ax0.imshow(affected_data) #added
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_title('Photons')

divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb0 = plt.colorbar(img0, cax=cax)
cb0.set_ticks([])

img1 = ax1.imshow(electrons, vmin=200, vmax=600)
image1 = ax1.imshow(affected_data_2) #added
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Electrons')

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(img1, cax=cax)

plt.show()

#step3
#calculate dark noise by modelling it as a Gaussian distribution whose standard deviation is equivalent to the dark noise spec of the camera
dark_noise = 2.29 # electrons # also assumption - need exact spec
electrons_out = np.round(rs.normal(scale=dark_noise, size=electrons.shape) + electrons)

affected_data_3 = np.copy(affected_data_2)
dim = np.shape(affected_data_3)
for i in range(0, dim[0]):
  for j in range(0, dim[1]):
    if electrons_out[i,j] > 512:
      affected_data_3[i,j] = 0

fig, (ax0, ax1) = plt.subplots(ncols=2)
img0 = ax0.imshow(electrons, vmin=250, vmax=450)
image0 = ax0.imshow(affected_data_2) #added
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_title('Electrons In')

divider = make_axes_locatable(ax0)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb0 = plt.colorbar(img0, cax=cax)
cb0.set_ticks([])

img1 = ax1.imshow(electrons_out, vmin=250, vmax=450)
image1 = ax1.imshow(affected_data_3) #added
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Electrons Out')

divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb = plt.colorbar(img1, cax=cax)
cb.set_label('Electrons')

plt.show()

# Plot the difference between the two
fig, ax = plt.subplots()
img = ax.imshow(electrons - electrons_out, vmin=-10, vmax=10)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Difference')

cb = plt.colorbar(img)
cb.set_label('Electrons')

plt.show()

#step 4
#convert the value of each pixel from electrons to ADU
sensitivity = 5.88 # ADU/e- #also assumption - need spec
bitdepth = 12 #need exact spec
# ensure that the final ADU count is discrete and to set the maximum upper value of ADU's to the 2k−1 where k is the camera's bit-depth
max_adu = int(2**bitdepth - 1)
#multiply the number of electrons after the addition of read noise by the sensitivity
adu = (electrons_out * sensitivity).astype(int)
adu[adu > max_adu] = max_adu # models pixel saturation

fig, ax = plt.subplots()
img = ax.imshow(adu)
ax.set_xticks([])
ax.set_yticks([])

cb = plt.colorbar(img)
cb.set_label('ADU')

plt.show()

# #step5
# #add a baseline - prevents the number of ADU's from becoming negative at low input signal
# baseline = 100 # ADU #should also be given by the camera model as well
# adu += baseline

# adu[adu > max_adu] = max_adu

# fig, ax = plt.subplots()
# img = ax.imshow(adu)
# ax.set_xticks([])
# ax.set_yticks([])

# cb = plt.colorbar(img)
# cb.set_label('ADU')

# plt.show()
