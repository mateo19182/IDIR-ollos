import os
import imageio.v2 as imageio
from matplotlib import pyplot as plt
from utils import general
from models import models
import numpy as np


current_directory = os.getcwd()
out_dir = os.path.join(current_directory, 'out')

'''
#FIRE
data_dir = os.path.join(current_directory, 'data', 'FIRE')
saved_images = []
saved_images_names = []
mask_path, feature_mask_path = os.path.join(data_dir, 'Masks', 'mask.png'), os.path.join(data_dir,'Masks', 'feature_mask.png')
fixed_mask, moving_mask = imageio.imread(mask_path), imageio.imread(feature_mask_path)
for i in range(8, 41): 
    (fixed_image, moving_image, ground_truth, fixed, moving) = general.load_image_FIRE(i, (data_dir))
    kwargs = {}
    kwargs["verbose"] = True
    kwargs["hyper_regularization"] = False
    kwargs["jacobian_regularization"] = False
    kwargs["bending_regularization"] = True
    kwargs["network_type"] = "MLP"  # Options are "MLP" and "SIREN"
    kwargs["save_folder"] = out_dir + str(i)
    kwargs["mask"] = fixed_mask

    #dfv = np.load('dfv.npy')

    ImpReg = models.ImplicitRegistrator2d(moving_image, fixed_image, **kwargs)
    ImpReg.fit()
    registered_img, dfv = ImpReg()

    images = [fixed_image, moving_image, registered_img, moving_mask] 
    image_names = ['fixed_image', 'moving_image', 'transform Image', 'geo_mask Image']

    general.display_dfv(registered_img, dfv, fixed, moving)
    #general.display_grid(dfv)
    #general.display_images(images, image_names, 'gray')
    #general.test_accuracy(transformation, ground_truth)
    #saved_images.append(registered_img)
    #saved_images_names.append("MLP no regularization")
'''

#------------------------------------------------------------------------------------

#RFMID
data_dir = os.path.join(current_directory, 'data', 'RFMID')
saved_images = []
saved_images_names = []
for i in range(8, 41): 
    (og_img, geo_img, clr_img, full_img, mask, geo_mask, original) = general.load_image_RFMID(f"{data_dir}/Testing_{i}.npz")
    kwargs = {}
    kwargs["verbose"] = True
    kwargs["hyper_regularization"] = False
    kwargs["jacobian_regularization"] = False
    kwargs["bending_regularization"] = True
    kwargs["network_type"] = "MLP"  # Options are "MLP" and "SIREN"
    kwargs["save_folder"] = out_dir + str(i)
    kwargs["mask"] = mask

    ImpReg = models.ImplicitRegistrator2d(geo_img, og_img, **kwargs)
    ImpReg.fit()

    registered_img, dfv = ImpReg()

    images = [og_img, geo_img, registered_img, geo_mask] 
    image_names = ['Original Image', 'Geometric Image', 'transform Image', 'geo_mask Image']
    general.display_dfv(registered_img, dfv, og_img.numpy(), geo_img.numpy())
    registered_img = ImpReg()
    #print(registered_img.shape)
    #resized_image=cv2.cvtColor(cv2.resize(original, (500, 500)), cv2.COLOR_BGR2GRAY)
    images = [og_img, geo_img, registered_img, geo_mask] 
    image_names = ['Original Image', 'Geometric Image', 'transform Image', 'geo_mask Image']
    #general.display_images(images, image_names, 'gray')
    #print("{} {} {}".format(i, accuracy_mean, accuracy_std))
    #saved_images.append(registered_img)
    #saved_images_names.append("MLP no regularization")

#general.display_images(saved_images, saved_images_names, 'gray')

#------------------------------------------------------------------------------------
'''
#IDIR
data_dir = os.path.join(current_directory, 'data', 'IDIR')

for i in range(6, 11): 
    case_id = i

    (
        img_insp,
        img_exp,
        landmarks_insp,
        landmarks_exp,
        mask_exp,
        voxel_size,
    ) = general.load_image_DIRLab(case_id, "{}/Case".format(data_dir))


    kwargs = {}
    kwargs["verbose"] = True
    kwargs["hyper_regularization"] = False
    kwargs["jacobian_regularization"] = False
    kwargs["bending_regularization"] = True
    kwargs["network_type"] = "SIREN"  # Options are "MLP" and "SIREN"
    kwargs["save_folder"] = out_dir + str(case_id)
    kwargs["mask"] = mask_exp
    ImpReg = models.ImplicitRegistrator(img_exp, img_insp, **kwargs)
    ImpReg.fit()
    new_landmarks_orig, _ = general.compute_landmarks(
        ImpReg.network, landmarks_insp, image_size=img_insp.shape
    )

    #print(voxel_size)
    accuracy_mean, accuracy_std = general.compute_landmark_accuracy(
        new_landmarks_orig, landmarks_exp, voxel_size=voxel_size
    )

    print("{} {} {}".format(case_id, accuracy_mean, accuracy_std))
'''