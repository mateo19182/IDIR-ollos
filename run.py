import os
import imageio.v2 as imageio
from matplotlib import pyplot as plt
from utils import general
from models import models
import torch
import numpy as np

current_directory = os.getcwd()

'''
#FIRE

data_dir = os.path.join(current_directory, 'data', 'FIRE')
out_dir = os.path.join(current_directory, 'out/', 'FIRE/')

saved_images = []
saved_images_names = []
mask_path, feature_mask_path = os.path.join(data_dir, 'Masks', 'mask.png'), os.path.join(data_dir,'Masks', 'feature_mask.png')
fixed_mask, moving_mask = imageio.imread(mask_path), imageio.imread(feature_mask_path)
#for i in range(0, 5):
for i in [0 , 10, 20, 30, 100, 110]:
    (fixed_image, moving_image, ground_truth, fixed, moving) = general.load_image_FIRE(i, (data_dir))
    kwargs = {}
    kwargs["loss_function"] = "ncc" #mse, l1, ncc, smoothl1, ssim, huber
    kwargs["lr"] = 0.00001
    kwargs["epochs"] = 2000    #2500
    kwargs["batch_size"] = 20000   #10000
    kwargs["image_shape"] = [2000, 2000]
    kwargs["hyper_regularization"] = False
    kwargs["jacobian_regularization"] = False
    kwargs["bending_regularization"] = True
    kwargs["network_type"] = "MLP"  # Options are "MLP" and "SIREN"
    kwargs["save_folder"]= out_dir + str(i) + '-' + kwargs["network_type"] + '-' + kwargs["loss_function"] + '-' + str(kwargs["lr"]) + '-' + str(kwargs["epochs"]) + '-' + str(kwargs["batch_size"])
    kwargs["mask"] = fixed_mask
    kwargs["save_checkpoints"] = False

    #dfv = np.load('dfv_01.npy')
    ImpReg = models.ImplicitRegistrator2d(moving_image, fixed_image, **kwargs)
    ImpReg.fit()
    registered_img, dfv = ImpReg(output_shape=kwargs["image_shape"])
    #np.save('dfv_01.npy', dfv)

    images = [fixed_image, moving_image, registered_img, moving_mask] 
    image_names = ['fixed_image', 'moving_image', 'transform Image', 'geo_mask Image']
    #general.display_dfv(registered_img, dfv, fixed, moving, ImpReg.save_folder)
    general.test_FIRE(dfv, ground_truth, kwargs["image_shape"], ImpReg.save_folder, registered_img, fixed_image, moving_image)
    
    #general.clean_memory()

'''
#------------------------------------------------------------------------------------
#RFMID
data_dir = os.path.join(current_directory, 'data/', 'RFMID')
out_dir = os.path.join(current_directory, 'out/', 'RFMID/')

saved_images = []
saved_images_names = []
for i in range(1, 50): 
    result = general.load_image_RFMID(f"{data_dir}/Testing_{i}.npz")
    if result is None:
        continue
    else:
        (fixed_image, moving_image, clr_img, full_img, fixed_mask, moving_mask, matrix) = result
    kwargs = {}
    kwargs["loss_function"] = "ncc" #mse, l1, ncc, smoothl1, ssim, huber
    kwargs["lr"] = 0.00001
    kwargs["epochs"] = 2500   #2500
    kwargs["batch_size"] = 20000   #10000
    kwargs["image_shape"] = [1708, 1708]
    kwargs["hyper_regularization"] = False
    kwargs["jacobian_regularization"] = False
    kwargs["bending_regularization"] = True
    kwargs["network_type"] = "MLP"  # Options are "MLP" and "SIREN"
    kwargs["save_folder"]= out_dir + str(i) + '-' + kwargs["network_type"] + '-' + kwargs["loss_function"] + '-' + str(kwargs["lr"]) + '-' + str(kwargs["epochs"]) + '-' + str(kwargs["batch_size"])
    kwargs["mask"] = fixed_mask
    kwargs["save_checkpoints"] = False

    #dfv = np.load('dfv_01.npy')
    ImpReg = models.ImplicitRegistrator2d(moving_image, fixed_image, **kwargs)
    ImpReg.fit()
    registered_img, dfv = ImpReg(output_shape=kwargs["image_shape"])
    #np.save('dfv_01.npy', dfv)

    images = [fixed_image, moving_image, registered_img, moving_mask] 
    image_names = ['fixed_image', 'moving_image', 'transform Image', 'geo_mask Image']
    #general.display_dfv(registered_img, dfv, fixed, moving, ImpReg.save_folder)
    general.test_RFMID(dfv, matrix, kwargs["image_shape"], ImpReg.save_folder, registered_img, fixed_image, moving_image, fixed_mask)
    
    general.clean_memory()


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