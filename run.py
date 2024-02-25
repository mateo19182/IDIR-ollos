from utils import general
from models import models

out_dir = "/home/mateo/uni/cuarto/TFG/IDIR/out"

#RFMID
data_dir = "/home/mateo/uni/cuarto/TFG/IDIR/data/RFMID"

for i in range(1): 
    (og_img, geo_img, clr_img, full_img, mask, geo_mask) = general.load_image_RFMID(1, "{}/Testing_1.npz".format(data_dir))

    kwargs = {}
    kwargs["verbose"] = True
    kwargs["hyper_regularization"] = False
    kwargs["jacobian_regularization"] = False
    kwargs["bending_regularization"] = True
    kwargs["network_type"] = "SIREN"  # Options are "MLP" and "SIREN"
    kwargs["save_folder"] = out_dir + str(i)
    kwargs["mask"] = mask

    ImpReg = models.ImplicitRegistrator2d(geo_img, og_img, **kwargs)
    ImpReg.fit()
    #print("{} {} {}".format(i,
    #  accuracy_mean, accuracy_std))

    print("--------------------")

#----------------------------------------------------------------------
'''
data_dir = "/home/mateo/uni/cuarto/TFG/IDIR/data/IDIR"

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
    #print(img_insp.shape)=torch.Size([128, 512, 512])
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