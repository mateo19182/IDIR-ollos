import os
import imageio.v2 as imageio
from matplotlib import pyplot as plt
from utils import general
from models import models
import numpy as np

current_directory = os.getcwd()
results = []

TARGET = "RFMID"  # "FIRE", "RFMID"
difficulty = ["0_150", "150_300", "300_450", "450_600", "600_750"]

# batch_sizes =    [1000,     10000,   25000,    50000,   100000, 200000]
# learning_rates = [0.000001, 0.00001, 0.000025, 0.00005, 0.0001, 0.0002]  # Adjusted using linear scaling rule for MLP

# batch_sizes =    [1000,     10000,   25000,    50000,   100000, 200000]
# learning_rates = [0.0000001, 0.000001, 0.0000025, 0.000005, 0.00001, 0.00002]  # Adjusted using linear scaling rule for SIREN (10x smaller lr)

learning_rates = [0.00001, 0.00005, 0.0001]   
batch_sizes = [1000, 10000, 50000, 100000]
network_types = ["MLP", "SIREN"] 
lottery = 1  #1

for lr in learning_rates:
    for batch_size in batch_sizes:
        for dif in difficulty:
            results = []  # clear previous results for this configuration run
            kwargs = {}
            kwargs["network_type"] = "SIREN"  # Options are "MLP" and "SIREN"
            kwargs["loss_function"] =  "ncc" #mse, l1, ncc, smoothl1, ssim, huber
            kwargs["lr"] = lr
            kwargs["batch_size"] = batch_size  #max 1.500.000 ... 16 -> 256 -> 65536
            kwargs["phases"] = 1  # 1 is normal, 2 does fiest half with sqrt(batch_size), second half with batch_size, etc...
            kwargs["sampling"] = "random"  # random, weighted, percentage, uniform
            kwargs["epochs"] = 1500  #2500
            kwargs["patience"] = 500
            kwargs["image_shape"] = [1708, 1708]  
            kwargs["hyper_regularization"] = True
            kwargs["alpha_hyper"] = 0.25   #0.25
            kwargs["jacobian_regularization"] = False
            kwargs["alpha_jacobian"] = 0.05  #0.05 default
            kwargs["bending_regularization"] = True
            kwargs["alpha_bending"] = 50.0   #10.0
                    
            kwargs["save_checkpoints"] = False

            data_dir = os.path.join(current_directory, 'data/', TARGET)
            base_out_dir = os.path.join(current_directory, 'out', 'zzz', TARGET, f"{kwargs['network_type']}-{kwargs['lr']}-{kwargs['epochs']}-{kwargs['batch_size']}-{kwargs['sampling']}", dif)
            out_dir = general.create_unique_dir(base_out_dir)

            if TARGET == "FIRE":
                mask_path, feature_mask_path = os.path.join(data_dir, 'Masks', 'mask.png'), os.path.join(data_dir,'Masks', 'feature_mask.png')
                fixed_mask, moving_mask = imageio.imread(mask_path), imageio.imread(feature_mask_path)
                for i in range(14+49, 14+49+70):  #+49+70
                    result = general.load_image_FIRE(i, (data_dir))
                    if result is None:
                        continue
                    else:
                        (fixed_image, moving_image, ground_truth, fixed, moving) = result
                    if i<14:
                        cat = "_A"
                    elif i<63: 
                        cat = "_P"
                    else: cat = "_S"
                    kwargs["save_folder"]= os.path.join(out_dir, str(i+1) + cat + '/')
                    kwargs["mask"] = fixed_mask
                    print(f"Running FIRE {i}")
                    if lottery < 2:
                        model = models.ImplicitRegistrator2d(moving_image, fixed_image, **kwargs)
                    else:
                        model, best_loss = general.select_best_initialization(moving_image, fixed_image, kwargs, num_trials=lottery, plot=False)
                        print(f"Selected initialization with total loss: {best_loss:.6f}")
                    model.fit()
                    registered_img, dfv = model(output_shape=kwargs["image_shape"])
                    
                    results.append(general.test_FIRE(dfv, ground_truth, kwargs["image_shape"], model.save_folder, registered_img, fixed_image, moving_image))
                    general.clean_memory()

            elif TARGET == "RFMID":
                file_path = f"accepted_files_{dif}.txt"
                for i in range(0,25):
                    accepted_files_path = os.path.join(current_directory, f'data/treshRFMID/{file_path}')
                    with open(accepted_files_path, 'r') as f:
                        file_paths = [line.strip() for line in f if line.startswith('data/RFMID/')]
                    result = general.load_image_RFMID(file_paths[i])
                    if result is None:
                        continue
                    else:
                        print(f"File path: {file_paths[i]}")
                        (fixed_image, moving_image, clr_img, full_img, fixed_mask, moving_mask, matrix) = result
                    # print("image sizes: ", fixed_image.shape, moving_image.shape)
                    kwargs["save_folder"]= os.path.join(out_dir, str(i) + '/')
                    kwargs["mask"] = fixed_mask
                    print(f"Running RFMID {i}")
                    if lottery < 2:
                        model = models.ImplicitRegistrator2d(moving_image, fixed_image, **kwargs)
                    else:   
                        model, best_loss = general.select_best_initialization(moving_image, fixed_image, kwargs, num_trials=lottery, plot=False)
                        print(f"Selected initialization with total loss: {best_loss:.6f}")
                    model.fit()
                    registered_img, dfv = model(output_shape=kwargs["image_shape"])

                    results.append(general.test_RFMID(dfv, matrix, kwargs["image_shape"], model.save_folder, registered_img, fixed_image, moving_image, fixed_mask))
                    general.clean_memory()

            # Separate results into individual lists
            auc_list = [result[0] for result in results]
            mean_distance_list = [result[1] for result in results]
            success_rates = np.array([result[2] for result in results])
            thresholds = np.arange(0, 25, 0.1)  # 0.1 to 25.0 in steps of 0.1
            mean_success_rates = np.mean(success_rates, axis=0)
            num_successful_registrations = sum([result[3] for result in results])
            
            with open(os.path.join(out_dir, 'results.txt'), 'w') as f:
                # f.write("auc, mean_distance\n")
                # for result in results:
                #     f.write(f"{result[0]}, {result[1]}\n")
                f.write(f"Mean auc (max 25): {np.mean(auc_list)}\nMean mean_distances: {np.mean(mean_distance_list)}\n")
                f.write(f"Number of successful registrations: {num_successful_registrations}/{len(results)}\n")
                f.write("\nHyperparameters:\n")
                for key, value in kwargs.items():
                    if key != "mask":
                        f.write(f"{key}: {value}\n")
            plt.figure()
            plt.plot(thresholds, mean_success_rates)
            plt.xlabel('Threshold')
            plt.ylabel('Mean Success Rate')
            plt.title('mean Success Rate vs Threshold')
            plt.ylim([0, 1])
            plt.gcf().text(0.02, 0.02, "out_dir", fontsize=8)
            plt.savefig(os.path.join(out_dir, 'eval_all.png'), format='png')
            plt.close('all')  # Close all open figures n  
            print(f"saved final figure to {os.path.join(out_dir, 'eval_all.png')}")


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