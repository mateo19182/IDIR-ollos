import os
import imageio.v2 as imageio
from matplotlib import pyplot as plt
from utils import general
from models import models
import numpy as np

current_directory = os.getcwd()
results = []

targets = ["RFMID", "FIRE"]  # "FIRE", "RFMID"
difficulty = ["0_150", "150_300", "300_450", "450_600", "600_750"]
learning_rates = [0.000001,0.00001, 0.0001, 0.001, 0.01]   
batch_sizes = [10000]
network_types = ["MLP", "SIREN"] 
# Removed regs sweep and using default regularization values:
default_reg = [0.5, 0, 50]

lottery = 1  #1

for type in network_types:    
    for target in targets:
        for lr in learning_rates:

            results = []  # clear previous results for this configuration run
            kwargs = {}
            kwargs["network_type"] = type  # Options are "MLP" and "SIREN"
            kwargs["loss_function"] =  "ncc" #mse, l1, ncc, smoothl1, ssim, huber
            kwargs["lr"] = lr
            kwargs["batch_size"] = 10000  #max 1.500.000 ... 16 -> 256 -> 65536
            kwargs["phases"] = 1  # 1 is normal, 2 does fiest half with sqrt(batch_size), second half with batch_size, etc...
            kwargs["sampling"] = "random"  # random, weighted, percentage, uniform
            kwargs["epochs"] = 1500  #2500
            kwargs["patience"] = 500
            kwargs["image_shape"] = [1000, 1000] 
            kwargs["hyper_regularization"] = True
            kwargs["alpha_hyper"] = default_reg[0]  # Fixed default value
            kwargs["jacobian_regularization"] = True
            kwargs["alpha_jacobian"] = default_reg[1]  # Fixed default value
            kwargs["bending_regularization"] = True
            kwargs["alpha_bending"] = default_reg[2]   # Fixed default value
                    
            kwargs["save_checkpoints"] = False

            data_dir = os.path.join(current_directory, 'data/', target)
            base_out_dir = os.path.join(current_directory, 'out', 'lr', target, f"{kwargs['network_type']}-{kwargs['lr']}-{kwargs['epochs']}-{kwargs['batch_size']}")
            out_dir = general.create_unique_dir(base_out_dir)

            if target == "FIRE":
                mask_path = os.path.join(data_dir, 'Masks', 'mask.png')
                feature_mask_path = os.path.join(data_dir,'Masks', 'feature_mask.png')
                fixed_mask = imageio.imread(mask_path)
                moving_mask = imageio.imread(feature_mask_path)
                for i in range(14+49, 14+49+70):  #14+49+70
                    result = general.load_image_FIRE(i, (data_dir))
                    if result is None or i%3 != 0:
                        continue
                    else:
                        (fixed_image, moving_image, ground_truth, fixed, moving) = result
                    if i < 14:
                        cat = "_A"
                    elif i < 63: 
                        cat = "_P"
                    else: 
                        cat = "_S"
                    kwargs["save_folder"] = os.path.join(out_dir, str(i+1) + cat + '/')
                    kwargs["mask"] = fixed_mask
                    print(f"Running FIRE {i} with lr {lr}")
                    if lottery < 2:
                        model = models.ImplicitRegistrator2d(moving_image, fixed_image, **kwargs)
                    else:
                        model, best_loss = general.select_best_initialization(moving_image, fixed_image, kwargs, num_trials=lottery, plot=False)
                        print(f"Selected initialization with total loss: {best_loss:.6f}")
                    model.fit()
                    registered_img, dfv = model(output_shape=kwargs["image_shape"])
                    
                    results.append(general.test_FIRE(dfv, ground_truth, kwargs["image_shape"], model.save_folder, registered_img, fixed_image, moving_image))
                    general.clean_memory()

            elif target == "RFMID":
                for i in range(0,250,3):
                    result = general.load_image_RFMID(f"{data_dir}/Testing_{i}.npz")
                    if result is None:
                        continue
                    else:
                        print(f"File path: {data_dir}/Testing_{i}.npz")
                        (fixed_image, moving_image, clr_img, full_img, fixed_mask, moving_mask, matrix) = result
                    kwargs["save_folder"] = os.path.join(out_dir, str(i) + '/')
                    kwargs["mask"] = fixed_mask
                    print(f"Running RFMID {i} with lr {lr}")
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
            plt.title('Mean Success Rate vs Threshold')
            plt.ylim([0, 1])
            plt.gcf().text(0.02, 0.02, "out_dir", fontsize=8)
            plt.savefig(os.path.join(out_dir, 'eval_all.png'), format='png')
            plt.close('all')
            print(f"Saved final figure to {os.path.join(out_dir, 'eval_all.png')}")