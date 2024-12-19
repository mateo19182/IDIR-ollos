import os
import imageio.v2 as imageio
from matplotlib import pyplot as plt
from scipy import integrate
from utils import general
from models import models
import torch
import numpy as np
import optuna

current_directory = os.getcwd()
results = []

TARGET = "FIRE"  # "FIRE", "RFMID"

def objective(trial):
    kwargs = {
        "loss_function": "ncc",
        "lr": trial.suggest_float('lr', 1e-5, 1e-2),
        "epochs": 3000,
        "batch_size": trial.suggest_int('batch_size', 5000, 30000),
        "patience": 500,
        "image_shape": [2000, 2000],
        "hyper_regularization": False,
        "jacobian_regularization": False,
        "bending_regularization": True,
        "network_type": "MLP",
        "save_checkpoints": False
    }

    if TARGET == "RFMID":
        data_dir = os.path.join(current_directory, 'data/', 'RFMID')
        base_out_dir = os.path.join(current_directory, 'out', 'RFMID', f"{kwargs['network_type']}-{kwargs['lr']}-{kwargs['epochs']}-{kwargs['batch_size']}")
        out_dir = general.create_unique_dir(base_out_dir)

        auc_list = []
        mean_distance_list = []
        success_rates_list = []

        for i in range(1, 10):  # Reduced range for faster optimization
            result = general.load_image_RFMID(f"{data_dir}/Testing_{i}.npz")
            if result is None:
                continue
            (fixed_image, moving_image, clr_img, full_img, fixed_mask, moving_mask, matrix) = result

            kwargs["save_folder"] = os.path.join(out_dir, str(i) + '/')
            kwargs["mask"] = fixed_mask
            print(f"Running RFMID {i}")
            ImpReg = models.ImplicitRegistrator2d(moving_image, fixed_image, **kwargs)
            ImpReg.fit()
            registered_img, dfv = ImpReg(output_shape=kwargs["image_shape"])

            result = general.test_RFMID(dfv, matrix, kwargs["image_shape"], ImpReg.save_folder, registered_img, fixed_image, moving_image, fixed_mask)
            auc_list.append(result[0])
            mean_distance_list.append(result[1])
            success_rates_list.append(result[2])

        mean_auc = np.mean(auc_list)
        mean_distance = np.mean(mean_distance_list)
        mean_success_rates = np.mean(success_rates_list, axis=0)
        auc = integrate.trapeziod  (mean_success_rates, np.arange(0, 25, 0.1))

        return -auc  # We want to maximize AUC, so we return negative value

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)  # Adjust n_trials as needed

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Use the best parameters for a final run
best_params = trial.params
kwargs = {
    "loss_function": "ncc",
    "lr": best_params['lr'],
    "epochs": 3000,
    "batch_size": best_params['batch_size'],
    "patience": 500,
    "image_shape": [2000, 2000],
    "hyper_regularization": False,
    "jacobian_regularization": False,
    "bending_regularization": True,
    "network_type": "MLP",
    "save_checkpoints": False
}
