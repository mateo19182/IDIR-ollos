out_dir + str(i) + '-' + kwargs["network_type"] + '-' + kwargs["loss_function"] + '-' + str(kwargs["lr"]) + '-' + str(kwargs["epochs"]) + '-' + str(kwargs["batch_size"])

find . -type d -exec bash -c 'count=$(find "{}" -type f | wc -l); [ $count -lt 4 ] && rm -rf "{}"' \;


# Run 
- python run.py


# Reference
Based on the Implicit Neural Representations for Deformable Image Registration MIDL 2022 paper

    @inproceedings{wolterink2021implicit,
      title={Implicit Neural Representations for Deformable Image Registration},
      author={Wolterink, Jelmer M and Zwienenberg, Jesse C and Brune, Christoph},
      booktitle={Medical Imaging with Deep Learning 2022}
      year={2022}
    }
