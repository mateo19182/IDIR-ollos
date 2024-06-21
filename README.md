# To Do




- evals
  - desplazamiento 0 epochs -> weights are initialized on netwroks.py -> are those good?
    - DFVs -> are they scale dependant? no lo entiendo aun
      - 0s no deformation
  - RFMID puntos no aleatorios

- find good parameters
  - epochs, learning rate

- voxelmorph?

FIRE(naturales): 18(low overlap), 19, (medium overlap)
RFMID(simuladas): 558, 559
out_dir + str(i) + '-' + kwargs["network_type"] + '-' + kwargs["loss_function"] + '-' + str(kwargs["lr"]) + '-' + str(kwargs["epochs"]) + '-' + str(kwargs["batch_size"])


regularizacion con learning rate
stride usado en el estimador de la drivada. si infuye minibatch es uyn parametro relevante.¡ bending...



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
