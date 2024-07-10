# To Do

[[[-0.95334136 -1.0002785 ]
  [-0.95336145 -0.9982536 ]
  [-0.95338315 -0.9962281 ]
  ...
  ...
  [-0.9702296   1.0075067 ]
  [-0.97020715  1.0094862 ]
  [-0.97018415  1.0114652 ]]
 ...
 [[ 1.0539523  -0.9816803 ]
  [ 1.0539688  -0.97965467]
  [ 1.0539851  -0.977629  ]
  ...
  ...
  [ 1.0882883   0.96304315]
  [ 1.0883181   0.96496636]
  [ 1.0883479   0.96688956]]]

////

  [[[-0.6092379  -1.8250524 ]
  [-0.60941815 -1.8220845 ]
  [-0.6095985  -1.8191168 ]
  ...
  ...
  [-0.9712848   0.9351052 ]
  [-0.9714899   0.9368382 ]
  [-0.97169346  0.93857145]]

 ...

 [[ 0.90765893 -1.153735  ]
  [ 0.90750957 -1.1509731 ]
  [ 0.90736026 -1.1482111 ]
  ...
  ...
  [ 0.74457526  0.7051491 ]
  [ 0.7445846   0.7066877 ]
  [ 0.744594    0.7082263 ]]]


- evals
  - desplazamiento 0 epochs algo hay -> weights are initialized on netwroks.py -> are those good? (los d MLP arriba izq siempre?)
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
