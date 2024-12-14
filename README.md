# Aliñamento de imaxes oftalmolóxicas usando representacións neuronais implícitas

## Training

```python
ImpReg = models.ImplicitRegistrator2d(moving_image, fixed_image, kwargs)
  self.possible_coordinate_tensor = general.make_masked_coordinate_tensor_2d(self.mask,self.fixed_image.shape)
```

- crea un tensor con todas las coordenadas válidas de la imagen fija

ImpReg.fit
training_iteration

```python
indices = torch.randperm(self.possible_coordinate_tensor.shape[0],...)[: self.batch_size]
coordinate_tensor = self.possible_coordinate_tensor[indices, :]
```

- Seleciona 'batch_size' numero aleatorio de puntos válidos en la fixed image

```python
output = self.network(coordinate_tensor)
output = torch.add(output, coordinate_tensor)
transformed_image = self.transform_no_add(coord_temp)
fixed_image = general.bilinear_interpolation(
    self.fixed_image,
    coordinate_tensor[:, 0],
    coordinate_tensor[:, 1],
)

```

- Se pasa coordinate_tensor por la net, devuelve un vector de desplazamiento por cada coord
- se le suma el output a las coordenadas originales para producir dfv
- se le aplica a la imagen móvil para conseguir la imagen transformada.
- se samplean las mismas coordenadas en la imagen fija

```python
loss += self.criterion(transformed_image, fixed_image)
loss += self.alpha_bending * regularizers.compute_bending_energy_2d(
        coordinate_tensor, output_rel, batch_size=self.batch_size
    )
loss.backward()
```

- se computa el loss y se aplican los terminos de regulacion
- backprop

## Eval

ImpReg __ call

```python
general.make_coordinate_tensor_2d(output_shape)
output = self.network(coordinate_tensor)
coord_temp = torch.add(output, coordinate_tensor)
transformed_image = self.transform_no_add(coord_temp)
```

- crea tensor completo del tamaño requerido
- lo pasa por la red, coord_temp contiene todas las deformaciones, se le suman al tensor original para obtener el dfv
- devuelve dfv y transformed-img

general.test_FIRE // general.test_RFMID

```python


```

## Other

out_dir + str(i) + '-' + kwargs["network_type"] + '-' + kwargs["loss_function"] + '-' + str(kwargs["lr"]) + '-' + str(kwargs["epochs"]) + '-' + str(kwargs["batch_size"])

find . -type d -exec bash -c 'count=$(find "{}" -type f | wc -l); [ $count -lt 4 ] && rm -rf "{}"' \;

## Run

- python run.py

## Reference

Based on the Implicit Neural Representations for Deformable Image Registration MIDL 2022 paper

```js
    @inproceedings{wolterink2021implicit,
      title={Implicit Neural Representations for Deformable Image Registration},
      author={Wolterink, Jelmer M and Zwienenberg, Jesse C and Brune, Christoph},
      booktitle={Medical Imaging with Deep Learning 2022}
      year={2022}
    }
```
