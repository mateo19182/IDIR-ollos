import os
import numpy as np
import SimpleITK as sitk
from lungmask import LMInferer
import imageio
import matplotlib.pyplot as plt


# Define the directory containing dcm files
input_dir = "./data"

#cousas hard-codeadas do codigo do git
image_sizes = [
    0,
    [94, 256, 256],
    [112, 256, 256],
    [104, 256, 256],
    [99, 256, 256],
    [106, 256, 256],
    [128, 512, 512],
    [136, 512, 512],
    [128, 512, 512],
    [128, 512, 512],
    [120, 512, 512],
]

# Scale of data, per image pair
voxel_sizes = [
    0,
    [2.5, 0.97, 0.97],
    [2.5, 1.16, 1.16],
    [2.5, 1.15, 1.15],
    [2.5, 1.13, 1.13],
    [2.5, 1.1, 1.1],
    [2.5, 0.97, 0.97],
    [2.5, 0.97, 0.97],
    [2.5, 0.97, 0.97],
    [2.5, 0.97, 0.97],
    [2.5, 0.97, 0.97],
]



# Images
dtype = np.dtype(np.int16)

#pueden acabar en _s, _ssm o nada
for case_num in range(1, 11):  #case1 to case10
    shape = image_sizes[case_num]

    for time_point in ["T00", "T50"]:  # Two time points for each case
        for ending in ["", "_s", "-ssm"]:  # Two time points for each case

            file_name = f"case{case_num}_{time_point}{ending}.img"   #
            case_dir = f"Case{case_num}Pack"
            input_path = os.path.join(input_dir, case_dir, "Images", file_name)
            output_dir = os.path.join(input_dir, case_dir, "Masks")
            mask_name = f"case{case_num}_{time_point}{ending}.mhd" 
            out_path = os.path.join(output_dir, mask_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            #cargamos imagen
            try:
                with open(input_path, "rb") as f:
                    data = np.fromfile(f, dtype)
            except FileNotFoundError:
                continue

            #reshape porque os datos estan gardados en plano, se a forma adecuada
            image_insp = data.reshape(shape)
            #https://github.com/JoHof/lungmask/issues/63
            image_insp_mod = image_insp-1000

            
            #convertimos a simple itk, que non se leva ben con certos tipos
            #enton temos que convertir a unsgined
            itk_img = sitk.GetImageFromArray(image_insp.astype('uint16'))
            '''
            output_name = f"case{case_num}_{time_point}{ending}.dcm"    
            #aqui xa podemos escribir o dicom
            sitk.WriteImage(img, out_path)
            reader = sitk.ImageFileReader()
            reader.SetFileName(out_path)
            image1 = reader.Execute()
            nda = sitk.GetArrayFromImage(image1)

            #comprobamos que son iguales, deberia dar true
            im = imageio.imread(out_path)
            print(im.meta['Modality'])
            print(min(im.flatten()), max(im.flatten()))
            plt.imshow(im, cmap='gray')
            plt.axis('off')
            plt.title('Axial Slice')
            plt.show()
            #lungmask
            inferer = LMInferer()
            input_image = sitk.ReadImage(out_path)
            segmentation = inferer.apply(input_image)  
            mask = sitk.GetImageFromArray(segmentation)
            mask_path = os.path.join(output_dir, mask_name)
            sitk.WriteImage(mask, mask_path)
            '''
            inferer = LMInferer() 
            #segmentación
            segmentation = inferer.apply(image_insp_mod)
            #ten 94 slices, o 64 (por ejemplo) deberia estar polo medio e sacar algo
            #plt.imshow(segmentation[64])
            #plt.show()
            result_out = sitk.GetImageFromArray(segmentation)
            result_out.CopyInformation(itk_img)
            sitk.WriteImage(result_out, out_path)
