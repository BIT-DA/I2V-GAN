# I2V-GAN  
This repository is the official Pytorch implementation for ACMMM2021 paper  
"I2V-GAN: Unpaired Infrared-to-Visible Video Translation".  

#### Traffic I2V Example:
![compair_gif01](img/Comparison01.gif)

#### Monitoring I2V Example:
![compair_gif02](img/Comparison02.gif)

#### Flower Translation Example:
![compair_gif03](img/Comparison03.gif)

## Introduction  
### Abstract  
Human vision is often adversely affected by complex environmental factors, especially in night vision scenarios. Thus, infrared cameras are often leveraged to help enhance the visual effects via detecting infrared radiation in the surrounding environment, but the infrared videos are undesirable due to the lack of detailed semantic information. In such a case, an effective video-to-video translation method from the infrared domain to the visible counterpart is strongly needed by overcoming the intrinsic huge gap between infrared and visible fields.  
Our work propose an infrared-to-visible (I2V) video translation method I2V-GAN to generate fine-grained and spatial-temporal consistent visible light video by given an unpaired infrared video.  
The backbone network follows Cycle-GAN and Recycle-GAN.  
![compaire](img/compair.png)


Technically, our model capitalizes on three types of constraints: adversarial constraint to generate synthetic frame that is similar to the real one, cyclic consistency with the introduced perceptual loss for effective content conversion as well as style preservation, and similarity constraint across and within domains to enhance the content and motion consistency in both spatial and temporal spaces at a fine-grained level. 

![network-all](img/network.png)

### IRVI Dataset
Click [here]() to download IRVI dataset.

![data_samples](img/samples.png)

## Installation
The code is implemented with `Python(3.6)` and `Pytorch(1.9.0)` for `CUDA Version 11.2`

Install dependencies:  
`pip install -r requirements.txt`

## Usage

### Train
```
python train.py --dataroot /path/to/dataset \
--display_env visdom_env_name --name exp_name \
--model i2vgan --which_model_netG resnet_6blocks \
--which_model_netP unet_128 --npf 8 --dataset_mode unaligned_triplet
```

### Test
```
python test.py --dataroot /path/to/dataset \
--which_epoch latest --name exp_name --model i2vgan \
--which_model_netG resnet_6blocks --which_model_netP unet_128 \
--dataset_mode unaligned --no_dropout --loadSize 256
```

## Citation
If you find our work useful in your research or publication, please cite our work:  
```

```
