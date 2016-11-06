# tensorflow-fcn
This code uses [Fully Convolutional Networks](http://arxiv.org/abs/1411.4038) in Tensorflow to solve the semantic segmentation problem. The post traininf process is not included.

Deconvolution Layers are initialized as bilinear upsampling. Conv and FCN layer weights using VGG weights. Numpy load is used to read VGG weights.<b>The .npy file for <a href="https://dl.dropboxusercontent.com/u/50333326/vgg16.npy">VGG16</a> however need to be downloaded before using this needwork.</b>

## Usage

use readimage.py for image storage.<br />
run train_pipeline.py for training process.<br />
change the checkpoint path in test.py and run it for testing<br />
(there're already 200 pic in the test_data folder for reference)

## Requirement

tensorflow 0.10<br />
keras<br />
skimage<br />
cv2<br />
numpy<br />
matplotlib<br />

## result
![figure_1](https://cloud.githubusercontent.com/assets/17188890/20035003/ce1be06e-a3a9-11e6-9157-23c03b3f3fe7.png)

