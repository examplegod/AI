# AI


### NVIDIA CONCEPT
* CUDA-Driver 顯示卡驅動(playgame only need driver)
* CUDA framework, exec compleix colculation.
* CUDA Toolkit compiler,example,library
### Docker NVIDIA
* [NVIDIA driver install .run](https://www.nvidia.com.tw/Download/index.aspx?lang=tw#)

[Nvidia-docker install step](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver)
Nvidia-docker only need cuda driver, don't need another cuda packages.
recommended package manager and install cuda-driver 
[Nvidia offical install document](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation)
* preinstall
* package manger installation.  
unbuntu: [Meta packages](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-metas) := Deb packages

> deb includes cuda-tools cuda drivers cuda compiler etc.

[cuda-drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#driver-installation) (Installs all Driver packages. Handles upgrading(apt upgrade) to the next version of the Driver packages when they're released.)

check version
* ubuntu-driver
* apt-cache
Driver-installation (specific driver version) in ubuntu use ubuntu-drivers see which driver is recommand.
### ubuntu-drivers
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices
```
### apt-cahe

> apt-cache search --names-only 'cuda-driver'

### When no packages are available, you should use an official "runfile".
[official runfine web](https://www.nvidia.com/Download/index.aspx?lang=en-us)
[step](https://www.rosdyanakusuma.com/tutorials/how-to-install-nvidia-driver-in-ubuntu-18-04/)
* Install NVIDIA Driver via Installer (Runfile)
* login gui tty  ctrl+alt+F7


NVIDIA drivers are backward-compatible(向後支援) with CUDA toolkits versions.
[CUDA Toolkit and Compatible Driver version](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)

### ubuntu offcial installation
```bash
$ sudo ubuntu-drivers devices
$ sudo apt-get install nvidia-driver-418 nvidia-modprobe
```
## [Nvidia-Docker installation](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-1.0)#installing-version-10):
* [package manager](https://nvidia.github.io/nvidia-docker/)
$ sudo apt-get install nvidia-docker

Basic usage:

> nvidia-docker run --rm nvidia/cuda nvidia-smi






The __nvidia-smi__ tool gets installed by the cuda-driver installer


### Tools
* [labelme](https://github.com/wkentaro/labelme)
* [doc.opencv](https://docs.opencv.org/master/)
* [nengoAi](https://www.nengo.ai/)

### Object detection API
* [如何用 TensorFlow object detection API 的 Single Shot MultiBox Detector 來做 hand detection](https://blog.techbridge.cc/2019/02/16/ssd-hand-detection-with-tensorflow-object-detection-api/)
* [openpose](https://blog.techbridge.cc/2019/01/18/openpose-installation/)

### Hand Recognition
* paper
[Hand Keypoint Detection in Single Images using Multiview Bootstrapping](https://arxiv.org/pdf/1704.07809.pdf)

* code
[hand-keypoint-detection-using-deep-learning-and-opencv](https://www.learnopencv.com/hand-keypoint-detection-using-deep-learning-and-opencv/)
[github openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

* data
kaggle:
https://www.kaggle.com/gti-upm/leaphandgestuav?

* svm,knn

### Face Recognition
* code
[face_recognition](https://github.com/ageitgey/face_recognition)

### Object Tracking
* [simplest method on opencv](https://docs.opencv.org/master/df/d9d/tutorial_py_colorspaces.html)
* TensorFlow object detection API
  + [如何用 TensorFlow object detection API 的 Single Shot MultiBox Detector 來做 hand detection](https://blog.techbridge.cc/2019/02/16/ssd-hand-detection-with-tensorflow-object-detection-api/)
  + [tensorflow-object-detection-api-tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)
  + [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
* R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN, YOLO, SSD
* [(mAP)mean average precision, Precision, recall](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173)
 

## Ref.
* https://www.coderbridge.com/@Po-Jen?page=1
* https://chtseng.wordpress.com/category/%e5%bf%83%e5%be%97-%e6%a9%9f%e5%99%a8%e5%ad%b8%e7%bf%92/
  + triplet loss
  + Labelme & COCO
  + Image Labeling
