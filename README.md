# AI


### NVIDIA CONCEPT
* CUDA-Driver 顯示卡驅動(playgame only need driver)
* CUDA framework, exec compleix colculation.
* CUDA Toolkit compiler,example,library
## Docker NVIDIA
[Nvidia-docker install step](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver)

offical recommand use package manager & only need cuda driver, don't need another cuda packages.

## [Nvidia offical cuda installation document](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation)

* pre-install
* package manger installation.  
* post-install

### package manger
unbuntu: [Meta packages](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-metas) := Deb packages

> deb includes cuda-tools cuda drivers cuda compiler etc.

[cuda-drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#driver-installation) :  
Installs all Driver packages. Handles upgrading(apt upgrade) to the next version of the Driver packages when they're released.)

### apt-cache search version

> apt-cache search --names-only 'cuda-driver'

Driver-installation (specific driver version) in ubuntu use ubuntu-drivers see which driver is recommand.

### ubuntu-drivers
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install ubuntu-drivers-common
ubuntu-drivers devices
```

## When no packages are available, you should use an official "runfile".

### RUNFILES INSTALLTION
* [NVIDIA driver download runfile](https://www.nvidia.com.tw/Download/index.aspx?lang=tw#)
* [NVIDIA offical runfile installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile)

Hit Ctrl+Alt+F1 and login using your credentials.  
kill your current X server session by typing sudo service gdm stop or sudo gdm stop  
Enter runlevel 3 by typing sudo init 3  

> sudo init 3  

Install your *.run file.  
* you change to the directory where you have downloaded the file by typing for instance cd Downloads. If it is in another directory, go there. Check if you see the file when you type ls NVIDIA*  
* Make the file executable with 

 > chmod +x ./your-nvidia-file.run  
 sudo ./your-nvidia-file.run

* You might be required to reboot when the installation finishes. If not, run sudo service gdm start or sudo start gdm to start your X server again. sudo init 5
* It's worth mentioning, that when installed this way, you'd have to __redo the steps after each kernel update.__ becuase [local install](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)



NVIDIA drivers are __backward-compatible(向後支援)__ with CUDA toolkits versions.  
[CUDA Toolkit and Compatible Driver version](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)

### ubuntu offcial installation
```bash
$ sudo ubuntu-drivers devices
$ sudo apt-get install nvidia-driver-{recommand version} nvidia-modprobe
```
## After install CUDA Driver [Nvidia-Docker installation](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-1.0)#installing-version-10):
* [package manager](https://nvidia.github.io/nvidia-docker/)  
$ sudo apt-get install nvidia-docker  
$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit  
$ sudo systemctl restart docker

Basic usage:

> nvidia-docker run --rm nvidia/cuda nvidia-smi  
docker --gpu  
sudo docker run --gpus all nvidia/cuda:9.0-base nvidia-smi


### troubleshoot
[runfile xserver](https://askubuntu.com/questions/149206/how-to-install-nvidia-run)
tell us need to see official document.

The __nvidia-smi__ tool gets installed by the cuda-driver installer

### Tools
* [labelme](https://github.com/wkentaro/labelme)
* [doc.opencv](https://docs.opencv.org/master/)
* [nengoAi](https://www.nengo.ai/)

### Object detection API
* [如何用 TensorFlow object detection API 的 Single Shot MultiBox Detector 來做 hand detection](https://blog.techbridge.cc/2019/02/16/ssd-hand-detection-with-tensorflow-object-detection-api/)
* [openpose](https://blog.techbridge.cc/2019/01/18/openpose-installation/)

### Hand Recognition

### google MediaPipe
* Goal:
static hand recognition
dynamic hand recognition
hand controller

* ref.
* * https://google.github.io/mediapipe/solutions/hands#python
* * https://github.com/google/mediapipe
* * https://allen108108.github.io/blog/2020/06/06/%E6%89%8B%E9%83%A8%E9%97%9C%E9%8D%B5%E9%BB%9E%20(Hand%20Keypoints)%20%E7%9A%84%E9%A0%90%E6%B8%AC%E5%8F%8A%E5%85%B6%E6%87%89%E7%94%A8/

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
