# README #

ImageNet-tiny contains scripts to generate a "tiny" version of the ImageNet
Large Scale Visual Classification Challenge
(see http://www.image-net.org/challenges/LSVRC/). Tiny in terms of image
sizes and number of classes. These scripts help to make more machine learning
classes oriented tiny challenges that can be solved during exercises by
machine learning students.

### Author(s) ###

* Joni Kamarainen [http://vision.cs.tut.fi](http://vision.cs.tut.fi)

** Table of Contents **

[TOC]

# Introduction

The aim of this project is to provide a platform to study machine learning (ML) methods using realistic challenging data. The particular emphasis is on vision problems since they are easy to visualise and entertaining for ML students. The basic workflow of the most machine learning systems is depicted below and we provide basic functionality to experiment and study importance of each stage.
![machine_learning_workflow_small.png](https://bitbucket.org/repo/AnroBb/images/437448844-machine_learning_workflow_small.png)

You will find out that your machine learning engineering work is always tied to the specific problem and dataset and therefore we have divided the Wiki to the main sections that are dataset specific, but under each section you find details of 1) data selection, 2) data preprocessing, 3) classifier selection and its meta-parameter tuning and 4) final evaluation. Each section provides a complete solution or multiple solutions for each data set and it is up to you to dispatch them to real applications and make some money.

## Publications

The following scientific publications have been made using this code (please, cite when appropriate):

* Incremental Convolutional Neural Network Training (Y. Liu, Y. Qian, K. Chen, J.-K. Kämäräinen, H. Huttunen, L. Fan, J. Saarinen), In ICPR 1st Int. Workshop on Deep Learning for Pattern Recognition, 2016. [PDF](http://vision.cs.tut.fi/data/publications/icpr2016ws_incremental_cnn.pdf)

# Dataset 1: ImageNet Large Scale Visual Recognition Challenge - ILSVRC

See the challenge [Web page](http://www.image-net.org/challenges/LSVRC/) for more details. The data itself contains millions of images of thousands of different object categories (see [http://www.image-net.org/](http://www.image-net.org/)). Currently (2014) this challenge is one of the grand challenges in computer vision and machine learning and many top research groups all around the world are working on it (including also research centres of big companies like Google, Microsoft, Facebook etc.)

## Get data

Register via the challenge Web page and they will provide you a link via which you can download the full image sets (within the next 7 days).

** NOTE: ** We provide several ready-generated ImageNet-tiny datasets in the download section of this repository. Then you don't need to download the original data at all!

## Development KIT

The development kit that is mainly useful to run the evaluation scripts for comparable numbers and is available at the project Web page (code is not very well document and certainly not elegant):

 * Development kit 2014 ([download](http://image-net.org/image/ilsvrc2014/ILSVRC2014_devkit.tgz))

The challenge contains several different tasks, but here we mainly focus on the classification task without detection (no bounding boxes needed), i.e. your method should return the correct class in an input image. In 2014 the task was identified as **Task 2a** and its results under *ordered by classification error* where the winner in 2014 is GoogLeNet with the top-5 accuracy of 0.06656 corresponding to the accuracy of 93.3% correct (the correct class is identified on average in 93 images out of 100 within 5 best "guesses"). That is rather remarkable taking into consideration the difficulty of many classes!

### Evaluation 

The development kit contains the following demo script for evaluating with the "random classifier" output they provide:

* <DEVKITDIR>/evaluation/demo_eval_clsloc.m

You may run it in Matlab and you should get the classification output. However, that is pretty crappy and therefore you may found our own version of the same script

* <IMAGENET-TINY_DIR>/Matlab/demo_eval_clsloc2.m

Edit its config file *devkit_conf.m* to correspond your paths and run in Matlab and you should get the following output:
```
....
....
# guesses vs clsloc error vs cls-only error
    1.0000         0    0.9992
    2.0000         0    0.9982
    3.0000         0    0.9971
    4.0000         0    0.9962
    5.0000         0    0.9955

>>            
```
That is, top-1 error is 0.9992 which is pretty close to the random guess (1-1/1000=0.999 for 1000 classes with equal number) and top-5 0.9955 (random: 1-5/1000=0.995).

The same evaluation code you may use to evaluate your own classifier output by just changing the name of the prediction file.


# Dataset 2: ImageNet-tiny dataset

The main idea is that we otherwise replicate the original ImageNet challenge, but downscale it in terms of the number of classes and size of the images. This way we can easily fit data into available memory and run training and testing in minutes or hours instead of days as it often happens with the original data. We wish this way to help to study different machine learning methods with true "big data" problem that is up-to-date challenge also for the researchers. Ftor the tiny datasets we give descriptive names such as:

* ImageNet-tiny-8x8-gray-5
* ImageNet-tiny-16x16-sRGB-full
* etc.

That describes the size of the images, colour coding and the number of classes. In the following, we explain how you can make your own dataset. It is also easy to extend the functionality for your personal needs!

## Pre-made datasets

See the download section of this repo: [Downloads](https://bitbucket.org/kamarain/imagenet-tiny/downloads)

## Generating ImageNet-tiny

This is very simple if you have downloaded the original images somewhere in your disk:
```
~$ cd <IMAGENET-TINY_DIR>/Matlab
~$ cp  generate_ImageNet_tiny_conf.m.example generate_ImageNet_tiny_conf.m
EDIT: generate_ImageNet_tiny_conf.m // set paths correctly etc.
~$ matlab -nodesktop
>> generate_ImageNet_tiny
```
That will take some time to generate and copy those tiny images to your new "ImageNet-tiny/" directory.

** NOTE: ** The bounding boxes in the ImageNet_tiny data set are wrong as we don't re-make them to correspond tiny images. We are only interested on the classification task at the moment!

** NOTE: ** The size of the data set extracted to your hard disk can be several gigabytes - not because of the size of the images, but due to the minimum size of the disk data blocks (on my system 4kB making it the smallest possible size of an image file, no matter how small resolution).

### SLURM based generation
If you have a suitable SLURM environment available, then you don't want to burn down your laptop or desktop. As above, edit the configure file and run:

```
~$ cd <IMAGENET-TINY_DIR>/Matlab
~$ cp  generate_ImageNet_tiny_conf.m.example generate_ImageNet_tiny_conf.m
EDIT: generate_ImageNet_tiny_conf.m // set paths correctly etc.
~$ cp  generate_ImageNet_tiny_slurm.sh.example generate_ImageNet_tiny_slurm.sh
EDIT: generate_ImageNet_tiny_slurm.sh // set paths correctly etc.
~$ sbatch generate_ImageNet_tiny_slurm.sh
```
This will generate log and err files where you can monitor progress. Note that at the moment we don't support indexing so if you wish to run several conversions on parallel, you need to execute the shell
script in different directories (otherwise log and err files will overwrite).


### Decreasing the number of classes
For students it may be useful to play with less than 1,000 classes since then running their code is faster and classification percentages even with weaker methods are better. Start first by generating the full ImageNet-tiny dataset of your preferred images sizes etc. 

Run the following command:
```
~$ cd <IMAGENET-TINY_DIR>/Matlab
~$ cp  make_tiny_eval_clsloc_conf.m.example make_tiny_eval_clsloc_conf.m
EDIT: make_tiny_eval_clsloc_conf.m // set paths and select classes
~$ matlab -nodesktop
>> make_tiny_eval_clsloc
```
This file generates new files for evaluation and my editing *devkit_conf.m* you may run evaluation for the newly generated sub-set of ImageNet classes.

# Classifier 1: Convolutional Deep Neural Network (DCNN) ([Caffe](http://caffe.berkeleyvision.org))

## Installing Caffe (Ubuntu 14.04) CPU-only
Basically follow the instructions at [http://caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html) which should help you to install the system on any distribution (start with the CPUonly option):
to
```
~$ sudo apt-get update
~$ sudo apt-get install libatlas-base-dev python-dev build-essential pkg-config cmake cmake-qt-gui libprotobuf-dev protobuf-compiler libgflags-dev libhdf5-serial-dev libleveldb-dev liblmdb-dev libgoogle-glog-dev libsnappy-dev
```
Install a local version of the OpenCV as described in [https://bitbucket.org/kamarain/mvprmatlab/wiki/Home](https://bitbucket.org/kamarain/mvprmatlab/wiki/Home) and

```
~$ cd <CAFFE_SRC_DIR>
~$ cp Makefile.config.example Makefile.config
[EDIT Makefile.config add <OPENCVDIR>/build/install/include and <OPENCVDIR>/build/install/lib directories to  INCLUDE_DIRS and  LIBRARY_DIRS]
~$ make all
~$ make test
~$ make runtest
```

## Local Caffe Installation (Ubuntu 14.04) CPU-only
You may prefer for some reason to avoid using system level libraries or they are not available and you want to avoid interaction with your system administrators. This means that you need to install every possible required library locally and compile Caffe against them. In the following are a few special tricks that might be useful:

**protoc**
protoc tool is required by Caffe and since the system wide "protoc" binary is available and by default used you may receive errors related to *incompatible versions*.

First install the protoc:
```
:::text
~$ cd <EXT_DIR>
~$ git clone https://github.com/google/protobuf.git
~$ cd protobuf
[Check readme and compile]
~$ cd <CAFFE_DIR>
~$ ln -s <EXT_DIR>/protobuf/install/bin/protoc
[EDIT Makefile.config]
  Replace
    $(Q)protoc --proto_path=$(PROTO_SRC_DIR) --cpp_out=$(PROTO_BUILD_DIR) $< 
  With
    ./protoc --proto_path=$(PROTO_SRC_DIR) --cpp_out=$(PROTO_BUILD_DIR) $<
```

**boost**
Download the latest Boost C++ library sources, compile and check the output for Include and
Library directories which you need to add to Caffe Makefile.config.

Add Include and library directories to their local installation dirs:
```
:::text
[Edit Makefile.config , for example, in my system:]
INCLUDE_DIRS := \
              /home/kamarain/Work/ext/boost_1_56_0/install/include \
              /home/kamarain/Work/ext/gflags/build/install/include \
              /home/kamarain/Work/ext/glog-0.3.3/install/include \
              /home/kamarain/Work/ext/protobuf/install/include \
              /home/kamarain/Work/ext/ATLAS3.10.0/include \
              /home/kamarain/Work/ext/leveldb/include \
              /home/kamarain/Work/ext/opencv-2.4.4/build/install/include \
              /home/kamarain/Work/ext/hdf5-1.9.195/install/include \
              /home/kamarain/Work/ext/mdb/libraries/liblmdb
LIBRARY_DIRS := /home/kamarain/Work/ext/boost_1_56_0/install/lib \
             /home/kamarain/Work/ext/gflags/build/install/lib \
             /home/kamarain/Work/ext/glog-0.3.3/install/lib \
              /home/kamarain/Work/ext/protobuf/install/lib \
              /home/kamarain/Work/ext/leveldb/ \
              /home/kamarain/Work/ext/opencv-2.4.4/build/install/lib \
              /home/kamarain/Work/ext/hdf5-1.9.195/install/lib \
              /home/kamarain/Work/ext/mdb/libraries/liblmdb \
              /home/kamarain/Work/ext/caffe
```

Now, everything should compile just by
```
:::text
~$ make all
~$ make test
~$ make runtest
```



## Adding GPU support (CUDA 6.5)
Basically everything goes just as above, but first you need to install CUDA and prior to that you need to install working drivers for your GPU card. In Ubuntu 14.04 I followed the instructions from [http://askubuntu.com/questions/451221/ubuntu-14-04-install-nvidia-driver](http://askubuntu.com/questions/451221/ubuntu-14-04-install-nvidia-driver) where you have two options:

* simple option: *sudo apt-get install nvidia-304 nvidia-304-updates*
* less simple option: look the answer starting "You can download the driver for your video card for Ubuntu..." - that worked with me, but after every kernel update I need to run *sudo nvidia-xconfig*

When finished and re-booted, then follow the CUDA installation instructions [here](http://www.r-tutor.com/gpu-computing/cuda-installation/cuda6.5-ubuntu) and recompile Caffe as above instructed (first run "make clean").

## Examples
We have used the Caffe functionality for training and testing, but altered them the way it is easy to test different networks and solvers for the same data. For example, to run Caffe training for 8x8 gray level images using the solver 1 and networks 1, 2, 3 and 4, you need to execute the following (note that execution of *create* and *train* functionality will take some time):
```
$ ./create_imagenet-tiny.sh defs/defs_imagenet-tiny-8x8-sRGB-net-1.sh
$ ./make_mean_imagenet-tiny.sh defs/defs_imagenet-tiny-8x8-sRGB-net-1.sh
$ ./train_imagenet-tiny.sh defs/defs_imagenet-tiny-8x8-sRGB-net-1.sh
$ ./train_imagenet-tiny.sh defs/defs_imagenet-tiny-8x8-sRGB-net-2.sh
$ ./train_imagenet-tiny.sh defs/defs_imagenet-tiny-8x8-sRGB-net-3.sh
$ ./train_imagenet-tiny.sh defs/defs_imagenet-tiny-8x8-sRGB-net-4.sh
```
You may wish to plot the validation results that is now written into TEMPWORK/LOG* files. To do that you can parse these files, e.g.,
```
$ source ../tools/parse_log_for_scores.sh TEMPWORK/LOG_caffe_ImageNet-tiny-8x8-sRGB-net-1.txt > 8x8-sRGB-net-1.dat
$ source ../tools/parse_log_for_scores.sh TEMPWORK/LOG_caffe_ImageNet-tiny-8x8-sRGB-net-2.txt > 8x8-sRGB-net-2.dat
$ source ../tools/parse_log_for_scores.sh TEMPWORK/LOG_caffe_ImageNet-tiny-8x8-sRGB-net-3.txt > 8x8-sRGB-net-3.dat
$ source ../tools/parse_log_for_scores.sh TEMPWORK/LOG_caffe_ImageNet-tiny-8x8-sRGB-net-4.txt > 8x8-sRGB-net-4.dat
```
Now, let's write a gnuplot file to plot the graphs:

```
#!bash 
# exp_8x8.gp gnuplot commands to plot and store extracted Caffe log data

set terminal postscript eps enhanced mono "Helvetica" 20
set encoding iso_8859_1
set out 'exp_8x8.eps'
parsefile1 = '8x8-sRGB-net-1.dat'
parsefile2 = '8x8-sRGB-net-2.dat'
parsefile3 = '8x8-sRGB-net-3.dat'
parsefile4 = '8x8-sRGB-net-4.dat'

set grid
set key right bottom
set style line 1 lc rgb '#aa0000' lt 1 lw 2 pt 4 ps 1.5 pi 5
set style line 2 lc rgb '#aa0000' lt 1 lw 2 pt 5 ps 1.5 pi 5
set style line 3 lc rgb '#aa0000' lt 2 lw 2 pt 6 ps 1.5 pi 5
set style line 4 lc rgb '#aa0000' lt 3 lw 2 pt 7 ps 1.5 pi 5
plot parsefile1 every 5 with linespoints ls 1 t "8x8-sRGB/net-1", \
     parsefile2 every 5 with linespoints ls 2 t "8x8-sRGB/net-2", \
     parsefile3 every 5 with linespoints ls 3 t "8x8-sRGB/net-3", \
     parsefile4 every 5 with linespoints ls 4 t "8x8-sRGB/net-4"
```
Then you execute it and plot:
```
$ gnuplot exp_8x8.gp
$ epspdf exp_8x8.eps
$ evince exp_8x8.pdf
```
![exp_8x8.png](https://bitbucket.org/repo/AnroBb/images/1660873890-exp_8x8.png)

# Final evaluation results for selected datasets
For all reported methods this repository also contains source code and scripts to replicate them.

Data                                |  Year  |  Method  |  top-1 err  |  top-5 err  |
------------------------------  |  ------  |  ----------  |  -----------  |  -----------  |
ImageNet-tiny-8x8-gray | 2014  |  caffe       | XX.X        |     YY.Y     |

