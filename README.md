# README #

ImageNet-tiny contains Matlab code (yes, sorry, made this long ago, before Python became the ML language) to generate a tiny version of the full ImageNet dataset. Below are step-by-step instructions to make your own data or download pre-made datasets of various sizes of images.

## Install Matlab

Yes, unfortunately this piece of code was done before the Python era and therefore you need commercial product, Matlab, to run this code. I am sorry for that. Someone should consider porting code to Octave or Python.

## Install necessary Linux tools

For *convert* command line tool:
```
$ sudo apt install imagemagick
```

## Download data

The original Web site of the challenge is here: [Web page](http://www.image-net.org/challenges/LSVRC/). However, the dataset has been moved to Kaggle.

Register to Kaggle and move to the [dataset page](https://www.kaggle.com/c/imagenet-object-localization-challenge/) agree with the rules and download dataset. Easiest is to use the Kaggle API (assuming you have installed it to your Anaconda environment):

```
~$ cd <MY_DATA_DIR>
~$ kaggle competitions download -c imagenet-object-localization-challenge
```

The dataset is 155GB so it will take some time.

## Download the original Development KIT

The development KIT is still available here (maybe for some time):

 * ILSVRC 2014 [download page](https://image-net.org/challenges/LSVRC/2014/2014-downloads.php)

but since it is likely to disappear we also include the .tgz in our Files/ directory.

```
$ tar zxfv ILSVRC2014_devkit.tgz
$ cd ILSVRC2014_devkit/
```

The development kit functionality is mainly useful to make sure that you evaluate your method correspondingly to the original challenge (their Matlab code is rather terrible, though).

### ILSRC 2014 classification task

The original challenge contains several different tasks, but here we mainly focus on the classification task without detection (no bounding boxes needed), i.e. your method should return the correct class in an input image. In 2014 the task was identified as *Task 2a* and its results under *ordered by classification error* where the winner in 2014 is GoogLeNet with the top-5 error of 0.06656 corresponding to the accuracy of 93.3% correct (the correct class is identified on average in 93 images out of 100 within 5 best "guesses"). That is rather remarkable taking into consideration the difficulty of many classes!

### Evaluation 

The development kit contains the following demo script for evaluating with the "random classifier" output they provide:

* <DEVKITDIR>/evaluation/demo_eval_clsloc.m

You may run it in Matlab and you should get the classification output. However, we have added a modified version of the script to our Matlab/ dir which is easier to configure to your paths etc.

* <IMAGENET-TINY_DIR>/Matlab/demo_eval_clsloc2.m

Edit its config file *devkit_conf.m* to correspond your paths and run in Matlab and you should get the following output:
```
$ cd <IMAGENET-TINY_DIR>/matlab
$ cp devkit_conf.m.example devit_conf.m
EDIT paths in devkit_conf.m to correspond your directory structure
$ nice matlab -nodesktop
>> demo_eval_clsloc2
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


# Generating ImageNet-tiny datasets

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
