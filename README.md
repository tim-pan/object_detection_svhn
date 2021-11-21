# object_detection_svhn
### object detection
---
for TA: if you want to reproduce the submission
you can download the `inference.ipynb` and run all
---
## task
digit detection[8]
## Environment
training:
- pytorch1.10.2
- colab GPU(Tesla P100)
- numpy1.20.3
## introduction
there are 
- 4 ipynb files

**1. main.ipynb**
this file is composed of all training procedures with my best model architecture.

**2. experiment.ipynb**
all the experiment implement code in the report.pdf on e3.

**3. inference.ipynb**
modified from the ipynb file TA attached.

**4. prepare_svhn.ipynb**
preprocess the svhn dataset(i.e. convert mat file to the json file, and save it in the dataset folder)

## reproduce
- download `inference.ipynb` and just run it

**note** 
The model para and all py files you need to reproduce the submission are embedded in the `inference.ipynb`, but if you want to check my weight para is normal, please follow this [link](https://drive.google.com/file/d/1fP4ZEEfgL05pj3hRZ8oqmaZ0uK1SM6Vi/view?usp=sharing)
## reference
[1] Focal Loss for Dense Object Detection. by Tsung-Yi Lin, et al.</br>
[2] https://github.com/yhenon/pytorch-retinanet</br>
[3]https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html </br>
[4]https://github.com/pytorch/vision/tree/main/references/detection</br>
[5]Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. by Shaoqing Ren, et al.</br>
[6]https://colab.research.google.com/github/ozanpekmezci/capstone/blob/master/prepare-svhn.ipynb </br>
[7]Bag of Freebies for Training Object Detection Neural Networks. By Zhi Zhang, et al.</br>
[8]Reading Digits in Natural Images with Unsupervised Feature Learning. By 
Yuval Netzer, et al.</br>
[9]Microsoft COCO: Common Objects in Context. By Tsung-Yi Lin, et al.</br>



