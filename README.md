# Low Shot Box Correction for Weakly Supervised Object Detection

This code repo is built on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).


## Installation and Preparation

Firstly, clone the code

```
git clone https://github.com/ptx9363/BCNet.git
```

and then follow [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) 's preparation to install the environment and dependency. This repo's specific dependencies are shown below:
* Python 3.5.6
* Torch 0.4.1
* Torchvision 0.2.1
* Numpy 1.15.4

### Prepare dataset
We use VOC2007 dataset in our most experiments. We have run weakly-supervised method, [OICR](https://github.com/ppengtang/oicr),  to provide pseudo bounding boxes for images in VOC2007. Some of our experiments are trained from weakly pre-trained models. In general, we provide all of pretrained models and generated labels here.

* VOC2007 dataset with pseudo labels, [data](https://drive.google.com/open?id=15ZhFEOedbjR8Z05LBJJxOzdE9SL8vVIX)
* Pretrained models, [models](https://drive.google.com/open?id=1YLpcGVKluahKMHK2pO0lng4mpVsdgTMC)
* Edge boxes proposals, [data](https://drive.google.com/open?id=1YvfLfsVb0pU-51pKtRikNQ_SR-B8plMV)

the final data folder should be placed like:

```
BCNet/data/pretrained_model
      data/VOCdevkit/VOC2007
      data/edge_boxes_data
```

## Citation

    @article{jjfaster2rcnn,
        Author = {Jianwei Yang and Jiasen Lu and Dhruv Batra and Devi Parikh},
        Title = {A Faster Pytorch Implementation of Faster R-CNN},
        Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
        Year = {2017}
    }

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
