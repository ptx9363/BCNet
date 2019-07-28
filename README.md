# Low Shot Box Correction for Weakly Supervised Object Detection

This code repo is built on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).


## Installation and Preparation

Firstly, clone the code

```
git clone https://github.com/ptx9363/BCNet.git
```

and then follow [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) 's preparation guide to install the environment and dependancy.


### Prepare dataset
We use VOC2007 dataset in our most experiments. We have run weakly-supervised method to provide pseudo bounding boxes for images in VOC2007. Some of our experiments are trained from weakly pre-trained models. In general, we provide all of pretrained models and generated labels here.

* VOC2007 dataset with pseudo labels, data
* Pretrained models, models
* Edge boxes proposals, data

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
