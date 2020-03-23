# BLSeg (BaseLine Segmentation)


PyTorch's Semantic Segmentation Toolbox

- [BLSeg (BaseLine Segmentation)](#blseg-baseline-segmentation)
  - [Requirement](#requirement)
  - [Supported Module](#supported-module)
  - [Docs](#docs)
  - [References](#references)

## Requirement

- Python 3
- PyTorch >= 1.0.0

## Supported Module

- Backbone
  - [VGG16]
  - [MobileNet v1] (1.0)
  - [MobileNet v2] (1.0)
  - [ResNet 34]
  - [ResNet 50] (Modified according to [Bag of Tricks])
  - [SE ResNet 34]
  - [SE ResNet 50] (Modified according to [Bag of Tricks])
  - [Modified Aligned Xception]
- Model
  - [FCN]
  - [U-Net]
  - [PSPNet]
  - [DeepLab v3+]
  - [GCN] (Large Kernel Matters)
- Loss
  - BCEWithLogitsLossWithOHEM
  - CrossEntropyLossWithOHEM
  - DiceLoss (only for binary classification)
  - SoftCrossEntropyLossWithOHEM
- Metric
  - Pixel Accuracy
  - Mean IoU



## Docs
See [Docs](Docs.md)


## References

- Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
- Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." arXiv preprint arXiv:1704.04861 (2017).
- Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.
- He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
- Xie, Junyuan, et al. "Bag of tricks for image classification with convolutional neural networks." arXiv preprint arXiv:1812.01187 (2018).
- Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
- Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
- Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
- Zhao, Hengshuang, et al. "Pyramid scene parsing network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
- Chen, Liang-Chieh, et al. "Encoder-decoder with atrous separable convolution for semantic image segmentation." Proceedings of the European Conference on Computer Vision (ECCV). 2018.
- Peng, Chao, et al. "Large Kernel Matters--Improve Semantic Segmentation by Global Convolutional Network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

---

[VGG16]:https://arxiv.org/abs/1409.1556
[MobileNet v1]:https://arxiv.org/abs/1704.04861
[MobileNet v2]:https://arxiv.org/abs/1801.04381
[ResNet 34]:https://arxiv.org/abs/1512.03385
[ResNet 50]:https://arxiv.org/abs/1512.03385
[SE ResNet 34]:https://arxiv.org/abs/1709.01507
[SE ResNet 50]:https://arxiv.org/abs/1709.01507
[Modified Aligned Xception]:https://arxiv.org/abs/1802.02611
[Bag of Tricks]:https://arxiv.org/abs/1812.01187

[FCN]:https://arxiv.org/abs/1411.4038
[U-Net]:https://arxiv.org/abs/1505.04597
[PSPNet]:https://arxiv.org/abs/1612.01105
[DeepLab v3+]:https://arxiv.org/abs/1802.02611
[GCN]:https://arxiv.org/abs/1703.02719




