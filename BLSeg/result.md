Each model can choose any backbone without any modification

|       Backbone \ Model        | **FCN** | **U-Net** | **PSPNet** | **DeepLab v3+** | **GCN** |
| :---------------------------: | :-----: | :-------: | :--------: | :-------------: | :-----: |
|           **VGG16**           | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |
|       **MobileNet v1**        | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |
|       **MobileNet v2**        | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |
|         **ResNet34**          | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |
|         **ResNet50**          | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |
|        **SE ResNet34**        | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |
|        **SE ResNet50**        | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |
| **Modified Aligned Xception** | &radic; |  &radic;  |  &radic;   |     &radic;     | &radic; |

Model pre-trained on augmented PASCAL VOC2012 dataset with 10582 images for training and 1449 images for validation.

## Parameters

|       Backbone \ Model        | **FCN** | **U-Net** | **PSPNet** | **DeepLab v3+** | **GCN** |
| :---------------------------: | ------: | --------: | ---------: | --------------: | ------: |
|           **VGG16**           | 134.41M |    25.26M |     19.70M |          20.15M |  14.99M |
|       **MobileNet v1**        | 225.66M |    14.01M |     13.70M |          12.44M |   3.58M |
|       **MobileNet v2**        | 276.06M |     2.67M |     15.67M |          13.35M |   2.51M |
|         **ResNet34**          | 140.98M |    24.08M |     26.27M |          26.71M |  21.48M |
|         **ResNet50**          | 451.51M |    66.35M |     46.61M |          40.37M |  24.24M |
|        **SE ResNet34**        | 141.14M |    24.25M |     26.43M |          26.87M |  21.64M |
|        **SE ResNet50**        | 454.02M |    69.03M |     49.12M |          42.88M |  26.76M |
| **Modified Aligned Xception** | 465.85M |    57.46M |     60.95M |          54.70M |  38.46M |
