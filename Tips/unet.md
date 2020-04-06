U-Net存在的不足。

    U-Net的encoder部分由若干卷积层和池化层组成，由于他们都是local的运算，只能看到局部的信息，因此需要通过堆叠多层来提取长距离信息，这种方式较为低效，参数量大，计算量也大。过多的下采样导致更多空间信息的损失（U-Net下采样16倍），图像分割要求对每个像素进行准确地预测，空间信息的损失会导致分割图不准确。decoder的形式与encoder部分正好相反，包括若干个上采样运算，使用反卷积或插值方法，他们也都是local的方法。

创新点

    为了解决以上问题，作者基于self-attention提出了一个Non-local的结构，global aggregation block用于在上/下采样时可以看到全图的信息，这样会使得到更精准的分割图。简化U-Net，减少参数量，提高推理速度，上采样和下采样使用global aggregation block，使分割更准确。



### reference
[Non-local U-Nets](https://zhuanlan.zhihu.com/p/109514384)