## 图像分割中从Loss上解决数据集imbalance的方法

前言

最近在做小目标图像分割任务（医疗方向），往往一幅图像中只有一个或者两个目标，而且目标的像素比例比较小，使网络训练较为困难，一般可能有三种的解决方式：

    选择合适的loss function，对网络进行合理的优化，关注较小的目标。
    改变网络结构，使用attention机制（类别判断作为辅助）。
    与2的根本原理一致，类属attention，即：先检测目标区域，裁剪之后进行分割训练。

通过使用设计合理的loss function，相比于另两种方式更加简单易行，能够保留图像所有信息的情况下进行网络优化，达到对小目标精确分割的目的。
场景

    使用U-Net作为基准网络。
    实现使用keras
    小目标图像分割场景，如下图举例。

loss function
一、Log loss

对于二分类而言，对数损失函数如下公式所示： −1N∑Ni=1(yilogpi+(1−yi)log(1−pi))
−N1​i=1∑N​(yi​logpi​+(1−yi​)log(1−pi​))
其中，yiyi​为输入实例xixi​的真实类别, pipi​为预测输入实例 xi

xi​ 属于类别 1 的概率. 对所有样本的对数损失表示对每个样本的对数损失的平均值, 对于完美的分类器, 对数损失为 0。
此loss function每一次梯度的回传对每一个类别具有相同的关注度！所以极易受到类别不平衡的影响，在图像分割领域尤其如此。
例如目标在整幅图像当中占比也就仅仅千分之一，那么在一副图像中，正样本（像素点）与父样本的比例约为1～1000，如果训练图像中还包含大量的背景图，即图像当中不包含任何的疾病像素，那么不平衡的比例将扩大到>10000，那么训练的后果将会是，网络倾向于什么也不预测！生成的mask几乎没有病灶像素区域！
此处的案例可以参考airbus-ship-detection。
二、WBE Loss

带权重的交叉熵loss — Weighted cross-entropy (WCE)[6]
R为标准的分割图，其中rn
rn​为label 分割图中的某一个像素的GT。P为预测的概率图，pnpn​为像素的预测概率值，背景像素图的概率值就为1-P。
只有两个类别的带权重的交叉熵为：
WCE=−1N∑Nn=1wrnlog(pn)+(1−rn)log(1−pn)WCE=−N1​n=1∑N​wrn​log(pn​)+(1−rn​)log(1−pn​)
ww为权重，w=N−∑npn∑npn

w=∑n​pn​N−∑n​pn​​

缺点是需要人为的调整困难样本的权重，增加调参难度。
三、Focal loss

能否使网络主动学习困难样本呢？
focal loss的提出是在目标检测领域，为了解决正负样本比例严重失衡的问题。是由log loss改进而来的，为了于log loss进行对比，公式如下：
−1N∑Ni=1(αyi(1−pi)γlogpi+(1−α)(1−yi)pγilog(1−pi))
−N1​i=1∑N​(αyi​(1−pi​)γlogpi​+(1−α)(1−yi​)piγ​log(1−pi​))
说白了就多了一个(1−pi)γ(1−pi​)γ，loss随样本概率的大小如下图所示：

其基本思想就是，对于类别极度不均衡的情况下，网络如果在log loss下会倾向于只预测负样本，并且负样本的预测概率pipi​也会非常的高，回传的梯度也很大。但是如果添加(1−pi)γ(1−pi​)γ则会使预测概率大的样本得到的loss变小，而预测概率小的样本，loss变得大，从而加强对正样本的关注度。
可以改善目标不均衡的现象，对此情况比 binary_crossentropy 要好很多。
目前在图像分割上只是适应于二分类。
代码：https://github.com/mkocabas/focal-loss-keras

```python
from keras import backend as K
'''
Compatible with tensorflow backend
'''
def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

```
使用方法：

model_prn.compile(optimizer=optimizer, loss=[focal_loss(alpha=.25, gamma=2)])

目前实验得到结论：

    经过测试，发现使用focal loss很容易就会过拟合？？且效果一般。。。I don’t know why?
    此方法代码有待改进，因为此处使用的网络为U-net，输入和输出都是一张图！直接使用会导致loss的值非常的大！
    需要添加额外的两个全局参数alpha和gamma，对于调参不方便。

    以上的方法Log loss，WBE Loss，Focal loss都是从本源上即从像素上来对网络进行优化。针对的都是像素的分类正确与否。有时并不能在评测指标上DICE上得到较好的结果。

---- 更新2020-4-1
将K.sum改为K.mean, 与其他的keras中自定义的损失函数保持一致:

```python
-K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
```
四、Dice loss

dice loss 的提出是在Ｖ-net中，其中的一段原因描述是在感兴趣的解剖结构仅占据扫描的非常小的区域，从而使学习过程陷入损失函数的局部最小值。所以要加大前景区域的权重。

Dice 可以理解为是两个轮廓区域的相似程度，用A、B表示两个轮廓区域所包含的点集，定义为：
DSC(A,B)=2∣A⋂B∣∣A∣+∣B∣
DSC(A,B)=2∣A∣+∣B∣∣A⋂B∣​
其次Dice也可以表示为：
DSC=2TP2TP+FN+FPDSC=2TP+FN+FP2TP​
其中TP，FP，FN分别是真阳性、假阳性、假阴性的个数。
二分类dice loss:
DL2=1−∑Nn=1pnrn+ϵ∑Nn=1pn+rn+ϵ−∑Nn=1(1−pn)(1−rn)+ϵ∑Nn=12−pn−rn+ϵDL2​=1−∑n=1N​pn​+rn​+ϵ∑n=1N​pn​rn​+ϵ​−∑n=1N​2−pn​−rn​+ϵ∑n=1N​(1−pn​)(1−rn​)+ϵ​

```python
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_coef_loss(y_true, y_pred):
	1 - dice_coef(y_true, y_pred, smooth=1)
```

结论：

    有时使用dice loss会使训练曲线有时不可信，而且dice loss好的模型并不一定在其他的评价标准上效果更好，例如mean surface distance 或者是Hausdorff surface distance。
    不可信的原因是梯度，对于softmax或者是log loss其梯度简化而言为p−t

p−t，tt为目标值，pp为预测值。而dice loss为2t2(p+t)2(p+t)22t2​，如果pp，t
t过小则会导致梯度变化剧烈，导致训练困难。
属于直接在评价标准上进行优化。
不均衡的场景下的确好使。

五、IOU loss

可类比DICE LOSS，也是直接针对评价标准进行优化[11]。
在图像分割领域评价标准IOU实际上IOU=TPTP+FP+FN
IOU=TP+FP+FNTP​，而TP，FP，FN分别是真阳性、假阳性、假阴性的个数。
而作为loss function，定义IOU=I(X)U(X)IOU=U(X)I(X)​，其中，I(X)=X∗YI(X)=X∗Y
U(X)=X+Y−X∗YU(X)=X+Y−X∗Y，X为预测值而Y为真实标签。


```python
## intersection over union
def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)
```

IOU loss的缺点呢同DICE loss是相类似的，训练曲线可能并不可信，训练的过程也可能并不稳定，有时不如使用softmax loss等的曲线有直观性，通常而言softmax loss得到的loss下降曲线较为平滑。
六、Tversky loss

提到Tversky loss不得不提Tversky 系数，Tversky系数是Dice系数和 Jaccard 系数的一种广义系数，公式如下：
T(A,B)=∣A⋂B∣∣A⋂B∣+α∣A−B∣+β∣B−A∣
T(A,B)=∣A⋂B∣+α∣A−B∣+β∣B−A∣∣A⋂B∣​
再抄一遍Dice系数公式：
DSC(A,B)=2∣A⋂B∣∣A∣+∣B∣DSC(A,B)=2∣A∣+∣B∣∣A⋂B∣​，此时A为预测,而B为真实标签。
观察可得当设置α=β=0.5α=β=0.5，此时Tversky系数就是Dice系数。而当设置α=β=1α=β=1时，此时Tversky系数就是Jaccard系数。
对于Tversky loss也是相似的形式就不重新编写了，但是在T(A,B)T(A,B)中，∣A−B∣∣A−B∣则意味着是FP（假阳性），而∣B−A∣∣B−A∣则意味着是FN（假阴性）；α和βα和β分别控制假阴性和假阳性。通过调整αα和ββ我们可以控制假阳性和假阴性之间的权衡。

七、敏感性–特异性 loss

首先敏感性就是召回率，检测出确实有病的能力：
Sensitivity=TPTP+FN
Sensitivity=TP+FNTP​
特异性，检测出确实没病的能力：
Specificity=TNTN+FPSpecificity=TN+FPTN​
Sensitivity - Speciﬁcity (SS)[8]提出是在:
SS=λ∑Nn=1(rn−pn)2rn∑Nn=1rn+ϵ　+(1−λ)∑Nn=1(rn−pn)2(1−rn)∑Nn=1(1−rn)+ϵ

SS=λ∑n=1N​rn​+ϵ∑n=1N​(rn​−pn​)2rn​​　+(1−λ)∑n=1N​(1−rn​)+ϵ∑n=1N​(rn​−pn​)2(1−rn​)​

其中左边为病灶像素的错误率即，1−Sensitivity
1−Sensitivity，而不是正确率，所以设置λ 为0.05。其中(rn−pn)2

(rn​−pn​)2是为了得到平滑的梯度。
八、Generalized Dice loss

区域大小和Dice分数之间的相关性：
在使用DICE loss时，对小目标是十分不利的，因为在只有前景和背景的情况下，小目标一旦有部分像素预测错误，那么就会导致Dice大幅度的变动，从而导致梯度变化剧烈，训练不稳定。
首先Generalized Dice loss的提出是源于Generalized Dice index[12]。当病灶分割有多个区域时，一般针对每一类都会有一个DICE，而Generalized Dice index将多个类别的dice进行整合，使用一个指标对分割结果进行量化。

GDL(the generalized Dice loss)公式如下(标签数量为2)：
GDL=1−2∑2l=1wl∑nrlnpln∑2l=1wl∑nrln+pln
GDL=1−2∑l=12​wl​∑n​rln​+pln​∑l=12​wl​∑n​rln​pln​​
其中rlnrln​为类别l在第n个像素的标准值(GT)，而plnpln​为相应的预测概率值。此处最关键的是wlwl​，为每个类别的权重。其中wl=1(∑Nn=1rln)2wl​=(∑n=1N​rln​)21​，这样，GDL就能平衡病灶区域和Dice系数之间的平衡。
论文中的一个效果：七、敏感性–特异性 loss

首先敏感性就是召回率，检测出确实有病的能力：
Sensitivity=TPTP+FN
Sensitivity=TP+FNTP​
特异性，检测出确实没病的能力：
Specificity=TNTN+FPSpecificity=TN+FPTN​
Sensitivity - Speciﬁcity (SS)[8]提出是在:
SS=λ∑Nn=1(rn−pn)2rn∑Nn=1rn+ϵ　+(1−λ)∑Nn=1(rn−pn)2(1−rn)∑Nn=1(1−rn)+ϵ

SS=λ∑n=1N​rn​+ϵ∑n=1N​(rn​−pn​)2rn​​　+(1−λ)∑n=1N​(1−rn​)+ϵ∑n=1N​(rn​−pn​)2(1−rn​)​

其中左边为病灶像素的错误率即，1−Sensitivity
1−Sensitivity，而不是正确率，所以设置λ 为0.05。其中(rn−pn)2

(rn​−pn​)2是为了得到平滑的梯度。
八、Generalized Dice loss

区域大小和Dice分数之间的相关性：
在使用DICE loss时，对小目标是十分不利的，因为在只有前景和背景的情况下，小目标一旦有部分像素预测错误，那么就会导致Dice大幅度的变动，从而导致梯度变化剧烈，训练不稳定。
首先Generalized Dice loss的提出是源于Generalized Dice index[12]。当病灶分割有多个区域时，一般针对每一类都会有一个DICE，而Generalized Dice index将多个类别的dice进行整合，使用一个指标对分割结果进行量化。

GDL(the generalized Dice loss)公式如下(标签数量为2)：
GDL=1−2∑2l=1wl∑nrlnpln∑2l=1wl∑nrln+pln
GDL=1−2∑l=12​wl​∑n​rln​+pln​∑l=12​wl​∑n​rln​pln​​
其中rlnrln​为类别l在第n个像素的标准值(GT)，而plnpln​为相应的预测概率值。此处最关键的是wlwl​，为每个类别的权重。其中wl=1(∑Nn=1rln)2wl​=(∑n=1N​rln​)21​，这样，GDL就能平衡病灶区域和Dice系数之间的平衡。
论文中的一个效果：

但是在AnatomyNet中提到GDL面对极度不均衡的情况下，训练的稳定性仍然不能保证。
参考代码：
```python
def generalized_dice_coeff(y_true, y_pred):
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2))
    w = 1/(w**2+0.000001)
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2,3))
    numerator = K.sum(numerator)
    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2,3))
    denominator = K.sum(denominator)
    gen_dice_coef = 2*numerator/denominator
    return gen_dice_coef
def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)

```

以上本质上都是根据评测标准设计的loss function，有时候普遍会受到目标太小的影响，导致训练的不稳定；对比可知，直接使用log loss等的loss曲线一般都是相比较光滑的。

九、BCE + Dice loss

BCE : Binary Cross Entropy
说白了，添加二分类交叉熵损失函数。在数据较为平衡的情况下有改善作用，但是在数据极度不均衡的情况下，交叉熵损失会在几个训练之后远小于Dice 损失，效果会损失。
代码：
```python
import keras.backend as K
from keras.losses import binary_crossentropy
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3]) ##y_true与y_pred都是矩阵！（Unet）
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

```

十、Dice + Focal loss

最近腾讯医疗AI新突破：提出器官神经网络，全自动辅助头颈放疗规划 | 论文[2] 中提出了Dice + Focal loss来处理小器官的分割问题。在前面的讨论也提到过，直接使用Dice会使训练的稳定性降低[1]，而此处再添加上Focal loss这个神器。
首先根据论文的公式：
TPp(c)=∑Nn=1pn(c)gn(c)
TPp​(c)=n=1∑N​pn​(c)gn​(c)
FNp(c)=∑Nn=1(1−pn(c))gn(c)FNp​(c)=n=1∑N​(1−pn​(c))gn​(c)
FPn(c)=∑Nn=1pn(c)(1−gn(c))FPn​(c)=n=1∑N​pn​(c)(1−gn​(c))
LDice=∑Cc=0TPn(c)TPp(c)+αFNp(c)+βFPp(c)−LDice​=c=0∑C​TPp​(c)+αFNp​(c)+βFPp​(c)TPn​(c)​−
其中TPp(c)，FNp(c)，FPp(c)TPp​(c)，FNp​(c)，FPp​(c)，分别对于类别c的真阳性，假阴性，假阳性。此处的α=β=0.5α=β=0.5，此时Tversky系数就是Dice系数，为Dice loss。
最终的loss为：
L=LDice+λLFocal=C−∑Cc=0TPn(c)TPp(c)+αFNp(c)+βFPp(c)−λ1N∑Cc=0∑Nn=1gn(c)(1−pn(c))2log(pn(c))L=LDice​+λLFocal​=C−c=0∑C​TPp​(c)+αFNp​(c)+βFPp​(c)TPn​(c)​−λN1​c=0∑C​n=1∑N​gn​(c)(1−pn​(c))2log(pn​(c))
