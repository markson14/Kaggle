# Experiment Dairy

### 19.11.04

- [x] try EffcientB7 : 0.652

- [x] Try Gray input image
  - discussion show that this may not improve the LB score

### 19.11.05

- [x] [Pytorch Parallel training](https://github.com/qubvel/segmentation_models.pytorch/issues/34)
- [ ] [Pytorch 多卡训练 负载不均衡](https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21)

### 19.11.06

- [x] try augmentation
  - Hfilp
  - Vfilp
  - RandomBrightnessContrast
- [x] 单卡训练batch测试: 结果证明的确是用多卡训练，单卡batch仅支持2
- [x] 制作kFlod pipeline

### 19.11.07
- [x] 调试KFold训练每个Fold会导致GPU0内存累计的问题：`del all` 
- [ ] 调通KFold Inference Pipeline

### 19.11.08
- [ ] 尝试Image Pyramid
- [x] 尝试ResNet34 single model
- [x] 尝试不同Callback
- [x] TTA
- 尝试不同的decoder
  - [x] UNet
  - [x] FPN

|        Size        |  Time   |
| :----------------: | :-----: |
|      Original      | 24 mins |
|  numpy resize x4   | 11 min  |
| Original_resize x4 | 20 mins |

### 19.11.09
- [x] 尝试 5Fold ResNet34 做对比

| Encoder  | Size | Decoder |   Loss    | KFold | Avg Dice(best/last) | LB(best/last) |
| :------: | :--: | :-----: | :-------: | :---: | :-----------------: | :-----------: |
| Resnet34 |  X5  |  Unet   |  BCEDice  |   1   |   0.64396/0.64701   | 0.6429/0.6420 |
| Resnet34 |  X5  |   FPN   |  BCEDice  |   1   |                     |    0.6439/    |
| Resnet34 |  X5  | LinkNet |  BCEDice  |   1   |   0.63067/0.62729   |               |
| Resnet34 |  x4  |  Unet   |  BCEDice  |   1   |   0.64462/0.64391   | 0.6445/0.6445 |
| Resnet34 |  X4  |   FPN   |  BCEDice  |   1   |   0.64359/0.64543   | 0.6412/0.6412 |
| Resnet34 |  X1  |  Unet   |  BCEDice  |   1   |   0.64982/0.64879   | 0.6367/0.6367 |
| Resnet34 |  X1  |   FPN   |  BCEDice  |   1   |   0.63129/0.63410   | 0.6333/0.6333 |
|  Effib3  |  X5  |  UNet   |  BCEDice  |   1   |   0.75240/0.75429   | 0.6289/0.6289 |
|          |      |         |           |       |                     |               |
| Resnet34 |  X2  |   FPN   |   Focal   |   1   |   0.64007/0.64007   |               |
| Resnet34 |  X2  |   FPN   | FocalDice |   1   |   0.63404/0.63460   |    0.6269/    |

### 19.11.11

- Try 🐸神方法

  - [x] Augmentation

    - Flip

    - random_crop_rescale(925, 630)
    - random_crop_rotate_rescale
    - resize(384, 576)

  - [x] Ensemble: 1xUnet + 3xFPN

### 19.11.12

- 修改post-processing
  - [x] 设置threshold代替自动筛选
  - [x] 将output包装成单独function

- [x] 将数据打包成npy，加速读取：无法做到，太占内存！！！
- [x] 修改Dataset Loader：Pin_memory 和 drop_last 
- [x] 查看ensemble output mask效果
- 弄清楚输出内容: 输出为概率

### 19.11.13

- [x] resize mask约等于multi scale
- [x] Image Pyramid：
  - [x] 320 x 480
  - [x] 384 x 576
  - [x] 640 x 960

|          Encoder          |   Size    | Decoder |  Loss   | KFold | Avg Dice(best/last) | LB(best/last) |
| :-----------------------: | :-------: | :-----: | :-----: | :---: | :-----------------: | :-----------: |
|         Resnet34          |  256x384  |  Unet   | BCEDice |   1   |                     |               |
|         Resnet34          | 320 x 480 |  Unet   | BCEDice |   1   |   0.66749/0.66638   |    0.6447/    |
|         Resnet34          | 384 x 576 |  Unet   | BCEDice |   1   |   0.61193/0.61172   |               |
|         Resnet34          | 640 x 960 |  Unet   | BCEDice |   1   |   0.64922/0.64304   |               |
|                           |           |         |         |       |                     |    0.6340     |
|                           |           |         |         |       |                     |               |
|         Resnet34          | 320 x 480 |   FPN   | BCEDice |   1   |   0.65590/0.65195   |    0.6431/    |
|         Resnet34          | 384 x 576 |   FPN   | BCEDice |   1   |   0.65054/0.64945   |    0.6434/    |
|         Resnet34          | 640 x 960 |   FPN   | BCEDice |   1   |   0.65779/0.65522   |    0.6478/    |
|         Ensemble          |           |   FPN   |         |       |                     |    0.6531/    |
|                           |           |         |         |       |                     |               |
|         Resnet34          |  256x384  |   PSP   | BCEDice |   1   |   0.64492/0.65259   |               |
|         Resnet34          | 384 x 576 |   PSP   | BCEDice |   1   |      0.64810/       |               |
|         Resnet34          | 640 x 960 |   PSP   | BCEDice |   1   |   0.63962/0.63691   |               |
|         Ensemble          |           |         |         |       |                     |    0.6472     |
|                           |           |         |         |       |                     |               |
| Effib2 :white_check_mark: | 640 x 960 |   FPN   | BCEDice |   3   |   0.70375/0.71711   |    0.6582     |
| Effib2 :white_check_mark: | 384 x 576 |   FPN   | BCEDice |   3   |                     |               |
| Effib2 :white_check_mark: | 320 x 480 |   FPN   | BCEDice |   3   |   0.65477/0.65140   |               |
| Effib2 :white_check_mark: |  256x384  |   FPN   | BCEDice |   3   |   0.64424/0.64224   |    0.6497     |
| Effib2 :white_check_mark: |   64x96   |   FPN   | BCEDice |   3   |   0.60951/0.61131   |               |
|         Ensemble          |           |         |         |       |                     |    0.6661     |
|                           |           |         |         |       |                     |               |
| Effib2 :white_check_mark: | 640 x 960 |  UNet   | BCEDice |   2   |   0.67662/0.67614   |    0.6555     |
| Effib2 :white_check_mark: | 384 x 576 |  UNet   | BCEDic  |   3   |   0.64576/0.64633   |               |
| Effib2 :white_check_mark: |  256x384  |  UNet   | BCEDice |   3   |      0.65237/       |               |
|         Ensemble          |           |         |         |       |                     |    0.6638     |

### 19.11.14

- **TTA可以将augmentation概率设置成1**，然后将输出的mask按照TTA还原，最后加权平均。

- [x] TTA pipeline
- [x] 尝试自己的集成代码

### 19.11.15

- [ ] 集成🐸神metric :x:

- [x] 改进ensemble方法
  - Pixel threshold 0.65, maize 10000
  - Using **Temperature sharpening** instead of **average**
  - 加入sigmoid :x:
- [x] 将best，last集成进一个csv

- Pytorch Dataloader无法用index读取

- [x] Train PSPNet

- [ ] 尝试grayscale :x:

### 19.11.18

- Final submission
  - [x] Unet ensemble 0.7 14000 — 0.6638
  - [x] Unet + FPN ensemble 0.7 14000 — 0.6660
  - [x] Unet 640 0.7 14000 — 0. 6555
  - [x] FPN ensemble 0.75 10000 — 0.6624
  - [x] ALL ensemble 0.75 15000 — 0.6680



### Conclusion

- Cross Validation非常有助于涨点
- Image Pyramid适合模型对于多尺度mask，局部整体的理解学习
- 可以尝试训练分类器，将训练集设置为1，测试集设置为0，来测量训练与测试集的分布是否相同。
- 可以尝试融合best epoch和last epoch的结果，而非单纯只用一个epoch的输出
- Test time Augmentation的时候，将augmentation概率设置成1，用作预测，然后将结果在反augmentation回来。最后融合TTA的结果
- 将rawpredict压缩保存 `probability8bit = (probability*255).astype(np.uint8)`：分布方差在0.17747左右