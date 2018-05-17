# Densenet for JingDong AI Fashion Challenge

作者入门深度学习仅有半年左右的时间，观点和代码较为不成熟，还请各位大牛不惜赐教，谢谢。 

在参加过阿里的天池大数据的挑战之后，作者深刻的明白到知识的积累还完全不够，于是抱着继续提高的心理，作者参加了这次京东的fathion挑战赛，选择了风格识别类的赛题。 
题目要求大致如下：
根据穿着场景不同、地区风格不同、年龄不同以及风格不同将服装分为13个大类，服装可以包含其中一个或者多个类别，大致列子如下图所示：

<img src="https://github.com/sephirothhua/JingDong_fashion_challenge/blob/master/images/sample.jpg" width="300" height="300" alt="sample" align=center />

本着学习以及加快训练速度的原则，作者挑选使用了Densenet作为本次挑战的主网络，使用了keras官方的Densenet作为参考，并进一步封装了自己的model函数。 

针对本次训练任务，作者在Densenet最后一层卷积基础上增加了13个分类学习任务，分别对每个类别进行了卷积和全链接操作。一开始作者将输入最后合并为一个向量进行输出并计算loss，但由于本次比赛样本数据的不均，作者采用了13个输出并分开做loss，并施加不同权重的方法进行训练。

## 1.环境要求：
    Python 3.4/3.5
    numpy
    scikit-image
    tensorflow>=1.3.0
    keras>=2.0.8
    opencv-python
    h5py

## 2文件结构
```
project 
  |--README.md  
  |--data  
     |--train+val.txt
     |--test.tfrecords
     |--train.tfrecords  
  |--model
  |--model_save
     |--densenet_notop_model.h5
  |--utils  
  |--log
  |--test.py
  |--detect_trail.py
  |--Generate_dataset.py
  ```
 ### 2.1文件目录说明
 * data主要用于存放训练数据，起始只有一个txt文件，需要使用Generate_dataset.py来生成数据集。
 * model存放主要模型文件。
 * model_save主要用于保存下载的权重文件。
 * utils主要用于进行数据的生成和读取。
 * log存放训练日志和训练过程中的模型
## 3.主目录代码说明
   * test.py 主要训练代码，用于进行训练。
   * detect_trail.py 主要测试代码，用于检测出结果。
   * Generate_dataset.py 数据集生成代码，可读取txt中的url地址并将图片预处理后保存为tfrecord文件。
## 4.如何训练
   * 首先，需要加载训练集，作者提供了Generate_dataset.py文件用于读取data文件夹下的txt文件，并将url图片和标签封装为tfrecord方便训练调用。
   * 接着，只需要运行test.py就能够进行训练了。 
   
    如果想要修改训练参数，请修改test.py下的My_config类别：
    class My_config(config):
        Name = "ImageNet"
        Num_categrade = 13
        Batch_size = 10
        Use_learning_rate_reduce = False
    如果想要训练不同的层，以及调整epoch请改动26行代码，目前只支持训练"heads"和"all"：
    model.train(train_dataset=train_dataset,
            val_dataset=val_dataset,
            learning_rate=real_config.Learning_rate,
            epochs=200,
            layers='heads')
   *注意：如果您没有预先下载Densnet的预训练模型，那么请在创建模型时填入model_dir参数，会自动帮您下载到存在的目录！*  
## 5.如何测试
   * 将训练好的模型放在已知目录下，修改detect_trail.py 中的 model_path 和 model_filename 两个参数至您的目录。
   * 运行 detect_trail.py即可看到预测效果。
   
   *注意：输入大小shape必须为(224，224，3)，日后会进行改进，谢谢谅解!*
## 6.存在的问题
前面提到过，本次比赛的数据集正负样本极度失衡，作者对提供的大约54000张图片进行了一次统计，发现所得到13个类别分布如下所示：

*[104, 4292, 49893, 262, 18722, 8882, 197, 1356, 49389, 7094, 3712, 1249, 361]*

可以发现，5w张图中间大部分样本均是不平均的，其中第1，4，7，12正样本极少，而其中第3，9类负样本又极少。

同样作者也通过编码的方式对样本进行了一次统计，统计结果亦出现了大量单个样本，极度不平均，这里就不贴出结果了。

作者希望能有更多的人能够参与到这份工程的研究中来，也希望能够得到各位大牛的指导，也欢迎大家在下面进行评论和提问，谢谢大家。
