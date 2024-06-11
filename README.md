# G2G
This is the Official PyTorch implemention of our ICASSP2024 paper "G2G: Generalized Learning by Cross-Domain Knowledge Transfer for Federated Domain Generalization".

# G2G:Generalized Learning by Cross-Domain Knowledge Transfer for Federated Domain Generalization
Code to reproduce the experiments of **G2G:Generalized Learning by Cross-Domain Knowledge Transfer for Federated Domain Generalization**.
## How to use it
* Clone or download the repository
### Install the requirement
 >  pip install -r requirements.txt
### Download VLCS, PACS and Office-Home datasets
* Download VLCS from https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md, extract, move it to datasets/VLCS/VLCS/

([MEGA](https://mega.nz/#F!gTJxGTJK!w9UJjZVq3ClqGj4mBDmT4A)|[Baiduyun](https://pan.baidu.com/s/1nuNiJ0l))使用百度云下载下来5个mat文件，读取不了。只能又去百度搜索，找到http://www.mediafire.com/file/7yv132lgn1v267r/vlcs.tar.gz/file，在此下载

* Download PACS from https://drive.google.com/uc?id=0B6x7gtvErXgfbF9CSk53UkRxVzg, move it to datasets/PACS/

(PACS在这下载不了，没权限。去谷歌一下找到一个仓库，这里发布了新的谷歌云盘，下载成功：https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ，从这个仓库：https://github.com/MachineLearning2020/Homework3-PACS/tree/master/PACS也可以下载，但是只是纯图像的压缩包，还得自己手动划分训练集测试集，参考https://blog.csdn.net/qq_43827595/article/details/121345640。两种数据加载方式不同，一个是hdf5一个是raw images)

* Download Office-Home from https://datasets.activeloop.ai/docs/ml/datasets/office-home-dataset/, move it to datasets/OfficeHome/
### Download Pre-trained models
* Download pre-trained AlexNet from https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth and move it to models/
* Download pre-trained Resnet50 from https://download.pytorch.org/models/resnet50-19c8e357.pth and move it to models/
* Download pre-trained VGG16 https://download.pytorch.org/models/vgg16-397923af.pth and move it to models/


下面的三个数据集都是只有4个domain，所以设置node_num=3，每个客户端一个source domain，另一个作为target domain
### Running ours on VLCS
``` 
pymao main_warm.py --node_num 3  --device cuda:0 --dataset vlcs --classes 5 --lr 0.0008 --global_model Alexnet --local_model Alexnet --algorithm fed_mutual --R 50 --E 7 --pretrained True --batch_size 64 --iteration 0 
```
### Running ours on PACS
``` 
pymao main_warm.py --node_num 3  --device cuda:0 --dataset pacs --classes 7 --lr 0.0008 --global_model Alexnet --local_model Alexnet --algorithm fed_mutual --R 50 --E 7 --pretrained True --batch_size 32 --iteration 0 
```
### Running ours on Office-Home
``` 
pymao main_warm.py --node_num 3 --device cuda:0 --dataset office-home --classes 65 --lr 0.0008 --global_model ResNet50 --local_model ResNet50 --algorithm fed_mutual --R 50 --E 7 --batch_size 32 --iteration 0 
```

### My

#### VLCS
```
pymao main_warm.py --node_num 3  --device cuda:0 --dataset vlcs --classes 5 --lr 0.0008 --global_model Alexnet --local_model Alexnet --algorithm fed_adv --R 50 --E 7 --pretrained True --batch_size 64 --iteration 0  
```

pymao main_warm.py --node_num 3  --device cuda:1 --dataset vlcs --classes 5 --lr 0.0008 --global_model Alexnet --local_model Alexnet --algorithm fed_adv --R 60 --E 1 --pretrained True --batch_size 512 --iteration 3 --method ssl --simclr_e 10 --gen_e 2 --discr_e 2 --disc_lr 0.001 --gen_lr 0.001 --cl_lr 0.0001 --cls_epochs 10 --cls_lr 0.001 --ce_epochs 100


#### RotatedMNIST

```
pymao main_warm.py --node_num 5  --device cuda:0 --dataset rotatedmnist --classes 10 --lr 0.0008 --global_model Alexnet --local_model Alexnet --algorithm fed_adv --cls_epochs 1  --R 50 --E 3 --pretrained True --batch_size 64 --iteration 0 --optimizer adam
```


```
pymao main_warm.py --node_num 5  --device cuda:0 --dataset rotatedmnist --classes 10 --lr 0.0008 --global_model Alexnet --local_model Alexnet --algorithm fed_mutual --R 60 --E 1 --pretrained True --batch_size 512 --iteration 4 --cls_epochs 10 --discr_e 2 --gen_e 2 --cl_lr 0.001 --cls_lr 0.001 --disc_lr 0.001 --gen_lr 0.001 --simclr_e 10 --method ssl --cls_epochs 10 --cls_lr 0.001 --optimizer adam
```

