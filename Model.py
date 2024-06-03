from collections import OrderedDict
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from torch import optim
import copy
import random
import torch


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, 3)
        # self.conv1 = nn.Conv2d(3, 96, 3)
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# todo Bottleneck
class Bottleneck(nn.Module):
    
    expansion = 4   
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)   
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, bias=False, padding=1)  
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)   
        self.bn3 = nn.BatchNorm2d(num_features=out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x    
        if self.downsample is not None:
            identity = self.downsample(x)   

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity     
        out = self.relu(out)

        return out


class ResNet(nn.Module):
   
    def __init__(self, block, block_num, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channel = 64   

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)     
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     
        self.layer1 = self._make_layer(block=block, channel=64, block_num=block_num[0], stride=1)   
        self.layer2 = self._make_layer(block=block, channel=128, block_num=block_num[1], stride=2)  
        self.layer3 = self._make_layer(block=block, channel=256, block_num=block_num[2], stride=2)  
        self.layer4 = self._make_layer(block=block, channel=512, block_num=block_num[3], stride=2)  
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))  
        self.fc1 = nn.Linear(in_features=512*block.expansion, out_features=num_classes)
        for m in self.modules():   
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None  
        if stride != 1 or self.in_channel != channel*block.expansion:  
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel*block.expansion, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(num_features=channel*block.expansion))
        layers = []  
        layers.append(block(in_channel=self.in_channel, out_channel=channel, downsample=downsample, stride=stride)) 
        self.in_channel = channel*block.expansion  
        for _ in range(1, block_num):  
            layers.append(block(in_channel=self.in_channel, out_channel=channel))
        return nn.Sequential(*layers) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x
        

class VGGnet(nn.Module):
    def __init__(self,feature_extract=True,num_classes=5):
        super(VGGnet, self).__init__()
        model = models.vgg16(pretrained=False)
        pretrained_state_dict = torch.load("vgg16-397923af.pth")

        model.load_state_dict(pretrained_state_dict, strict=False)
        print('use VGG16 pretrained!')
        self.features = model.features
        set_parameter_requires_grad(self.features, feature_extract)#固定特征提取层参数
        self.avgpool=model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7 , 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 512*7*7)
        out=self.classifier(x)
        return out
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False





class AlexNet(nn.Module):
    def __init__(self,args):
        super(AlexNet, self).__init__()
        alexnet_fetExtrac = feature_extractor(optim.SGD, args.lr0, args.momentum, args.weight_dec)
        state_dict = torch.load("models/alexnet_caffe.pth.tar")

        # for key, value in state_dict.items():
        #     print(key, value.size())
        # 这里修改
        del state_dict["classifier.6.weight"]
        del state_dict["classifier.6.bias"]
        alexnet_fetExtrac.load_state_dict(state_dict)
        alexnet__classifier = task_classifier(args.hidden_size, optim.SGD, args.lr0, args.momentum, args.weight_dec,
                                                class_num=args.classes)
        self.net = nn.Sequential(alexnet_fetExtrac,alexnet__classifier)
        
    def forward(self, x):
        return self.net(x)

# def Alexnet(args):
#     alexnet= models.alexnet(pretrained=True)
#     num_fc = alexnet.classifier[6].in_features
#     alexnet.classifier[6] = torch.nn.Linear(in_features=num_fc, out_features=args.classes)
#     return alexnet


class feature_extractor(nn.Module):
    def __init__(self, optimizer,lr,momentum,weight_decay, num_classes=5):
        super(feature_extractor,self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(OrderedDict([
            # 修改
            ("0",nn.Conv2d(3,64,kernel_size=11,stride=4)),
            ("relu1",nn.ReLU(inplace=True)),
            ("pool1",nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)),
            ("norm1",nn.LocalResponseNorm(5,1.e-4,0.75)),

            ("3",nn.Conv2d(64,192,kernel_size=5,padding=2)),
            ("relu2",nn.ReLU(inplace=True)),
            ("pool2",nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)),
            ("norm2",nn.LocalResponseNorm(5,1.e-4,0.75)),

            ("6",nn.Conv2d(192,384,kernel_size=3,padding=1)),
            ("relu3",nn.ReLU(inplace=True)),

            ("8",nn.Conv2d(384,256,kernel_size=3,padding=1)),
            ("relu4",nn.ReLU(inplace=True)),

            ("10",nn.Conv2d(256,256,kernel_size=3,padding=1)),
            ("relu5",nn.ReLU(inplace=True)),
            ("pool5",nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True))
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ("1", nn.Linear(256 * 6 * 6, 4096)),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout()),

            ("4", nn.Linear(4096, 4096)),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout())
        ]))

        self.optimizer = optimizer(list(self.features.parameters())+list(self.classifier.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.initial_params()

    def initial_params(self):
        for layer in self.modules():
            if isinstance(layer,torch.nn.Linear):
                init.xavier_uniform_(layer.weight,0.1)
                layer.bias.data.zero_()

    def forward(self, x):
        x = self.features(x*57.6)
        x = x.view((x.size(0),256*6*6))
        x = self.classifier(x)
        return x
# classifier
class task_classifier(nn.Module):
    def __init__(self, hidden_size, optimizer, lr, momentum, weight_decay, class_num=5):
        super(task_classifier,self).__init__()
        self.task_classifier = nn.Sequential()
        self.task_classifier.add_module('t1_fc1', nn.Linear(hidden_size, hidden_size))
        self.task_classifier.add_module('t1_fc2', nn.Linear(hidden_size, class_num))
        self.optimizer = optimizer(self.task_classifier.parameters(),
                                   lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.initialize_paras()

    def initialize_paras(self):
        for layer in self.modules():
            if isinstance(layer,torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight,a=0,mode='fan-out')
            elif isinstance(layer,torch.nn.Linear):
                init.kaiming_normal_(layer.weight)
            elif isinstance(layer,torch.nn.BatchNorm2d) or isinstance(layer,torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, x):
        x = torch.flatten(x, 1)
        y = self.task_classifier(x)
        return y


def ResNet50(args):
    model = ResNet(block=Bottleneck, block_num=[3, 4, 6, 3], num_classes=args.classes)
    if args.pretrained:
        pretrained_state_dict = torch.load("models/resnet50-19c8e357.pth")
        model.load_state_dict(pretrained_state_dict, strict=False)
        print('use resnet50 pretrained!')
    return model

def ResNet18(args):
    model= models.resnet18(pretrained=True)
    num_features=model.fc.in_features
    model.fc=nn.Linear(num_features,args.classes)
    return model


## as baseline
def Alexnet(args):
    return AlexNet(args)

def VGG16(args):
    model = VGGnet(feature_extract=True,num_classes=args.classes)
    return model

class Generator(nn.Module):
    def __init__(self, latent_space, num_classes, flat_img):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(latent_space+num_classes, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, flat_img),
            nn.Tanh()
            # nn.Linear(latent_space+num_classes, 128),
            # nn.LeakyReLU(0.2),
            # nn.Linear(128, 256),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(0.2),
            # nn.Linear(256, 512),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(0.2),
            # nn.Linear(512, 1024),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(0.2),
            # nn.Linear(1024, flat_img),
            # nn.Tanh()
        )

    def forward(self, x, y):
        out = torch.cat((x, y), dim=1)
        out = self.gen(out)
        return out

class Generator1(nn.Module):
    def __init__(self, num_classes=10, flat_img=784):
        super(Generator1, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(flat_img+num_classes, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, flat_img),
            nn.Tanh()
        )

    def forward(self, x, y):
        out = torch.cat((x, y), dim=1)
        out = self.gen(out)
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)




class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix



# 定义对比学习模型
class SimCLR(nn.Module):
    def __init__(self, args, in_channel):
        super(SimCLR, self).__init__()
        if args.dataset == 'rotatedmnist':
            # self.encoder = nn.Sequential(
            #     nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(64),
            #     nn.ReLU(),
            #     nn.MaxPool2d(kernel_size=2, stride=2),
            #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(128),
            #     nn.ReLU(),
            #     nn.MaxPool2d(kernel_size=2, stride=2),
            #     # MixStyle(p=0.5, alpha=0.1),
            #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            #     nn.BatchNorm2d(256),
            #     nn.ReLU(),
            #     nn.MaxPool2d(kernel_size=2, stride=2),
            #     Flatten(),
            #     # nn.Linear(1024, 1024),
            #     # nn.ReLU(),
            #     # nn.Linear(1024, 120),
            #     # nn.ReLU()
            # )
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=5, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2), # 14x14x32
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2), # 7x7x64
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2), # 3X3x64
                Flatten(),
                nn.Linear(576, 1024)
            )
            self.projection_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(1024, args.embedding_d),
            )
            self.prediction = nn.Sequential(
                nn.ReLU(),
                nn.Linear(args.embedding_d, args.classes)
            )
        else:
            self.encoder = feature_extractor(optim.SGD, args.lr0, args.momentum, args.weight_dec)
            state_dict = torch.load("models/alexnet_caffe.pth.tar")
            del state_dict["classifier.6.weight"]
            del state_dict["classifier.6.bias"]
            self.encoder.load_state_dict(state_dict)

            self.projection_head = nn.Sequential(OrderedDict([
                ("1", nn.Linear(4096, 4096)),
                ("relu6", nn.ReLU(inplace=True)),
                ("drop6", nn.Dropout()),

                ("4", nn.Linear(4096, args.embedding_d)),
                ("relu7", nn.ReLU(inplace=True)),
                ("drop7", nn.Dropout())
            ]))
        # self.initial_params()

    def initial_params(self):
        for layer in self.modules():
            if isinstance(layer,torch.nn.Linear):
                init.xavier_uniform_(layer.weight,0.1)
                layer.bias.data.zero_()

    def forward(self, x):
        feature = self.encoder(x)
        # feature = x.view((x.size(0), -1))
        embeddings = self.projection_head(feature)
        out = self.prediction(embeddings)
        return feature, embeddings, out


class Discriminator(nn.Module):
    def __init__(self, flat_img, num_classes):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(flat_img+num_classes, 128),  # 输入特征数为784，输出为512
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(128, 64),  # 进行一个线性映射
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.dis(x)
        return x

class Discriminator2(nn.Module):
    def __init__(self, flat_img):
        super(Discriminator2, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(flat_img, 512),  # 输入特征数为784，输出为512
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(512, 256),  # 进行一个线性映射
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.dis(x)
        return x

class Classifier(torch.nn.Module):
    def __init__(self, args, simclr_model, num_class=5):
        super(Classifier, self).__init__()
        # encoder
        self.encoder = simclr_model.encoder
        # classifier
        if args.dataset == 'rotatedmnist':
            self.fc = nn.Linear(1024, num_class, bias=True)
        else:
            self.fc = task_classifier(args.hidden_size, optim.SGD, args.lr0, args.momentum, args.weight_dec,
                                                class_num=args.classes)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return feature, out