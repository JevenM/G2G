import copy
import torch
import Model
from torch import optim
# from Trainer import SampleGenerator
from simsiam import SimSiam
from utils import GradualWarmupScheduler, KL_Loss, compute_distances
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

kl_loss = KL_Loss(temperature=3)

def init_model(model_type, args):
    model = []
    if model_type == 'LeNet5':
        model = Model.LeNet5()
    elif model_type == 'MLP':
        model = Model.MLP()
    elif model_type == 'ResNet50':
        model = Model.ResNet50(args)
    elif model_type == 'ResNet18':
        model = Model.ResNet18(args)
    elif model_type == 'VGG16':
        model = Model.VGG16(args)
    elif model_type == 'Alexnet':
        model = Model.Alexnet(args)
    elif model_type == 'CNN':
        model = Model.CNN()
    return model


def init_optimizer(model, args):
    optimizer = []
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    return optimizer


def weights_zero(model):
    for p in model.parameters():
        if p.data is not None:
            p.data.detach_()
            p.data.zero_()


class Node(object):
    def __init__(self, num, train_data, test_data, args, target_load=None):
        self.args = args
        self.num = num + 1
        # 初始化原型矩阵
        # if args.dataset == 'rotatedmnist':
        #     dim = 1024
        # else:
        #     dim = 4096
        self.prototypes = None
        self.prototypes_global = None
        # 统计每个类别的样本数量
        # self.class_counts = torch.zeros(self.args.classes)
        self.device = self.args.device
        self.train_data = train_data
        self.test_data = test_data
        self.target_loader = target_load
        self.model = init_model(self.args.local_model,args).to(self.device)
        self.optimizer = init_optimizer(self.model, self.args)
        if args.dataset == 'rotatedmnist':
            flatten_dim = 28*28
            in_channel = 1
            self.model = Model.SimCLR(args, in_channel).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.cl_lr, weight_decay=5e-4)
        else:
            flatten_dim = 225*225*3
            in_channel = 3
        # self.gen_model = Model.Generator1(args.classes, flatten_dim).to(self.device)
        # self.gen_model = Model.Generator(args.latent_space, args.classes, flatten_dim).to(self.device)
        self.gen_model = Model.GeneratorFeature(args.latent_space, args.classes, 2*args.embedding_d).to(self.device)
        self.optm_gen = optim.Adam(self.gen_model.parameters(), lr=args.gen_lr, weight_decay=5e-4)
        
        if args.method == 'simsiam':
            self.cl_model = SimSiam(in_channel).to(self.device)
        else:
            self.cl_model = Model.SimCLR(args, in_channel).to(self.device)
        self.optm_cl = optim.Adam(self.cl_model.parameters(), lr=args.cl_lr, weight_decay=5e-4)
        self.optm_fc = optim.SGD(self.cl_model.prediction.parameters(), lr=args.cls_lr, weight_decay=5e-4)
        self.ssl_scheduler = optim.lr_scheduler.StepLR(self.optm_cl, step_size=30, gamma=0.99)
        # 判断是否是真样本
        # self.disc_model = Model.Discriminator(flatten_dim, args.classes).to(self.device)
        self.disc_model = Model.DiscriminatorFeature(2*args.embedding_d, args.classes).to(self.device)
        self.optm_disc = optim.Adam(self.disc_model.parameters(), lr=args.disc_lr, weight_decay=5e-4)
        # 判断是否是目标域
        # self.disc_model2 = Model.Discriminator2(flatten_dim).to(self.device)
        self.disc_model2 = Model.DiscriminatorFeature2(2*args.embedding_d).to(self.device)
        self.optm_disc2 = optim.Adam(self.disc_model2.parameters(), lr=args.disc_lr, weight_decay=5e-4)

        # self.clser = Model.Classifier(args, self.cl_model, args.classes).to(self.device)
        # self.optm_cls = optim.Adam(self.clser.fc.parameters(), lr=args.cls_lr, weight_decay=5e-4)
        self.meme = init_model(self.args.global_model,args).to(self.device)
        self.meme_optimizer = init_optimizer(self.meme, self.args)
        if args.algorithm != 'fed_adv':
            self.meme = Model.SimCLR(args, in_channel).to(self.device)
            self.meme_optimizer = optim.Adam(self.meme.parameters(), lr=args.cl_lr, weight_decay=5e-4)
            
        self.Dict = self.meme.state_dict()

        afsche_local = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_local = GradualWarmupScheduler(self.optimizer, total_epoch=args.ite_warmup,
                                               after_scheduler=afsche_local)

        afsche_meme = optim.lr_scheduler.ReduceLROnPlateau(self.meme_optimizer, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_meme = GradualWarmupScheduler(self.meme_optimizer, total_epoch=args.ite_warmup,
                                               after_scheduler= afsche_meme)

        # self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=args.classes)

    def fork(self, global_node):
        self.meme = copy.deepcopy(global_node.model).to(self.device)
        # self.meme_optimizer = init_optimizer(self.meme, self.args)
        self.meme_optimizer = optim.Adam(self.meme.parameters(), lr=self.args.cl_lr, weight_decay=5e-4)

    def local_fork_ssl(self, global_model):
        # print(f"global: {global_model.model.state_dict()}")
        # self.clser.load_state_dict(global_model.model.state_dict())
        self.cl_model = copy.deepcopy(global_model.model)
        self.optm_cl = optim.Adam(self.cl_model.parameters(), lr=self.args.cl_lr, weight_decay=5e-4)
        self.ssl_scheduler = optim.lr_scheduler.StepLR(self.optm_cl, step_size=100, gamma=0.99)
        self.optm_fc = optim.SGD(self.cl_model.prediction.parameters(), lr=self.args.cls_lr, weight_decay=5e-4)

    def local_fork_gen(self, global_model):
        # print(f"global: {global_model.model.state_dict()}")
        self.gen_model = copy.deepcopy(global_model.gen_model)
        self.optm_gen = optim.Adam(self.gen_model.parameters(), lr=self.args.gen_lr, weight_decay=5e-4)

    def fork_proto(self, protos):
        self.prototypes_global = protos


class Global_Node(object):
    def __init__(self, test_data, args):
        self.num = 0
        self.args = args
        self.device = self.args.device
        self.proto = None
        # self.synthesis_train_dataset = None
        # self.synthesis_train_loader = None
        if args.dataset == 'rotatedmnist':
            in_channel = 1
            # self.gen_model = Model.Generator(args.latent_space, args.classes, 28*28).to(self.device)
            self.gen_model = Model.GeneratorFeature(args.latent_space, args.classes, 2*args.embedding_d).to(self.device)
            # self.gen_model = Model.Generator1(args.classes).to(self.device)
            if args.method == 'simsiam':
                self.model = SimSiam(in_channel).to(self.device)
            else:
                self.model = Model.SimCLR(args, in_channel).to(self.device)
            
            # self.model = Model.Classifier(args, self.cl_model, args.classes).to(self.device)
            # self.optm_cls = optim.Adam(self.model.fc.parameters(), lr=args.cls_lr, weight_decay=5e-4)
            # self.model = Model.SimCLR(args, in_channel).to(self.device)
            self.optm_ssl = optim.SGD(self.model.parameters(), lr=args.cls_lr, weight_decay=5e-4)
        else:
            in_channel = 3
            self.model = init_model(self.args.global_model, args).to(self.device)
        
        self.model_optimizer = init_optimizer(self.model, self.args)
        self.test_data = test_data
        self.Dict = self.model.state_dict()
        afsche_global = optim.lr_scheduler.ReduceLROnPlateau(self.model_optimizer, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_global = GradualWarmupScheduler(self.model_optimizer, total_epoch=args.ite_warmup,
                                               after_scheduler = afsche_global)

    def merge(self, Node_List):
        # 清零
        weights_zero(self.model)
        # FedAvg，每个node的meme和global model的结构一样
        Node_State_List = [copy.deepcopy(Node_List[i].meme.state_dict()) for i in range(len(Node_List))]
        for key in self.Dict.keys():
            for i in range(len(Node_List)):
                self.Dict[key] += Node_State_List[i][key]
            self.Dict[key] = self.Dict[key]/len(Node_List)
        self.model.load_state_dict(self.Dict)

    def merge_weights_gen(self, Node_List, acc_list):
        # 归一化
        # acc_list_norm = [float(acc) / sum(acc_list) for acc in acc_list]

        weights_zero(self.gen_model)
        # FedAvg，每个node的meme和global model的结构一样
        Node_State_List_ = [copy.deepcopy(Node_List[i].gen_model.state_dict()) for i in range(len(Node_List))]
        dict_ = self.gen_model.state_dict()
        for key in dict_.keys():
            for i in range(len(Node_List)):
                dict_[key] += Node_State_List_[i][key]
            dict_[key] = dict_[key]/len(Node_List)
        # print(f"simclr Dict: {dict_}")
        self.gen_model.load_state_dict(dict_) 

        # self.synthesis_train_dataset = SampleGenerator(1000, self.args.latent_space, [Node_i.gen_model for Node_i in Node_List], 10, self.args.device)
        # self.synthesis_train_loader = DataLoader(self.synthesis_train_dataset, batch_size=128, shuffle=False)


        '''
        # class 模型清零
        weights_zero(self.model)
        # FedAvg，每个node的meme和global model的结构一样
        Node_State_List = [copy.deepcopy(Node_List[i].clser.state_dict()) for i in range(len(Node_List))]
        dict_1 = self.model.state_dict()
        for key in dict_1.keys():
            for i in range(len(Node_List)):
                dict_1[key] += Node_State_List[i][key]
                # dict_1[key] += (Node_State_List[i][key].float()*acc_list_norm[i]).long()
            dict_1[key] = dict_1[key]/len(Node_List)
        # print(f"self.Dict: {self.Dict}")
        self.model.load_state_dict(dict_1)
        '''
    def merge_weights_ssl(self, Node_List, acc_list):
        # acc_list_norm = [float(acc) / sum(acc_list) for acc in acc_list]
        weights_zero(self.model)
        # FedAvg，每个node的meme和global model的结构一样
        Node_State_List = [copy.deepcopy(Node_List[i].cl_model.state_dict()) for i in range(len(Node_List))]
        dict_1 = self.model.state_dict()
        for key in dict_1.keys():
            for i in range(len(Node_List)):
                dict_1[key] += Node_State_List[i][key]
                # dict_1[key] += (Node_State_List[i][key].float()*acc_list_norm[i]).long()
            dict_1[key] = dict_1[key]/len(Node_List)
        # print(f"self.Dict: {self.Dict}")
        self.model.load_state_dict(dict_1)


    def aggregate(self, Node_List):
        Pro_List = [Node_List[i].prototypes for i in range(len(Node_List))]
        stacked_tensor = torch.stack(Pro_List)

        # 沿着指定的维度求平均值
        average_tensor = torch.mean(stacked_tensor, dim=0)
        print(average_tensor.shape)
        # L2 归一化
        # average_tensor = F.normalize(average_tensor, p=2, dim=1)
        self.proto = average_tensor
        return average_tensor

    def fork(self, node):
        self.model = copy.deepcopy(node.meme).to(self.device)
        self.model_optimizer = init_optimizer(self.model, self.args)

    # def fork_local(self, node):
    #     self.model = copy.deepcopy(node.model).to(self.device)
    #     self.model_optimizer = init_optimizer(self.model, self.args)

    def train(self, round, logger, sw):
        self.model.train()
        for epo in range(self.args.server_e):
            total_loss = 0
            loss = 0
            num = 0
            

            # for (data1, data2) in zip(self.synthesis_train_loader, self.test_data):
            #     self.optm_ssl.zero_grad()
            #     if data1 is None:
            #         images1, labels1 = torch.tensor([]), torch.tensor([])
            #     else:
            #         images1, labels1 = data1
            #         images1 = images1.view(images1.size(0), 1, 28, 28)
            #         images1.detach()
                    
            #     if data2 is None:
            #         images2, labels2 = torch.tensor([]), torch.tensor([])
            #     else:
            #         images2, labels2 = data2

            #     # 将数据移动到设备上
            #     images1, labels1 = images1.to(self.device).detach(), labels1.to(self.device).detach()
            #     images2, labels2 = images2.to(self.device), labels2.to(self.device)
                
            #     # # 合并两个批次的数据
            #     images = torch.cat((images1, images2), 0)
            #     labels = torch.cat((labels1, labels2), 0)

            for idx, (images, labels) in enumerate(self.test_data):
            # for idx, (images, labels) in enumerate(self.synthesis_train_loader):
                # 将输入数据移动到指定设备上
                images = images.to(self.device)
                labels = labels.to(self.device)
                labels = labels.detach()
            
                # z_ = torch.randn(64, self.args.latent_space).to(self.device)
                # labels = torch.randint(0, 10, (64,))
                # y_ = torch.eye(self.args.classes)[labels].to(self.device)  # 将类别转换为one-hot编码

                # # 假样本
                # fake_imgs = self.gen_model(z_, y_).detach()
                # features_f, outputs_f = self.model(fake_imgs.view(64, 1, 28, 28))
                # loss_ce = nn.CrossEntropyLoss()(outputs_f, y_)  # 使用logits计算交叉熵损失
                features, embeds, outputs = self.model(images.view(images.size(0), 1, 28, 28).detach())
                # pseudo_labels = compute_distances(features, self.proto)
                temp = self.proto[labels].detach()
                loss = kl_loss(features, temp)
                loss.backward()
                self.optm_ssl.step()
                total_loss += loss.item()
                num += 1

            sw.add_scalar(f'Train-ssl/loss/{self.num}', total_loss / num, round*self.args.server_e+epo)
            logger.info('S Epoch [%d/%d], node %d: Loss: %.4f' % (epo, self.args.server_e, self.num, total_loss / num))