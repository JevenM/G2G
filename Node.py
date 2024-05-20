import copy
import torch
import Model
from torch import optim
from simsiam import SimSiam
from utils import GradualWarmupScheduler
import torch.nn.functional as F
import torch.nn as nn

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
        else:
            flatten_dim = 225*225*3
            in_channel = 3
        # self.gen_model = Model.Generator1(args.classes, flatten_dim).to(self.device)
        self.gen_model = Model.Generator(args.latent_space, args.classes, flatten_dim).to(self.device)
        self.optm_gen = optim.Adam(self.gen_model.parameters(), lr=args.gen_lr, weight_decay=5e-4)
        if args.method == 'simclr':
            self.cl_model = Model.SimCLR(args, in_channel).to(self.device)
        elif args.method == 'simsiam':
            self.cl_model = SimSiam(in_channel).to(self.device)
        self.optm_cl = optim.Adam(self.cl_model.parameters(), lr=args.cl_lr, weight_decay=5e-4)
        # 判断是否是真样本
        self.disc_model = Model.Discriminator(flatten_dim, args.classes).to(self.device)
        self.optm_disc = optim.Adam(self.disc_model.parameters(), lr=args.disc_lr, weight_decay=5e-4)
        # 判断是否是目标域
        self.disc_model2 = Model.Discriminator2(flatten_dim).to(self.device)
        self.optm_disc2 = optim.Adam(self.disc_model2.parameters(), lr=args.disc_lr, weight_decay=5e-4)

        self.clser = Model.Classifier(args, self.cl_model, args.classes).to(self.device)
        self.optm_cls = optim.Adam(self.clser.fc.parameters(), lr=args.cls_lr, weight_decay=5e-4)

        self.meme = init_model(self.args.global_model,args).to(self.device)
        self.meme_optimizer = init_optimizer(self.meme, self.args)
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
        self.meme_optimizer = init_optimizer(self.meme, self.args)

    def local_fork(self, global_model):
        # print(f"global: {global_model.model.state_dict()}")
        self.clser.load_state_dict(global_model.model.state_dict())

    def local_fork_gen(self, global_model):
        # print(f"global: {global_model.model.state_dict()}")
        self.gen_model.load_state_dict(global_model.gen_model.state_dict())

    def fork_proto(self, protos):
        self.prototypes_global = protos


class Global_Node(object):
    def __init__(self, test_data, args):
        self.num = 0
        self.args = args
        self.device = self.args.device
        
        if args.dataset == 'rotatedmnist':
            in_channel = 1
            self.gen_model = Model.Generator(args.latent_space, args.classes, 28*28).to(self.device)
            # self.gen_model = Model.Generator1(args.classes).to(self.device)
            if args.method == 'simclr':
                self.cl_model = Model.SimCLR(args, in_channel).to(self.device)
            elif args.method == 'simsiam':
                self.cl_model = SimSiam(in_channel).to(self.device)
            
            self.model = Model.Classifier(args, self.cl_model, args.classes).to(self.device)
            self.optm_cls = optim.Adam(self.model.fc.parameters(), lr=args.cls_lr, weight_decay=5e-4)
        else:
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

    def merge_weights(self, Node_List, acc_list):
        # 归一化
        acc_list_norm = [float(acc) / sum(acc_list) for acc in acc_list]

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

        # 清零
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


    def aggregate(self, Node_List):
        Pro_List = [Node_List[i].prototypes for i in range(len(Node_List))]
        stacked_tensor = torch.stack(Pro_List)

        # 沿着指定的维度求平均值
        average_tensor = torch.mean(stacked_tensor, dim=0)
        print(average_tensor.shape)
        # L2 归一化
        average_tensor = F.normalize(average_tensor, p=2, dim=1)
        return average_tensor

    def fork(self, node):
        self.model = copy.deepcopy(node.meme).to(self.device)
        self.model_optimizer = init_optimizer(self.model, self.args)

    # def fork_local(self, node):
    #     self.model = copy.deepcopy(node.model).to(self.device)
    #     self.model_optimizer = init_optimizer(self.model, self.args)

    def train_classifier(self, round, logger, sw):
        self.model.train()
        for epo in range(100):
            loss_ce = 0,0
            self.optm_cls.zero_grad()
            z_ = torch.randn(64, self.args.latent_space).to(self.device)
            labels = torch.randint(0, 10, (64,))
            y_ = torch.eye(self.args.classes)[labels].to(self.device)  # 将类别转换为one-hot编码

            # 假样本
            fake_imgs = self.gen_model(z_, y_).detach()
            features_f, outputs_f = self.model(fake_imgs.view(64, 1, 28, 28))

            loss_ce = nn.CrossEntropyLoss()(outputs_f, y_)  # 使用logits计算交叉熵损失
            loss_ce.backward()
            self.optm_cls.step()
            sw.add_scalar(f'Train-cls/t_loss/{self.num}', loss_ce.item(), round*100+epo)
            logger.info('S Epoch [%d/%d], node %d: Loss: %.4f' % (epo+1, 100, self.num, loss_ce.item()))