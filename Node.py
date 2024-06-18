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


def get_model(model_type, args):
    return Model.BaseModel(model_type, args) 


def init_optimizer(model, args, in_lr):
    optimizer = []
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=in_lr, momentum=args.momentum, weight_decay=1e-7)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=in_lr, weight_decay=1e-4)
    return optimizer


def weights_zero(model):
    for p in model.parameters():
        if p.data is not None:
            p.data.detach_()
            p.data.zero_()


class Node(object):
    def __init__(self, num, train_data, test_data, args, target_load=None):
        self.args = args
        # num == 0 是服务器
        self.num = num + 1
        self.prototypes = None
        self.prototypes_global = None
        self.device = args.device

        self.train_data = train_data
        self.test_data = test_data
        self.target_loader = target_load

        # 本地个性化模型
        self.model = get_model(self.args.local_model, args).to(self.device)
        self.optimizer = init_optimizer(self.model, args, args.lr)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=25, gamma=0.99) # type: ignore
        self.optm_fc = init_optimizer(self.model.cls, args, args.cls_lr)

        dim = self.model.out_dim
        self.gen_model = Model.GeneratorFeature(args.latent_space, args.classes, dim).to(self.device)
        self.optm_gen = init_optimizer(self.gen_model, args, args.gen_lr)
        
        # 判断是否是真样本
        self.disc_model = Model.DiscriminatorFeature(dim, args.classes).to(self.device)
        self.optm_disc = init_optimizer(self.disc_model, args, args.disc_lr)
        # 判断是否是目标域
        self.disc_model2 = Model.DiscriminatorFeature2(dim).to(self.device)
        self.optm_disc2 = init_optimizer(self.disc_model2, args, args.disc_lr)

        # 本地之间流通的全局模型
        self.meme = get_model(self.args.global_model, args).to(self.device)
        self.meme_optimizer = init_optimizer(self.meme, args, args.lr)
        
        afsche_local = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_local = GradualWarmupScheduler(self.optimizer, total_epoch=args.ite_warmup,
                                               after_scheduler=afsche_local)

        afsche_meme = optim.lr_scheduler.ReduceLROnPlateau(self.meme_optimizer, factor=args.factor, patience=args.patience,
                                                          threshold=args.lr_threshold, min_lr=1e-7)
        self.sche_meme = GradualWarmupScheduler(self.meme_optimizer, total_epoch=args.ite_warmup,
                                               after_scheduler= afsche_meme)

        # self.lossFunc = LabelSmoothingLoss(args.label_smoothing, lbl_set_size=args.classes)
        if args.algorithm == 'fed_sr':
            self.r_mu = nn.Parameter(torch.zeros(args.classes,args.embedding_d, device=self.device))
            self.r_sigma = nn.Parameter(torch.ones(args.classes,args.embedding_d, device=self.device))
            self.C = nn.Parameter(torch.ones([], device=self.device))
            self.optimizer.add_param_group({'params':[self.r_mu,self.r_sigma,self.C],'lr':args.lr,'momentum':0.9})


    def fork(self, global_node):
        self.meme = copy.deepcopy(global_node.model).to(self.device)
        self.meme_optimizer = init_optimizer(self.meme, self.args, self.args.lr)

    def local_fork_ssl(self, global_model):
        # print(f"global: {global_model.model.state_dict()}")
        self.model = copy.deepcopy(global_model.model)
        self.optimizer = init_optimizer(self.model, self.args, self.args.lr)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.99)
        self.optm_fc = init_optimizer(self.model.cls, self.args, self.args.cls_lr)

    def local_fork_gen(self, global_model):
        # print(f"global: {global_model.model.state_dict()}")
        self.gen_model = copy.deepcopy(global_model.gen_model)
        # self.optm_gen = optim.Adam(self.gen_model.parameters(), lr=self.args.gen_lr, weight_decay=1e-4)
        self.optm_gen = init_optimizer(self.gen_model, self.args, self.args.gen_lr)

    def fork_proto(self, protos):
        self.prototypes_global = protos


class Global_Node(object):
    def __init__(self, test_data, args):
        self.num = 0
        self.args = args
        self.device = self.args.device
        self.proto = None
        self.test_data = test_data

        self.model = get_model(self.args.global_model, args).to(self.device)
        self.model_optimizer = init_optimizer(self.model, args, args.lr)

        dim = self.model.out_dim
        self.gen_model = Model.GeneratorFeature(args.latent_space, args.classes, dim).to(self.device)

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
                if i == 0:
                    self.Dict[key] = Node_State_List[i][key]
                else:
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
                if i == 0:
                    dict_[key] = Node_State_List_[i][key]
                else:
                    dict_[key] += Node_State_List_[i][key]
            dict_[key] = dict_[key]/len(Node_List)
        # print(f"simclr Dict: {dict_}")
        self.gen_model.load_state_dict(dict_) 

    def merge_weights_ssl(self, Node_List, acc_list=[]):
        # acc_list_norm = [float(acc) / sum(acc_list) for acc in acc_list]
        weights_zero(self.model)
        # FedAvg，每个node的meme和global model的结构一样
        Node_State_List = [copy.deepcopy(Node_List[i].model.state_dict()) for i in range(len(Node_List))]
        dict_1 = self.model.state_dict()
        for key in dict_1.keys():
            for i in range(len(Node_List)):
                if i == 0:
                    dict_1[key] = Node_State_List[i][key]
                else:
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
        self.proto = average_tensor
        return average_tensor

    def fork(self, node):
        self.model = copy.deepcopy(node.meme).to(self.device)
        self.model_optimizer = init_optimizer(self.model, self.args, self.args.lr)

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
                self.model_optimizer.zero_grad()
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
                features, _ = self.model(images)
                # pseudo_labels = compute_distances(features, self.proto)
                temp = self.proto[labels].detach()
                loss = kl_loss(features, temp)
                loss.backward()
                self.model_optimizer.step()
                total_loss += loss.item()
                num += 1

            sw.add_scalar(f'Train-ssl/loss/{self.num}', total_loss / num, round*self.args.server_e+epo)
            logger.info('S Epoch [%d/%d], node %d: Loss: %.4f' % (epo, self.args.server_e, self.num, total_loss / num))