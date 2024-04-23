import torch
import Node
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
import random
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.models.feature_extraction import create_feature_extractor
from datetime import datetime
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

class GradualWarmupScheduler(_LRScheduler):

    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, total_epoch, init_lr=1e-7, after_scheduler=None):
        self.init_lr = init_lr
        assert init_lr > 0, 'Initial LR should be greater than 0.'
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.finished = True
                return self.after_scheduler.get_lr()
            return self.base_lrs

        return [(((base_lr - self.init_lr) / self.total_epoch) * self.last_epoch + self.init_lr) for base_lr in
                self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [(((base_lr - self.init_lr) / self.total_epoch) * self.last_epoch + self.init_lr) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if (self.finished and self.after_scheduler) or self.total_epoch == 0:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

class Recorder(object):
    def __init__(self, args, logger):
        self.args = args
        self.counter = 0
        self.tra_loss = {}
        self.tra_acc = {}
        self.val_loss = {}
        self.val_acc = {}
        self.logger = logger
        for i in range(self.args.node_num + 1):
            self.val_loss[str(i)] = []
            self.val_acc[str(i)] = []
            self.val_loss[str(i)] = []
            self.val_acc[str(i)] = []
        self.acc_best = torch.zeros(self.args.node_num + 1)
        self.get_a_better = torch.zeros(self.args.node_num + 1)

    def validate(self, node):
        self.counter += 1
        node.model.to(node.device).eval()
        total_loss = 0.0
        correct = 0.0
        true_labels = []
        pred_labels = []
        out_labels = []
        with torch.no_grad():
            for idx, (data, target) in enumerate(node.test_data):
                data, target = data.to(node.device), target.to(node.device)
                output = node.model(data)
                if isinstance(output, tuple):
                    features, outputs = output
                    pred = compute_distances(features, node.prototypes)
                    # similarity_scores = torch.matmul(features, prototypes.t())  # 计算相似度(效果不如L2)
                    # _, pred = torch.max(similarity_scores, dim=1)  # 选择最相似的类别作为预测标签
                    _, outd = torch.max(outputs, dim=1)
                    true_labels.extend(target.cpu().numpy())
                    pred_labels.extend(pred.cpu().numpy())
                    out_labels.extend(outd.cpu().numpy())
                else:
                    total_loss += torch.nn.CrossEntropyLoss()(output, target)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            if true_labels != []:
                accuracy1 = accuracy_score(true_labels, pred_labels)
                print(f'pseudo Accuracy: {accuracy1}')
                accuracy2 = accuracy_score(true_labels, out_labels)
                print(f'test Accuracy: {accuracy2}')
                acc = max(accuracy1, accuracy2) * 100
            else:
                total_loss = total_loss / (idx + 1)
                acc = correct / len(node.test_data.dataset) * 100
        self.val_loss[str(node.num)].append(total_loss)
        self.val_acc[str(node.num)].append(acc)

        if self.val_acc[str(node.num)][-1] > self.acc_best[node.num]:
            self.get_a_better[node.num] = 1
            self.acc_best[node.num] = self.val_acc[str(node.num)][-1]
            torch.save(node.model.state_dict(),
                       'save/model/Node{:d}_{:s}_{:d}_{:s}.pt'.format(node.num, node.args.local_model, node.args.iteration, node.args.algorithm))
            # add warm_up lr 
            if self.args.warm_up == True and str(node.num) != '0':
                node.sche_local.step(metrics=self.val_acc[str(node.num)][-1])
                node.sche_meme.step(metrics=self.val_acc[str(node.num)][-1])

        if self.val_acc[str(node.num)][-1] <= self.acc_best[node.num]:
            self.logger.info('##### Node{:d}: Not better Accuracy: {:.2f}%'.format(node.num, self.val_acc[str(node.num)][-1]))


        # node.meme.to(node.device).eval()
        # total_loss = 0.0
        # correct = 0.0

        # with torch.no_grad():
        #     for idx, (data, target) in enumerate(node.test_data):
        #         data, target = data.to(node.device), target.to(node.device)
        #         output = node.meme(data)
        #         total_loss += torch.nn.CrossEntropyLoss()(output, target)
        #         pred = output.argmax(dim=1)
        #         correct += pred.eq(target.view_as(pred)).sum().item()
        #     total_loss = total_loss / (idx + 1)
        #     acc = correct / len(node.test_data.dataset) * 100

    def log(self, node):
        # print(node.num)
        return self.val_acc[str(node.num)][-1], self.val_loss[str(node.num)][-1]

    def printer(self, node):
        # num==0是global model
        if self.get_a_better[node.num] == 1 and node.num == 0:
            self.logger.info('Node{:d}: A Better Accuracy: {:.2f}%! Model Saved!'.format(node.num, self.acc_best[node.num]))
            self.get_a_better[node.num] = 0
        elif self.get_a_better[node.num] == 1:
            self.get_a_better[node.num] = 0
    

    def finish(self):
        torch.save([self.val_loss, self.val_acc],
                   'save/record/loss_acc_{:s}_{:s}_{:d}_{:s}.pt'.format(self.args.algorithm, self.args.notes, self.args.iteration, self.args.algorithm))
        self.logger.info('Finished!\n')
        for i in range(self.args.node_num + 1):
            self.logger.info('Node{}: Best Accuracy = {:.2f}%'.format(i, self.acc_best[i]))

def dimension_reduction(node, Data, round):
    model_trunc = create_feature_extractor(node.clser, return_nodes={'encoder': 'semantic_feature'})

    data_loader = torch.utils.data.DataLoader(node.test_dataset, batch_size=1, shuffle=False)
    encoding_array = []
    labels_list = []
    for batch_idx, (images, labels) in enumerate(data_loader):
        images, labels = images.to(node.device), labels.to(node.device)
        labels_list.append(labels.item())
        feature = model_trunc(images)['semantic_feature'].squeeze().flatten().detach().cpu().numpy() # 执行前向预测，得到 avgpool 层输出的语义特征
        encoding_array.append(feature)
    encoding_array = np.array(encoding_array)
    # 保存为本地的 npy 文件
    np.save(os.path.join("./save/", f'{node.num}_{round}_clser测试集语义特征_mnist.npy'), encoding_array)


    marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    class_list = Data.target_data.classes
    n_class = len(class_list) # 测试集标签类别数
    palette = sns.hls_palette(n_class) # 配色方案
    sns.palplot(palette)
    # 随机打乱颜色列表和点型列表
    random.seed(1234)
    random.shuffle(marker_list)
    random.shuffle(palette)


    for method in ['PCA', 'TSNE']:
        #选择降维方法
        if method == 'PCA': 
            X_2d = PCA(n_components=2).fit_transform(encoding_array)
        if method == 'TSNE': 
            X_2d = TSNE(n_components=2, random_state=0, n_iter=20000).fit_transform(encoding_array)

        class_to_idx = Data.target_data.class_to_idx

        plt.figure(figsize=(14, 14))
        for idx, fruit in enumerate(class_list): # 遍历每个类别
            #print(fruit)
            # 获取颜色和点型
            color = palette[idx]
            marker = marker_list[idx%len(marker_list)]
            # 找到所有标注类别为当前类别的图像索引号
            indices = np.where(np.array(labels_list)==class_to_idx[fruit])
            plt.scatter(X_2d[indices, 0], X_2d[indices, 1], color=color, marker=marker, label=fruit, s=150)
        plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
        plt.xticks([])
        plt.yticks([])

        dim_reduc_save_path = os.path.join('./save/', f'{node.num}_{round}_clser_mnist_语义特征{method}二维降维可视化.pdf')

        plt.savefig(dim_reduc_save_path, dpi=300, bbox_inches='tight') # 保存图像


# def Catfish(Node_List, args):
#     if args.catfish is None:
#         pass
#     else:
#         Node_List[0].model = Node.init_model(args.catfish)
#         Node_List[0].optimizer = Node.init_optimizer(Node_List[0].model, args)

# 计算距离矩阵
def compute_distances(features, prototypes):
    distances = torch.norm(torch.stack([f - prototypes for f in features]), dim=2)
    return torch.argmin(distances, dim=1)

def to_img(x):
    # from torchvision import transforms
    # x = transforms.ToTensor()(x)
    # x = transforms.ToPILImage()(x)
    # mean = torch.as_tensor([0.485, 0.456, 0.406])
    # std = torch.as_tensor([0.229, 0.224, 0.225])
    # # out = 0.5 * (x + 0.5)
    # out = x.add_(mean).mul_(std)
    out = x.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
    out = out.view(-1, 3, 225, 225)  # view()函数作用是将一个多行的Tensor,拼接成一行
    return out

def LR_scheduler(rounds, Node_List, args, Global_node = None, logger=None):
    #     trigger = 7
    if rounds > 15 and rounds <=30:
        trigger = 15
    elif rounds > 30 and rounds <=45:
        trigger = 25
    elif rounds > 45 and rounds <=50:
        trigger = 40
    else:
        trigger = 51

    if rounds != 0 and rounds % trigger == 0 and rounds < args.stop_decay:
        args.lr *= 0.5
        for i in range(len(Node_List)):
            Node_List[i].args.lr = args.lr
            Node_List[i].args.alpha = args.alpha
            Node_List[i].args.beta = args.beta
            Node_List[i].optimizer.param_groups[0]['lr'] = args.lr
            Node_List[i].meme_optimizer.param_groups[0]['lr'] = args.lr
        if Global_node !=None:
            Global_node.args.lr = args.lr
            Global_node.model_optimizer.param_groups[0]['lr'] = args.lr
    logger.info('Learning rate={:.10f}'.format(args.lr))


def Summary(args, logger):
    logger.info("Summary:")
    logger.info("algorithm:{}".format(args.algorithm))
    logger.info("iteration:{}".format(args.iteration))
    logger.info("lr:{}, is pretrained:{}".format(args.lr, args.pretrained))
    logger.info("dataset:{}\tbatchsize:{}\tclasses:{}".format(args.dataset, args.batch_size, args.classes))
    logger.info("node_num:{},\tsplit:{}".format(args.node_num, args.split))
    # print("iid:{},\tequal:{},\n".format(args.iid == 1, args.unequal == 0))
    logger.info("global epochs:{},\tlocal epochs:{}".format(args.R, args.E))
    logger.info("global_model:{},\tlocal model:{}".format(args.global_model, args.local_model))