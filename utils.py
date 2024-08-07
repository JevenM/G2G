import torch
import Node
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
import random
import os
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.models.feature_extraction import create_feature_extractor
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import torch.nn.functional as F

'''
def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)
'''

class KL_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(
            teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * \
            nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss

def Norm_(x, y):
    # print(x.shape, y.size())
    temp = torch.norm(x - y, dim=1, p=2)
    # print(temp.shape)
    distances = temp.mean()
    # print(distances)
    return distances
        

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

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.count = 0
        self.sum = 0
    def update(self,val,n=1):
        self.count += n
        self.sum += val*n
    def average(self):
        return self.sum/self.count
    def __repr__(self):
        r = self.sum/self.count
        if r<1e-3:
            return '{:.2e}'.format(r)
        else:
            return '%.4f'%(r)

class Recorder(object):
    def __init__(self, args, logger):
        self.args = args
        self.counter = 0
        self.tra_loss = {}
        self.tra_acc = {}
        self.val_loss = {}
        self.val_acc = {}
        self.target_acc = {}
        self.logger = logger
        for i in range(self.args.node_num + 1):
            self.val_loss[str(i)] = []
            self.val_acc[str(i)] = []
            self.target_acc[str(i)] = []
        self.acc_best = torch.zeros(self.args.node_num + 1)
        self.get_a_better = torch.zeros(self.args.node_num + 1)

    def validate(self, node, sw):
        self.counter += 1
        if self.args.algorithm == 'fed_avg':
            if node.num != 0:
                node.meme.to(node.device).eval()
            else:
                node.model.to(node.device).eval()
        else:
            node.model.to(node.device).eval()

        total_loss = 0.0
        correct = 0.0
        true_labels = []
        pred_labels = []
        out_labels = []
        # 测试编码器的准确率在源域
        with torch.no_grad():
            for idx, (data, target) in enumerate(node.test_data):
                data, target = data.to(node.device), target.to(node.device)
                output = None
                if self.args.algorithm == 'fed_avg':
                    if node.num != 0:
                        output = node.meme(data)
                    else:
                        output = node.model(data)
                elif self.args.algorithm == 'fed_adv' or self.args.algorithm == 'fed_mutual' or self.args.algorithm == 'fed_adg':
                    output = node.model(data)
                elif self.args.algorithm == 'fed_sr':
                    # z = node.model.featurize(data,num_samples=20)
                    # preds = torch.softmax(node.model.cls(z),dim=1)
                    # preds = preds.view([20,-1,node.args.classes]).mean(0)
                    # output = torch.log(preds)
                    output = node.model(data)

                if isinstance(output, tuple) and self.args.algorithm == 'fed_adv':
                    features, outputs = output
                    features = F.normalize(features, p=2, dim=1)
                    if node.prototypes_global is None:
                        pred = compute_distances(features.cpu(), node.prototypes)
                    else:
                        pred = compute_distances(features.cpu(), node.prototypes_global)
                    _, outd = torch.max(outputs, dim=1)
                    true_labels.extend(target.cpu().numpy())
                    pred_labels.extend(pred.numpy())
                    out_labels.extend(outd.cpu().numpy())
                elif self.args.algorithm == 'fed_sr':
                    total_loss += torch.nn.CrossEntropyLoss()(output, target)
                    p = output.argmax(dim=1)
                    correct += p.eq(target.view_as(p)).sum().item()
                else:
                    total_loss += torch.nn.CrossEntropyLoss()(output[1], target)
                    p = output[1].argmax(dim=1)
                    correct += p.eq(target.view_as(p)).sum().item()

            if true_labels != []:
                accuracy1 = accuracy_score(true_labels, pred_labels)
                self.logger.info(f'c{node.num} on Source: pseudo Accuracy: {accuracy1}')
                accuracy2 = accuracy_score(true_labels, out_labels)
                self.logger.info(f'c{node.num} on Source: test Accuracy: {accuracy2}')
                acc = max(accuracy1, accuracy2) * 100
            else:
                total_loss = total_loss / (idx + 1)
                acc = correct / len(node.test_data.dataset) * 100
        self.val_loss[str(node.num)].append(total_loss)
        self.val_acc[str(node.num)].append(acc)

        if self.val_acc[str(node.num)][-1] > self.acc_best[node.num]:
            self.get_a_better[node.num] = 1
            self.acc_best[node.num] = self.val_acc[str(node.num)][-1]
            if self.args.save_model:
                # torch.save(node.clser.state_dict(), node.args.save_path+'/save/model/Node{:d}_{:s}_{:d}_{:s}.pt'.format(node.num, node.args.local_model, node.args.iteration, node.args.algorithm))
                torch.save(node.model.state_dict(), node.args.save_path+'/save/model/Node{:d}_{:s}_{:d}_{:s}.pt'.format(node.num, node.args.local_model, node.args.iteration, node.args.algorithm))
            
            # add warm_up lr 
            if self.args.warm_up == True and str(node.num) != '0' and self.args.algorithm == 'fed_mutual':
                node.sche_local.step(metrics=self.val_acc[str(node.num)][-1])
                node.sche_meme.step(metrics=self.val_acc[str(node.num)][-1])
            self.logger.info('##### client{:d}: Better Accuracy on S: {:.2f}%'.format(node.num, self.val_acc[str(node.num)][-1]))

        elif self.val_acc[str(node.num)][-1] <= self.acc_best[node.num]:
            self.logger.info('##### client{:d}: Not better Accuracy on S: {:.2f}%'.format(node.num, self.val_acc[str(node.num)][-1]))

    def test_on_target(self, node, sw, round):
        if self.args.algorithm == 'fed_avg':
            if node.num != 0:
                node.meme.to(node.device)
                node.meme.eval()
            else:
                node.model.to(node.device)
                node.model.eval()
        else:
            node.model.to(node.device)
            node.model.eval()

        true_labels = []
        pred_labels = []
        out_labels = []
        # 测试编码器的准确率在目标域
        with torch.no_grad():
            accuracy1 = 0
            for idx, (data, target) in enumerate(node.target_loader):
                data, target = data.to(node.device), target.to(node.device)
                true_labels.extend(target.cpu().numpy())
                outputs = None
                if self.args.algorithm == 'fed_avg':
                    if node.num != 0:
                        output = node.meme(data)
                    else:
                        output = node.model(data)
                    features, outputs = output
                elif self.args.algorithm == 'fed_sr':
                    # z = node.model.featurize(data,num_samples=20)
                    # preds = torch.softmax(node.model.cls(z),dim=1)
                    # preds = preds.view([20,-1,node.args.classes]).mean(0)
                    # outputs = torch.log(preds)
                    outputs = node.model(data)
                elif self.args.algorithm == 'fed_adv':
                    output = node.model(data)
                    features, outputs = output
                    features = F.normalize(features, p=2, dim=1)
                    if node.prototypes_global is None:
                        pred = compute_distances(features.cpu(), node.prototypes)
                    else:
                        pred = compute_distances(features.cpu(), node.prototypes_global)
                    pred_labels.extend(pred.numpy())
                else:
                    output = node.model(data)
                    features, outputs = output
                
                # similarity_scores = torch.matmul(features, prototypes.t())  # 计算相似度(效果不如L2)
                # _, pred = torch.max(similarity_scores, dim=1)  # 选择最相似的类别作为预测标签
                _, outd = torch.max(outputs, dim=1)
                out_labels.extend(outd.cpu().numpy())

            accuracy2 = accuracy_score(true_labels, out_labels)
            self.logger.info(f'c{node.num} on Target: test Accuracy: {accuracy2}')
            if self.args.algorithm == 'fed_adv':
                accuracy1 = accuracy_score(true_labels, pred_labels)
                self.logger.info(f'c{node.num} on Target: pseudo Accuracy: {accuracy1}')
            acc = max(accuracy1, accuracy2) * 100
            self.logger.info(f"Better Acc of c{node.num} on Target domain is {acc}%")
            self.target_acc[str(node.num)].append(acc)
            sw.add_scalar(f'Test-target/{node.num}/pseu', accuracy1, round+1)
            sw.add_scalar(f'Test-target/{node.num}/true', accuracy2, round+1)

    def server_test_on_target(self, node, sw, round):
        # node.num == 0
        node.model.to(node.device)
        node.model.eval()
        true_labels = []
        pred_labels = []
        out_labels = []
        # 测试编码器的准确率在目标域
        with torch.no_grad():
            accuracy1 = 0
            for idx, (data, target) in enumerate(node.test_data):
                data, target = data.to(node.device), target.to(node.device)
                if self.args.algorithm != 'fed_sr':
                    output = node.model(data)
                    feature, outputs = output
                    if self.args.algorithm == 'fed_adv':
                        feature = F.normalize(feature.cpu(), p=2, dim=1)
                        pred = compute_distances(feature, node.proto)
                        pred_labels.extend(pred.numpy())
                else:
                    # z = node.model.featurize(data,num_samples=20)
                    # 1!!!!!!!!!!torch.Size([10240, 512])
                    # self.logger.info(f"1!!!!!!!!!!{z.shape}")
                    # preds = torch.softmax(node.model.cls(z),dim=1)
                    # 2!!!!!!!!!!torch.Size([10240, 10])
                    # self.logger.info(f"2!!!!!!!!!!{preds.shape}")
                    # preds = preds.view([20,-1,node.args.classes]).mean(0)
                    # 3!!!!!!!!!!torch.Size([512, 10])
                    # self.logger.info(f"3!!!!!!!!!!{preds.shape}")
                    # outputs = torch.log(preds)
                    # 4!!!!!!!!!!torch.Size([512, 10])
                    # self.logger.info(f"4!!!!!!!!!!{outputs.shape}")
                    outputs = node.model(data)
                
                _, outd = torch.max(outputs, dim=1)
                true_labels.extend(target.cpu().numpy())
                out_labels.extend(outd.cpu().numpy())
            if self.args.algorithm == 'fed_adv':
                accuracy1 = accuracy_score(true_labels, pred_labels)
                self.logger.info(f's{node.num} on Target: pseudo Accuracy: {accuracy1}')
            accuracy2 = accuracy_score(true_labels, out_labels)
            self.logger.info(f's{node.num} on Target: test Accuracy: {accuracy2}')
            acc = max(accuracy1, accuracy2) * 100
            self.logger.info(f"Better Acc of c{node.num} on Target domain is {acc}%")
            self.target_acc[str(node.num)].append(acc)
            sw.add_scalar(f'Test-target/{node.num}', acc, round+1)
            sw.add_scalar(f'Test-target/{node.num}/pseu', accuracy1, round+1)
            sw.add_scalar(f'Test-target/{node.num}/true', accuracy2, round+1)
        if self.args.save_model:
            save_path = os.path.join(self.args.save_path+'/save/model/', str(node.num)+self.args.global_model+'_model.pth')
            torch.save(node.model.state_dict(), save_path)

    def log(self, node):
        # print(node.num)
        return self.val_acc[str(node.num)][-1], self.val_loss[str(node.num)][-1]

    def printer(self, node):
        # num==0是global model
        if self.get_a_better[node.num] == 1 and node.num == 0:
            self.logger.info('client{:d}: A Better Accuracy: {:.2f}%! Model Saved!'.format(node.num, self.acc_best[node.num]))
            self.get_a_better[node.num] = 0
        elif self.get_a_better[node.num] == 1:
            self.get_a_better[node.num] = 0
    

    def finish(self):
        torch.save([self.val_loss, self.val_acc],
                   self.args.save_path+'/save/record/loss_acc_{:s}_{:s}_{:d}_{:s}.pt'.format(self.args.algorithm, self.args.notes, self.args.iteration, self.args.algorithm))
        self.logger.info('Finished!')
        for i in range(self.args.node_num + 1):
            self.logger.info(f'client{i}: Best acc on S = {self.acc_best[i]}')
        for key, value in self.target_acc.items():
            if value != []:
                self.logger.info(f"client{key}, Best acc on T = {max(value)}")

        for i in range(self.args.node_num + 1):
            self.logger.info(f'node{i}: list: {self.val_acc[str(i)]}')
        for key, value in self.target_acc.items():
            if value != []:
                self.logger.info(f"client{key}, list: {value}")



def plot_confusion_matrix(cm, classes, save_path, title='Confusion matrix', normalize=False, cmap=plt.cm.Greens):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        # print(cm)
    # else:
        # print('显示具体数字：')
        # print(cm)
    plt.figure(figsize=(8,8))
    # plt.imshow 负责对图像进行处理，并显示其格式，但是不能显示
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes), dtype=np.int32)
    # plt.tick_params(labelsize=16) # 设置类别文字大小
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize='10')
    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()


def get_all_preds(model, loader):
    '''
    Get all prediction results of a dataset.

    Args:
    ---
        `model`: network model
        `loader`: dataset loader

    Returns:
    ---
        `all_preds`: all prediction results of full dataset In the form of a one-dimensional tensor.
    '''
    all_preds = torch.tensor([])
    all_feats = torch.tensor([])
    for batch in loader:
        data, _ = batch
        feats,preds = model(data)
        all_preds = torch.cat(
            (all_preds, preds),
            dim=0
        )
        all_feats = torch.cat(
            (all_feats, feats),
            dim=0
        )
    return all_preds, all_feats


def my_confusion_matrix(node, Data, save_path, logger):
    '''
    Call `plot_confusion_matrix` function to draw Confuse-Matrix.

    Args:
    ---
        `model`: 模型
        `loader`: 数据集加载器
        `targets`(list): train_set.targets, 是[0,1,2,3,4,...9]的数字, 不是one-hot
        `save_path`(str): 保存路径
        `name`(tuple): 类别标签元组
    '''
    plt.rcParams.update({'font.size': 12})  # font size 10 12 14 16 main 16
    plt.rcParams['lines.linewidth'] = 2
    
    names = Data.classes
    node.model.eval()
    model = node.model.cpu()
    if node.num == 0:
        targets = node.test_data.dataset.targets
        train_preds, all_feats = get_all_preds(model, node.test_data)
        feature = F.normalize(all_feats.cpu(), p=2, dim=1)
        pred = compute_distances(feature, node.proto)
    else:
        targets = node.target_loader.dataset.targets
        train_predsm, all_feats = get_all_preds(model, node.target_loader)
        feature = F.normalize(all_feats.cpu(), p=2, dim=1)
        pred = compute_distances(feature, node.prototypes)
    # print(f"cm1111111111: {targets}, {train_preds.argmax(dim=1)}")
    # cm = confusion_matrix(targets, train_preds.argmax(dim=1))
    
    cm = confusion_matrix(targets, pred)
    diagonal_sum = np.trace(cm)
    total_sum = np.sum(cm)
    logger.info(f"cm: {cm}, diag:{diagonal_sum}, total:{total_sum}")
    np.save(os.path.join(node.args.save_path+'/save/', f'cm_target_node{node.num}_{node.args.dataset}.npy'), cm)

    # print(f"cm: {cm}")
    plot_confusion_matrix(cm, names, save_path)



def dimension_reduction(node, Data, round):
    marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    class_list = Data.classes
    class_to_idx = Data.class_to_idx
    n_class = len(class_list) # 测试集标签类别数
    palette = sns.hls_palette(n_class) # 配色方案
    sns.palplot(palette)
    # 随机打乱颜色列表和点型列表
    random.seed(1234)
    random.shuffle(marker_list)
    random.shuffle(palette)
    node.model.to(node.device)
    node.model.eval()
    if node.num != 0:
        model_trunc = create_feature_extractor(node.model, return_nodes={'net': 'semantic_feature'})
        #1 源域
        data_loader = torch.utils.data.DataLoader(node.test_data.dataset, batch_size=1, shuffle=False)
        encoding_array = []
        labels_list = []
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(node.device), labels.to(node.device)
            labels_list.append(labels.item())
            feature = model_trunc(images)['semantic_feature'].squeeze().flatten().detach().cpu().numpy() # 执行前向预测，得到 avgpool 层输出的语义特征
            encoding_array.append(feature)
        encoding_array = np.array(encoding_array)
        print(f"labels_list: {labels_list}")
        if node.args.save_model:
            # 保存为本地的 npy 文件
            np.save(os.path.join(node.args.save_path+'/save/', f'client{node.num}_{round}_clser源域{Data.client[node.num-1]}测试集语义特征_{node.args.dataset}.npy'), encoding_array)

        print(f"源域{Data.client[node.num-1]}, encoding_array.len = {len(encoding_array)}, labels_list_.len = {len(labels_list)}")

        # for method in ['PCA', 'TSNE']:
        for method in ['TSNE']:
            #选择降维方法
            if method == 'PCA': 
                X_2d = PCA(n_components=2).fit_transform(encoding_array)
            if method == 'TSNE': 
                X_2d = TSNE(n_components=2, random_state=0, n_iter=20000).fit_transform(encoding_array)
            
            print(f"Source len(X_2d) = {len(X_2d)}")

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

            dim_reduc_save_path = os.path.join(node.args.save_path+'/save/', f'client{node.num}_{round}_clser_{node.args.dataset}_源域{Data.client[node.num-1]}语义特征{method}二维降维可视化.pdf')

            plt.savefig(dim_reduc_save_path, dpi=300, bbox_inches='tight') # 保存图像

    if node.num == 0:
        model_trunc = create_feature_extractor(node.model, return_nodes={'net': 'semantic_feature'})
        data_loader_t = torch.utils.data.DataLoader(node.test_data.dataset, batch_size=1, shuffle=False)
    else:
        model_trunc = create_feature_extractor(node.model, return_nodes={'net': 'semantic_feature'})
        #1 目标域
        data_loader_t = torch.utils.data.DataLoader(node.target_loader.dataset, batch_size=1, shuffle=False)
    encoding_array_ = []
    labels_list_ = []
    for batch_idx, (images, labels) in enumerate(data_loader_t):
        images, labels = images.to(node.device), labels.to(node.device)
        # print(labels)
        # one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
        # labels = labels.unsqueeze(1)
        # print(labels)
        labels_list_.append(labels.item())
        feature = model_trunc(images)['semantic_feature'].squeeze().flatten().detach().cpu().numpy() # 执行前向预测，得到 avgpool 层输出的语义特征
        encoding_array_.append(feature)
    encoding_array_ = np.array(encoding_array_)
    if node.args.save_model:
        # 保存为本地的 npy 文件
        np.save(os.path.join(node.args.save_path+'/save/', f'node{node.num}_{round}_clser目标域{Data.client[-1]}测试集语义特征_{node.args.dataset}.npy'), encoding_array_)

    print(f"目标域: {Data.client[-1]}, encoding_array_.len = {len(encoding_array_)}, labels_list_.len = {len(labels_list_)}")

    for method in ['TSNE']:
        #选择降维方法
        if method == 'PCA': 
            X_2d_ = PCA(n_components=2).fit_transform(encoding_array_)
        if method == 'TSNE': 
            X_2d_ = TSNE(n_components=2, random_state=0, n_iter=20000).fit_transform(encoding_array_)

        # class_to_idx = test_dataset.class_to_idx
        print(f"Target len(X_2d_) = {len(X_2d_)}")
        plt.figure(figsize=(14, 14))
        for idx, fruit in enumerate(class_list): # 遍历每个类别
            #print(fruit)
            # 获取颜色和点型
            color = palette[idx]
            marker = marker_list[idx%len(marker_list)]
            # 找到所有标注类别为当前类别的图像索引号
            indices = np.where(np.array(labels_list_, dtype=object)==class_to_idx[fruit])
            plt.scatter(X_2d_[indices, 0], X_2d_[indices, 1], color=color, marker=marker, label=fruit, s=150)
        plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
        plt.xticks([])
        plt.yticks([])

        dim_reduc_save_path = os.path.join(node.args.save_path+'/save/', f'node{node.num}_{round}_clser_{node.args.dataset}_目标域{Data.client[-1]}语义特征{method}二维降维可视化.pdf')

        plt.savefig(dim_reduc_save_path, dpi=300, bbox_inches='tight') # 保存图像

# def Catfish(Node_List, args):
#     if args.catfish is None:
#         pass
#     else:
#         Node_List[0].model = Node.init_model(args.catfish)
#         Node_List[0].optimizer = Node.init_optimizer(Node_List[0].model, args)

# 计算距离矩阵
def compute_distances(features, prototypes):
    # print(features.shape)
    # print(prototypes.shape)
    distances = torch.norm(torch.stack([f - prototypes for f in features]), dim=2)
    # print(distances.shape)
    # features = F.softmax(features, dim=1)
    # prototypes = F.softmax(prototypes, dim=1)
    # distances = torch.stack([F.kl_div(f.log(), prototypes, reduction='none').sum(dim=1) for f in features])
    
    res = torch.argmin(distances, dim=1)
    # print(distances.shape)
    return res

def to_img(x, dataset):
    # from torchvision import transforms
    # x = transforms.ToTensor()(x)
    # x = transforms.ToPILImage()(x)
    # mean = torch.as_tensor([0.485, 0.456, 0.406])
    # std = torch.as_tensor([0.229, 0.224, 0.225])
    # out = x.add_(mean).mul_(std)
    if dataset == 'rotatedmnist':
        # out = 0.5 * (x + 0.5)
        out = x.clamp(0, 1)  # Clamp函数可以将随机变化的数值限制在一个给定的区间[min, max]内：
        out = out.view(-1, 1, 28, 28)
    else:
        out = x.view(-1, 3, 225, 225)  # view()函数作用是将一个多行的Tensor,拼接成一行
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
    logger.info("iid:{}".format(args.iid == 1))
    logger.info("global epochs:{},\tlocal epochs:{}".format(args.R, args.E))
    logger.info("global_model:{},\tlocal model:{}".format(args.global_model, args.local_model))




# 参考：https://www.cnblogs.com/wanghui-garcia/p/11393076.html
import os, torchvision
import torch.nn as nn
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import torch


def tensor2im(input_image, imtype=np.uint8):
    """"将tensor的数据类型转成numpy类型，并反归一化.

    Parameters:
        input_image (tensor) --  输入的图像tensor数组
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = [0.485,0.456,0.406] #自己设置的
    std = [0.229,0.224,0.225]  #自己设置的
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_img(im, path, size):
    """im可是没经过任何处理的tensor类型的数据,将数据存储到path中

    Parameters:
        im (tensor) --  输入的图像tensor数组
        path (str)  --  图像保存的路径
        size (int)  --  一行有size张图,最好是2的倍数
    """
    if size > 8:
        im_grid = torchvision.utils.make_grid(im) #将batchsize的图合成一张图
    else:
        im_grid = torchvision.utils.make_grid(im, size) #将batchsize的图合成一张图
        # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    
    image_numpy = im_grid.cpu().float().numpy()  # convert it into a numpy array
    if image_numpy.shape[0] == 1:  # grayscale to RGB
        im_numpy = im_grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    else:
        im_numpy = tensor2im(im_grid) #转成numpy类型并反归一化
    im_array = Image.fromarray(im_numpy)
    im_array.save(path)


def exp_details(args, logger):
    from prettytable import PrettyTable
    table = PrettyTable(["key", "value"])
    # table.align["key"] = "l"
    # table.align["value"] = "r"
    # print('Experimental details (all hyper-parameters):')
    logger.info('Experimental details (all hyper-parameters):\n')
    # print('-' * 70)
    # print('|%25s | %40s  |' % ('keys', 'values'))
    # print('-' * 70)
    for k in args.__dict__:
        # log.info(k + ": " + str(args.__dict__[k]))
        # print('|%25s | %40s  |' % (k, str(args.__dict__[k])))
        table.add_row([k, str(args.__dict__[k])])
    # print(table)
    logger.info(table)