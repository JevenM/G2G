from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
from torchvision.utils import save_image
from torchvision.models.feature_extraction import create_feature_extractor
import os
import copy
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from simsiam import D
from utils import Norm_, to_img, compute_distances, save_img
import torch.nn.functional as F

KL_Loss = nn.KLDivLoss(reduction='batchmean')
Softmax = nn.Softmax(dim=1)
LogSoftmax = nn.LogSoftmax(dim=1)
CE_Loss = nn.CrossEntropyLoss()
# 定义损失函数和优化器
criterion_BCE = nn.BCEWithLogitsLoss()

def train_normal(node,args,round):
    node.model.to(node.device).train()
    train_loader = node.train_data
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Training (the {:d}-batch): tra_Loss = {:.4f} tra_Accuracy = {:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.optimizer.zero_grad()
            epochs.set_description(description.format(idx + 1, avg_loss, acc))
            data, target = data.to(node.device), target.to(node.device)
            output = node.model(data)
            loss = CE_Loss(output, target)
            loss.backward()
            node.optimizer.step()
            total_loss += loss
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100


def train_avg(node, args, logger, round, sw, epo):
    node.meme.to(node.device).train()
    train_loader = node.train_data
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Node{:d}: loss={:.4f} acc={:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.meme_optimizer.zero_grad()
            epochs.set_description(description.format(node.num, avg_loss, acc))
            data, target = data.to(node.device), target.to(node.device)
            _,_,output = node.meme(data)
            loss = CE_Loss(output, target)
            loss.backward()
            node.meme_optimizer.step()
            total_loss += loss
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100
    sw.add_scalar(f'Train-avg/avg_loss/{node.num}', avg_loss, round*args.E+epo) # type: ignore
    sw.add_scalar(f'Train-avg/acc/{node.num}', acc, round*args.E+epo) # type: ignore
    
    logger.info('C{}-Round[{}/{}], avg_loss:{:.6f}, total_loss:{:.6f}, acc:{:.6f}'.format(node.num, round, args.R, avg_loss, total_loss, acc))
    # node.model = node.meme

def train_mutual(node,args,logger, round, sw, epo):
    node.model.to(node.device).train()
    node.meme.to(node.device).train()
    train_loader = node.train_data
    total_local_loss = 0.0
    avg_local_loss = 0.0
    correct_local = 0.0
    acc_local = 0.0
    total_meme_loss = 0.0
    avg_meme_loss = 0.0
    correct_meme = 0.0
    acc_meme = 0.0
    train_index = 0
    total_global_kl_loss = 0.0
    total_local_kl_loss = 0.0
    avg_global_kl_loss = 0.0
    avg_local_kl_loss = 0.0
    description = 'Node{:d}: loss_model={:.4f} acc_model={:.2f}% loss_meme={:.4f} acc_meme={:.2f}%'
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            train_index = train_index + 1 
            node.optimizer.zero_grad()
            node.meme_optimizer.zero_grad()
            epochs.set_description(description.format(node.num, avg_local_loss, acc_local, avg_meme_loss, acc_meme))
            data, target = data.to(node.device), target.to(node.device)
            output_local = node.model(data)
            output_meme = node.meme(data)

            # KL_Loss(input, target)
            kl_local = KL_Loss(LogSoftmax(output_local), Softmax(output_meme.detach()))    
            kl_meme = KL_Loss(LogSoftmax(output_meme), Softmax(output_local.detach()))

            
            total_local_kl_loss += kl_local
            total_global_kl_loss += kl_meme
            
            _output_local = nn.Softmax(dim=1)(output_local)
            _, src_idx = torch.sort(_output_local, 1, descending=True)
            _output_meme = nn.Softmax(dim=1)(output_meme)
            _, src_idx_meme = torch.sort(_output_meme, 1, descending=True)
            if args.topk > 0:
                topk = np.min([args.topk, args.classes])
                for i in range( _output_local.size()[0]):
                    # 将前topk除外的几个output的值进行修改，平均，使得每个类别的概率相等
                    output_local[i, src_idx[i, topk:]] = (1.0 -  _output_local[i, src_idx[i, :topk]].sum())/ ( _output_local.size()[1] - topk)
                    output_meme[i, src_idx[i, topk:]] = (1.0 -  _output_meme[i, src_idx[i, :topk]].sum())/ ( _output_meme.size()[1] - topk)


            ce_local = CE_Loss(output_local, target)
            ce_meme = CE_Loss(output_meme, target)  
            # default: alpha=0.5
            loss_local = node.args.alpha * ce_local + (1 - node.args.alpha) * kl_local
            loss_meme = node.args.beta * ce_meme + (1 - node.args.beta) * kl_meme
            loss_local.backward()
            loss_meme.backward()
            node.optimizer.step()
            node.meme_optimizer.step()

            total_local_loss += loss_local
            avg_local_loss = total_local_loss / (idx + 1)
            pred_local = output_local.argmax(dim=1)
            correct_local += pred_local.eq(target.view_as(pred_local)).sum()
            acc_local = correct_local / len(train_loader.dataset) * 100
            total_meme_loss += loss_meme
            avg_meme_loss = total_meme_loss / (idx + 1)
            pred_meme = output_meme.argmax(dim=1)
            correct_meme += pred_meme.eq(target.view_as(pred_meme)).sum()
            acc_meme = correct_meme / len(train_loader.dataset) * 100  

            avg_global_kl_loss = total_global_kl_loss / (idx + 1)
            avg_local_kl_loss = total_local_kl_loss / (idx + 1)


            # mixup (not used)
            if args.mix > 0:
                mixed_total_local_loss = 0.0
                mixed_avg_local_loss = 0.0
                mixed_correct_local = 0.0
                mixed_acc_local = 0.0
                mixed_total_meme_loss = 0.0
                mixed_avg_meme_loss = 0.0
                mixed_correct_meme = 0.0
                mixed_acc_meme = 0.0

                alpha = 0.3
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(data.size()[0]).cuda()

                other_data, other_target = data[index, :],target[index]
                other_data, other_target = other_data.to(node.device), other_target.to(node.device)

                mixed_input = lam * data + (1 - lam) * other_data
                mixed_label = lam * target + (1 - lam) * other_target

                mixed_output_local = node.model(mixed_input)
                mixed_output_meme = node.meme(mixed_input)

                # mixed_output_local,mixed_output_meme = mixed_output_local.to(node.device),mixed_output_meme.to(node.device)

                mixed_ce_local = CE_Loss(mixed_output_local, mixed_label.long())
                mixed_kl_local = KL_Loss(LogSoftmax(mixed_output_local), Softmax(mixed_output_meme.detach()))
                mixed_ce_meme = CE_Loss(mixed_output_meme, mixed_label.long())                    
                mixed_kl_meme = KL_Loss(LogSoftmax(mixed_output_meme), Softmax(mixed_output_local.detach()))
                mixed_loss_local = node.args.alpha * mixed_ce_local + (1 - node.args.alpha) * mixed_kl_local
                mixed_loss_meme = node.args.beta * mixed_ce_meme + (1 - node.args.beta) * mixed_kl_meme
                
                (args.mix * mixed_loss_local).backward()
                (args.mix * mixed_loss_meme).backward()
                node.optimizer.step()
                node.meme_optimizer.step()
                mixed_total_local_loss += mixed_loss_local
                mixed_avg_local_loss = mixed_total_local_loss / (idx + 1)
                mixed_pred_local = mixed_output_local.argmax(dim=1)
                mixed_correct_local += mixed_pred_local.eq(target.view_as(pred_local)).sum()
                mixed_acc_local = mixed_correct_local / len(train_loader.dataset) * 100
                mixed_total_meme_loss += mixed_loss_meme
                mixed_avg_meme_loss = mixed_total_meme_loss / (idx + 1)
                mixed_pred_meme = mixed_output_meme.argmax(dim=1)
                mixed_correct_meme += mixed_pred_meme.eq(mixed_label.view_as(mixed_pred_meme)).sum()
                mixed_acc_meme = mixed_correct_meme / len(train_loader.dataset) * 100      



class Trainer(object):

    def __init__(self, args, logger=None):
        if args.algorithm == 'fed_mutual':
            self.train = train_mutual
        elif args.algorithm == 'fed_avg':
            self.train = train_avg
        elif args.algorithm == 'normal':
            self.train = train_normal
        elif args.algorithm == 'fed_adv':
            self.train = train_adv

    def __call__(self, node, args, logger, round=0, sw=None, epo=None):
        self.train(node, args, logger, round, sw, epo) # type: ignore

# =========================================== my ==================================================

# latent_space = 64
# images_path = './images/'
# temperature用beta代替
# temperature=0.5
# alpha用源代码中的alpha代替
# alpha = 0.5

# Simclr
def NT_XentLoss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape 
    device = z1.device 
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

    negatives = similarity_matrix[~diag].view(2*N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2*N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)


def train_adv(node, args, logger, round, sw = None, epo=None):
    node.gen_model.to(node.device).train()
    node.cl_model.to(node.device).train()
    node.disc_model.to(node.device).train()
    node.disc_model2.to(node.device).train()
    train_loader = node.train_data

    d_loss,d_loss_1,g_loss,ls = 0,0,0,0
    data_iter = iter(node.target_loader)
    for iter_, (real_images, labels) in enumerate(tqdm(train_loader)):
        batchsize = real_images.size(0)
        real_images = real_images.to(node.device)
        
        y = torch.eye(args.classes)[labels].to(node.device)  # 将类别转换为one-hot编码
        # 真实样本的标签为1
        real_labels = torch.ones(batchsize, 1).to(node.device)
        # 生成器生成样本的标签为0
        fake_labels = torch.zeros(batchsize, 1).to(node.device)
        
        for k in range(args.discr_e):
            # 训练辨别器
            node.optm_disc.zero_grad()
            # real_outputs = discriminator(real_images.view(batchsize, -1))
            real_outputs = node.disc_model(real_images.view(batchsize, -1), y)
            loss_real = criterion_BCE(real_outputs, real_labels)
            real_scores = nn.Sigmoid()(real_outputs)  # 得到真实图片的判别值，输出的值越接近1越好
            # TODO Generator里面只用self.gen(x)的时候
            # fake_images = gen(z).to(device)
            z = torch.randn(batchsize, args.latent_space).to(node.device)
            # fake_images = node.gen_model(real_images.view(batchsize, -1), y).detach()
            fake_images = node.gen_model(z, y).detach()
            # print(fake_images[0])
            fake_outputs = node.disc_model(fake_images.to(node.device), y)
            # fake_outputs = discriminator(fake_images.to(device))
            # print(fake_outputs)
            loss_fake = criterion_BCE(fake_outputs, fake_labels)
            fake_scores = nn.Sigmoid()(fake_outputs)  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好

            # loss_disc = 0.25*loss_real + 0.25*loss_fake + 0.5*loss_real_t
            loss_disc = loss_real + loss_fake
            loss_disc.backward()
            node.optm_disc.step()
            # print(f'Epoch [{epoch+1}/10] [{k+1/str(discr_e)}], Loss Disc: {loss_disc.item()}')
        d_loss += loss_disc.item()

        for i in range(args.discr_e):
            node.optm_disc2.zero_grad()
            try:
                target_image, tar_y = next(data_iter)
                target_image = target_image.to(node.device)
                bs_t = target_image.size(0)
            except StopIteration:
                break
            true_labels_t = torch.ones(bs_t, 1).to(node.device)
            
            tar_outputs = node.disc_model2(target_image.view(bs_t, -1))
            loss_real_t = criterion_BCE(tar_outputs, true_labels_t)
            z = torch.randn(bs_t, args.latent_space).to(node.device)
            if bs_t >= batchsize:
                fake_images = node.gen_model(z[:batchsize], y).detach()
                # fake_images = node.gen_model(target_image.view(bs_t, -1)[:batchsize], y).detach()
                fake_labels_t = torch.zeros(batchsize, 1).to(node.device)
            elif bs_t < batchsize:
                fake_images = node.gen_model(z, y[:bs_t]).detach()
                # fake_images = node.gen_model(target_image.view(bs_t, -1), y[:bs_t]).detach()
                fake_labels_t = torch.zeros(bs_t, 1).to(node.device)
            # print(z.shape)
            # print(y[:bs_t])
            fake_outputs_t = node.disc_model2(fake_images.to(node.device))
            loss_fake_t = criterion_BCE(fake_outputs_t, fake_labels_t)
            loss_disc2 = loss_real_t + loss_fake_t
            loss_disc2.backward()
            node.optm_disc2.step()
        d_loss_1 += loss_disc2.item()    

        # 训练生成器
        for i in range(args.gen_e):
            node.optm_gen.zero_grad()
            z = torch.randn(batchsize, args.latent_space).to(node.device)
            gen_images = node.gen_model(z, y)
            # gen_images = node.gen_model(real_images.view(batchsize, -1), y)
            gen_outputs = node.disc_model(gen_images, y)
            loss_gen = criterion_BCE(gen_outputs, real_labels)
            gen_outputs1 = node.disc_model2(gen_images)
            loss_gen1 = criterion_BCE(gen_outputs1, real_labels)
            # loss_gen = loss_gen + loss
            # loss_gen = torch.log(1.0 - (discriminator(gen_images)).detach()) 
            loss_g = 2*loss_gen1 + 2*loss_gen
            loss_g.backward()
            node.optm_gen.step()
        # print(f'Epoch [{epoch+1}/10] [{i+1/str(gen_e)}], Loss: {loss.item()}, Loss Gen: {loss_gen.item()}')
        g_loss += loss_g.item()

        # print(f'Epoch [{epoch+1}/10], Loss: {loss.item()}, Loss Gen: {loss_gen.item()}, Loss Disc: {loss_disc.item()}')
        # 打印中间的损失
        # print('Epoch[{}/{}], d_loss:{:.6f}, g_loss:{:.6f}, loss:{:.6f}'
        #       'D real: {:.6f}, D fake: {:.6f}'.format(epoch, num_epoch, loss_disc.data.item(), loss_gen.data.item(), loss.data.item(),
        #                                              real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
        #     ))
        if round == 0 and iter_==len(train_loader)-1:
            real_images = to_img(real_images.cuda().data, args.dataset)
            save_image(real_images, os.path.join(args.save_path+"/gen_images/", '{}_real_images.png'.format(str(node.num))))
            # save_img(real_images, os.path.join(args.save_path+"/gen_images/", '_real_images.png'), batchsize)
        if iter_==len(train_loader)-1:
            fake_images = to_img(fake_images.cuda().data, args.dataset)
            save_image(fake_images, os.path.join(args.save_path+"/gen_images/", '{}_fake_images-{}.png'.format(str(node.num), round + 1)))
            # save_img(fake_images, os.path.join(args.save_path+"/gen_images/", '{}_fake_images-{}.png'.format(str(node.num), round + 1)), batchsize)
    sw.add_scalar(f'Train-adv/d_loss/{node.num}', d_loss/len(train_loader), round*args.E+epo) # type: ignore
    sw.add_scalar(f'Train-adv/d_loss1/{node.num}', d_loss_1/len(train_loader), round*args.E+epo) # type: ignore
    sw.add_scalar(f'Train-adv/g_loss/{node.num}', g_loss/len(train_loader), round*args.E+epo) # type: ignore
    sw.add_scalar(f'Train-adv/real_scores/{node.num}', real_scores.data.mean(), round*args.E+epo) # type: ignore
    sw.add_scalar(f'Train-adv/fake_scores/{node.num}', fake_scores.data.mean(), round*args.E+epo) # type: ignore
    logger.info('C{}-Round[{}/{}], d_loss:{:.6f}, d_loss1:{:.6f}, g_loss:{:.6f}, D real: {:.6f}, D fake: {:.6f}'.format(node.num, round, args.R, 
    d_loss/len(train_loader), d_loss_1/len(train_loader), g_loss/len(train_loader), real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
    ))

    if args.save_model:
        simclr_save_path = os.path.join(args.save_path+'/save/model/', str(node.num)+'_simclr.pth')
        gen_save_path = os.path.join(args.save_path+'/save/model/', str(node.num)+'_gen.pth')

        torch.save(node.cl_model.state_dict(), simclr_save_path)
        torch.save(node.gen_model.state_dict(), gen_save_path)

    node.cl_model.eval()
    # 训练完encoder之后
    class_counts = torch.zeros(args.classes)
    if args.dataset == 'rotatedmnist':
        dim = 576
    else:
        dim = 4096
    
    prototypes = torch.zeros(args.classes, dim).to(args.device)
    # train_dataset = SampleGenerator(2000, args.latent_space, [node.gen_model], 10, args.device)
    # train_loader_syn = DataLoader(train_dataset, batch_size=128, shuffle=False)
    # 遍历整个数据集
    for images, labels in node.test_data:
        images = images.to(node.device)
        feature, _, out = node.cl_model(images.view(images.size(0), 1, 28, 28))
        # feature = feature.cpu().detach()
        for i in range(args.classes):  # 遍历每个类别
            class_indices = (labels == i)  # 找到属于当前类别的样本的索引
            class_outputs = feature[class_indices]  # 提取属于当前类别的样本的特征向量
            prototypes[i] += torch.sum(class_outputs, dim=0)  # 将当前类别的特征向量累加到原型矩阵中
            class_counts[i] += class_outputs.shape[0]  # 统计当前类别的样本数量

    # 计算每个类别的平均特征向量
    for i in range(args.classes):
        if class_counts[i] > 0:
            prototypes[i] /= class_counts[i]
    # L2 归一化
    prototypes = F.normalize(prototypes, p=2, dim=1)

    # logger.info(f"Prototypes computed successfully! {node.prototypes}")
    node.prototypes = prototypes
    

# Constrastive Semantic Alignment Loss
def csa_loss(x, y, class_eq, device):
    margin = 0.5
    dist = F.pairwise_distance(x, y)
    class_eq = class_eq.to(device)
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()

def train_ssl(node, args, logger, round, sw=None):
    node.cl_model.to(node.device).train()
    train_loader = node.train_data
    # 训练编码器
    for k in trange(args.simclr_e):
        ls = 0
        running_loss_t, running_loss_sim, running_loss_ls, running_loss_ce_f = 0.0, 0.0, 0.0, 0.0
        loss = 0
        for iter_, (real_images, labels) in enumerate(train_loader):
            batchsize = real_images.size(0)
            real_images = real_images.to(node.device)
            labels = labels.to(node.device)
            node.optm_cl.zero_grad()
            z = torch.randn(batchsize, args.latent_space).to(node.device)
            # lab_ = torch.randint(0, 10, (batchsize,))
            num_same = batchsize // 2
            num_random = batchsize - num_same
            lab_same = labels.cpu()[torch.randperm(batchsize)[:num_same]]
            # 生成剩余 num_random 个随机元素
            lab_random = torch.randint(0, 10, (num_random,))
            # 将两部分拼接起来
            lab_ = torch.cat((lab_same, lab_random))
            # 打乱顺序
            lab_ = lab_[torch.randperm(batchsize)]
            # print(lab_, labels)
            y = torch.eye(args.classes)[lab_].to(node.device)
            # print(lab_[:10])
            # y = torch.eye(args.classes)[labels].to(node.device)  # 将类别转换为one-hot编码
            fake_images = node.gen_model(z, y).detach()
            # fake_images = node.gen_model(real_images.view(batchsize, -1), y).detach()
            # fake_images = gen(z)
            # print(real_images.size())
            # print(fake_images.size())
            # 计算原图和生成图像的表征
            if args.dataset == 'rotatedmnist':
                w_h = 28
                in_c = 1
            else:
                w_h = 255
                in_c = 3
            z1, embeddings_orig, out_r = node.cl_model(real_images.view(batchsize, in_c, w_h, w_h))
            z2, embeddings_gen, out_f = node.cl_model(fake_images.view(batchsize, in_c, w_h, w_h))
            
            if args.method == 'simclr':
                ls_ = NT_XentLoss(embeddings_orig, embeddings_gen)
            elif args.method == 'simsiam':
                ls_ = D(embeddings_orig, z2) / 2 + D(embeddings_gen, z1) / 2
            elif args.method == 'ccsa':
                lab_ = lab_.to(node.device)
                ls_ = csa_loss(embeddings_orig, embeddings_gen, (labels == lab_).float(), args.device)
            running_loss_ls += ls_.item()

            cc1 = torch.zeros(args.classes)
            proto1 = torch.zeros(args.classes, 576).to(args.device)
            cc2 = torch.zeros(args.classes)
            proto2 = torch.zeros(args.classes, 576).to(args.device)
            # 遍历z1对应的label
            for i in range(args.classes):  # 遍历每个类别
                class_ind = (labels == i)  # 找到属于当前类别的样本的索引
                class_outputs = z1[class_ind]  # 提取属于当前类别的样本的特征向量
                proto1[i] += torch.sum(class_outputs, dim=0)  # 将当前类别的特征向量累加到原型矩阵中
                cc1[i] += class_outputs.shape[0]  # 统计当前类别的样本数量
                class_ind2 = (lab_ == i)  # 找到属于当前类别的样本的索引
                class_outputs2 = z2[class_ind2]  # 提取属于当前类别的样本的特征向量
                proto2[i] += torch.sum(class_outputs2, dim=0)  # 将当前类别的特征向量累加到原型矩阵中
                cc2[i] += class_outputs2.shape[0]  # 统计当前类别的样本数量

            # 计算每个类别的平均特征向量
            for i in range(args.classes):
                if cc1[i] > 0:
                    proto1[i] /= cc1[i]
                if cc2[i] > 0:
                    proto2[i] /= cc2[i]
            # L2 归一化
            proto1 = F.normalize(proto1, p=2, dim=1)
            proto2 = F.normalize(proto2, p=2, dim=1)
            if node.prototypes_global is None:
                ou_loss_f = Norm_(proto1, node.prototypes.detach())
                ou_loss_f1 = Norm_(proto2, node.prototypes.detach())
            else:
                ou_loss_f = Norm_(proto1, node.prototypes_global.detach())
                ou_loss_f1 = Norm_(proto2, node.prototypes_global.detach())
            '''样本级别
            z1 = F.normalize(z1, p=2, dim=1)
            z2 = F.normalize(z2, p=2, dim=1)
            if node.prototypes_global is None:
                node.prototypes = node.prototypes.detach()
                # pseudo_labels_f = compute_distances(z2, node.prototypes)
                # cos_loss_f = KL_Loss(LogSoftmax(z2), Softmax(node.prototypes[lab_].detach()))
                ou_loss_f1 = Norm_(z2, node.prototypes[lab_].detach())
                ou_loss_f = Norm_(z1, node.prototypes[labels].detach())
            else:
                node.prototypes_global = node.prototypes_global.detach()
                # pseudo_labels_f = compute_distances(z2, node.prototypes_global)
                # ou_loss_f = KL_Loss(LogSoftmax(z2), Softmax(node.prototypes_global[lab_].detach()))
                ou_loss_f1 = Norm_(z2, node.prototypes_global[lab_].detach())
                ou_loss_f = Norm_(z1, node.prototypes_global[labels].detach())
            '''
            
            # pseudo_labels_f = pseudo_labels_f.detach()
            # loss_ce_f = CE_Loss(out_f, pseudo_labels_f)  # 使用伪标签计算交叉熵损失

            running_loss_sim += ou_loss_f.item()

            loss_ce_true = CE_Loss(out_r, labels)  # 使用logits计算交叉熵损失
            running_loss_t += loss_ce_true.item()

            loss_ce_f = CE_Loss(out_f, lab_.to(node.device))
            running_loss_ce_f += loss_ce_f.item()

            # loss = ou_loss_f + loss_ce_f + loss_ce_true + ls_
            # loss = 10*ou_loss_f + 10*ou_loss_f1 + loss_ce_f + loss_ce_true
            loss = loss_ce_f + loss_ce_true + ls_ + ou_loss_f + ou_loss_f1
        
            '''
            # 计算对比学习的损失
            # targets = torch.ones(embeddings_orig.size(0))
            # loss = criterion(embeddings_orig, embeddings_gen, targets)

            # 分母 ：X.X.T，再去掉对角线值，分析结果一行，可以看成它与除了这行外的其他行都进行了点积运算（包括out_1和out_2）,
            # 而每一行为一个batch的一个取值，即一个输入图像的特征表示，
            # 因此，X.X.T，再去掉对角线值表示，每个输入图像的特征与其所有输出特征（包括out_1和out_2）的点积，用点积来衡量相似性
            # 加上exp操作，该操作实际计算了分母
            # [2*B, D]
            out = torch.cat([embeddings_orig, embeddings_gen], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.beta)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batchsize, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * batchsize, -1)

            # 分子： *为对应位置相乘，也是点积
            # compute loss
            pos_sim = torch.exp(torch.sum(embeddings_orig * embeddings_gen, dim=-1) / args.beta)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            '''
            loss.backward()
            node.optm_cl.step()
            ls += loss.item()
        sw.add_scalar(f'Train-ssl/t_loss/{node.num}', running_loss_t / len(train_loader), round*args.simclr_e+k) # type: ignore
        sw.add_scalar(f'Train-ssl/sim_loss/{node.num}', running_loss_sim / len(train_loader), round*args.simclr_e+k) # type: ignore
        sw.add_scalar(f'Train-ssl/ce_f_loss/{node.num}', running_loss_ce_f / len(train_loader), round*args.simclr_e+k) # type: ignore
        sw.add_scalar(f'Train-ssl/ls_/{node.num}', running_loss_ls / len(train_loader), round*args.simclr_e+k) # type: ignore
        sw.add_scalar(f'Train-ssl/loss/{node.num}', ls / len(train_loader), round*args.simclr_e+k) # type: ignore
        logger.info('Epoch [%d/%d], node %d: Loss_ce_t: %.4f, Loss_f: %.4f, Loss_ce_f: %.4f, Loss_ls: %.4f, total loss: %.4f' % (k+1, args.simclr_e, node.num, running_loss_t / len(train_loader), running_loss_sim / len(train_loader), running_loss_ce_f / len(train_loader), running_loss_ls / len(train_loader), ls/len(train_loader))) # type: ignore

        # 更新学习率
        node.ssl_scheduler.step()
        # 打印当前学习率
        current_lr = node.optm_cl.param_groups[0]['lr']
        logger.info(f'Epoch {round*args.simclr_e+k}/{args.R*args.simclr_e}, Learning Rate: {current_lr}')

    node.cl_model.eval()
    # 训练完encoder之后
    class_counts = torch.zeros(args.classes)
    if args.dataset == 'rotatedmnist':
        dim = 576
    else:
        dim = 4096
    prototypes = torch.zeros(args.classes, dim).to(args.device)
    # train_dataset = SampleGenerator(2000, args.latent_space, [node.gen_model], 10, args.device)
    # train_loader_syn = DataLoader(train_dataset, batch_size=128, shuffle=False)
    # 遍历整个数据集
    for images, labels in node.test_data:
        images = images.to(node.device)
        feature, _, out = node.cl_model(images.view(images.size(0), 1, 28, 28))
        # feature = feature.cpu().detach()
        for i in range(args.classes):  # 遍历每个类别
            class_indices = (labels == i)  # 找到属于当前类别的样本的索引
            class_outputs = feature[class_indices]  # 提取属于当前类别的样本的特征向量
            prototypes[i] += torch.sum(class_outputs, dim=0)  # 将当前类别的特征向量累加到原型矩阵中
            class_counts[i] += class_outputs.shape[0]  # 统计当前类别的样本数量

    # 计算每个类别的平均特征向量
    for i in range(args.classes):
        if class_counts[i] > 0:
            prototypes[i] /= class_counts[i]
    # L2 归一化
    prototypes = F.normalize(prototypes, p=2, dim=1)

    # logger.info(f"Prototypes computed successfully! {node.prototypes}")
    node.prototypes = prototypes

def train_classifier(node, args, logger, round, sw=None):
    # 训练分类器
    node.clser.train()
    train_loader = node.train_data
    for epo in range(args.cls_epochs):
        running_loss_t, running_loss_f = 0.0, 0.0
        loss_ce = 0
        for images, labels in train_loader:
            node.optm_cls.zero_grad()
            images = images.to(node.device)
            z_ = torch.randn(images.size(0), args.latent_space).to(node.device)
            y_ = torch.eye(node.args.classes)[labels].to(node.device)  # 将类别转换为one-hot编码
            # 假样本
            fake_imgs = node.gen_model(z_, y_).detach()
            # fake_imgs = node.gen_model(images.view(images.size(0), -1), y_).detach()
            
            # fake_imgs = gen(z_)
            features_f, outputs_f = node.clser(fake_imgs.view(images.size(0), 1, 28, 28))
            features_f = F.normalize(features_f, p=2, dim=1)
            if node.prototypes_global is None:
                node.prototypes = node.prototypes.detach()
                # pseudo_labels_f = compute_distances(features_f, node.prototypes)
                loss_ce_f = 9*KL_Loss(LogSoftmax(features_f)/3, Softmax(node.prototypes[labels].detach())/3+ 10 ** (-7))
            else:
                node.prototypes_global = node.prototypes_global.detach()
                # pseudo_labels_f = compute_distances(features_f, node.prototypes_global)
                loss_ce_f = 9*KL_Loss(LogSoftmax(features_f)/3, Softmax(node.prototypes_global[labels].detach())/3+ 10 ** (-7))

            # pseudo_labels_f = pseudo_labels_f.detach()
            # loss_ce_f = CE_Loss(outputs_f, pseudo_labels_f)  # 使用伪标签计算交叉熵损失
            running_loss_f += loss_ce_f.item()
            # 真样本
            _, outputs = node.clser(images)
            # print(outputs)
            loss_ce_true = CE_Loss(outputs, labels.to(node.device))  # 使用logits计算交叉熵损失
            loss_ce = args.alpha * loss_ce_f + (1-args.alpha) * loss_ce_true
            loss_ce.backward()
            node.optm_cls.step()
            running_loss_t += loss_ce_true.item()
        sw.add_scalar(f'Train-cls/t_loss/{node.num}', running_loss_t / len(train_loader), round*args.cls_epochs+epo) # type: ignore
        sw.add_scalar(f'Train-cls/f_loss/{node.num}', running_loss_f / len(train_loader), round*args.cls_epochs+epo) # type: ignore
        sw.add_scalar(f'Train-cls/ce_loss/{node.num}', loss_ce.item() / len(train_loader), round*args.cls_epochs+epo) # type: ignore
        logger.info('Epoch [%d/%d], node %d: Loss_t: %.4f, Loss_f: %.4f' % (epo+1, args.cls_epochs, node.num, running_loss_t / len(train_loader), running_loss_f / len(train_loader)))
    # node.model = node.clser
    if args.save_model:
        cls_save_path = os.path.join(args.save_path+'/save/model/', str(node.num)+'_clser.pth')
        torch.save(node.clser.state_dict(), cls_save_path)
'''
class SampleGenerator(Dataset):
    def __init__(self, num_samples, latent_space, gen_model_list, classes=10, device='cuda:0'):
        self.num_samples = num_samples
        self.dataset = []
        self.device = device
        self.latent_space = latent_space
        self.classes = classes
        for gen_model in gen_model_list:
            gen_model.to(self.device)
            num_of_one_cls = int(self.num_samples / self.classes)
            for cls in range(self.classes):
                z_ = torch.randn(num_of_one_cls, self.latent_space).to(self.device)
                labels = torch.randint(0, 10, (num_of_one_cls,))
                y_ = torch.eye(self.classes)[labels].to(self.device)
                # 假样本
                fake_imgs = gen_model(z_, y_)
                for i in range(len(fake_imgs)):
                    self.dataset.append((fake_imgs[i], labels[i]))
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        fake_imgs, labels = self.dataset[idx]
        return fake_imgs, labels
'''

def train_fc(node, args, logger, round, sw=None):
    # 训练分类器
    node.cl_model.train()
    train_loader = node.train_data
    # 假设有测试数据生成器
    # train_dataset = SampleGenerator(4000, args.latent_space, [node.gen_model], 10, args.device)
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    for epo in range(args.cls_epochs):
        running_loss_t, running_loss_f = 0.0, 0.0
        loss_ce = 0
        for images, labels in train_loader:
            node.optm_fc.zero_grad()
            images = images.to(node.device).detach()
            labels = labels.to(node.device)
            # z_ = torch.randn(images.size(0), args.latent_space).to(node.device)
            # y_ = torch.eye(node.args.classes)[labels].to(node.device)  # 将类别转换为one-hot编码
            # 假样本
            # fake_imgs = node.gen_model(z_, y_).detach()
            # fake_imgs = node.gen_model(images.view(images.size(0), -1), y_).detach()
            
            # fake_imgs = gen(z_)
            # features_f, outputs_f = node.clser(fake_imgs.view(images.size(0), 1, 28, 28))
            # TODO 这里是条件GAN，生成的样本已经提前设定有标签，那么下面损失函数CE_criterion直接用y_就行了
            # 是否可以把clser中的encoder解除冻结，从而在这里利用标签进行fine-tune，使得其能更准确的预测目标域的类别
            fea, embd, out = node.cl_model(images.view(images.size(0), 1, 28, 28))
            # if node.prototypes_global is None:
            #     node.prototypes = node.prototypes.detach()
            #     pseudo_labels_f = compute_distances(fea, node.prototypes)
            # else:
            #     node.prototypes_global = node.prototypes_global.detach()
            #     pseudo_labels_f = compute_distances(fea, node.prototypes_global)
            # similarity_scores_f = torch.matmul(features_f, prototypes.t())  # 计算相似度(效果很差)
            # _, pseudo_labels_f = torch.max(similarity_scores_f, dim=1)  # 选择最相似的类别作为伪标签
            # pseudo_labels_f.requires_grad_(False)
            # pseudo_labels_f = pseudo_labels_f.detach()
            # loss_ce_f = CE_Loss(out, pseudo_labels_f)  # 使用伪标签计算交叉熵损失
            # running_loss_f += loss_ce_f.item()
            # 真样本
            # _, outputs = node.clser(images)
            # print(outputs)
            loss_ce_true = CE_Loss(out, labels)  # 使用logits计算交叉熵损失
            loss_ce = loss_ce_true
            loss_ce.backward()
            node.optm_fc.step()
            running_loss_t += loss_ce_true.item()
        sw.add_scalar(f'Train-cls/t_loss/{node.num}', running_loss_t / len(train_loader), round*args.cls_epochs+epo) # type: ignore
        sw.add_scalar(f'Train-cls/f_loss/{node.num}', running_loss_f / len(train_loader), round*args.cls_epochs+epo) # type: ignore
        sw.add_scalar(f'Train-cls/ce_loss/{node.num}', loss_ce.item() / len(train_loader), round*args.cls_epochs+epo) # type: ignore
        logger.info('Epoch [%d/%d], node %d: Loss_t: %.4f, Loss_f: %.4f' % (epo+1, args.cls_epochs, node.num, running_loss_t / len(train_loader), running_loss_f / len(train_loader)))
    # node.model = node.clser
    if args.save_model:
        cls_save_path = os.path.join(args.save_path+'/save/model/', str(node.num)+'_clser.pth')
        torch.save(node.clser.state_dict(), cls_save_path)