from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
from torchvision.utils import save_image
from torchvision.models.feature_extraction import create_feature_extractor
import os
import torch.distributions as distributions
import copy
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from simsiam import D
from utils import AverageMeter, Norm_, to_img, compute_distances, save_img
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
    cnt = 0
    description = "Node{:d}: loss={:.4f} acc={:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.meme_optimizer.zero_grad()
            epochs.set_description(description.format(node.num, avg_loss, acc))
            data, target = data.to(node.device), target.to(node.device)
            _,output = node.meme(data)
            loss = CE_Loss(output, target)
            loss.backward()
            node.meme_optimizer.step()
            total_loss += loss.item()
            cnt += 1
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
    avg_loss = total_loss / cnt
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
            _,output_local = node.model(data)
            _,output_meme = node.meme(data)

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
        elif args.algorithm == 'fed_adg':
            self.train = train_fedadg
        elif args.algorithm == 'fed_sr':
            self.train = train_fedsr

    def __call__(self, node, args, logger, round=0, sw=None, epo=None):
        self.train(node, args, logger, round, sw, epo) # type: ignore

# =========================================== my ==================================================


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

def train_fedsr(node, args, logger, round, sw = None, epo=None):
    node.model.train()
    node.model.to(node.device)
    lossMeter = AverageMeter()
    accMeter = AverageMeter()
    regL2RMeter = AverageMeter()
    regCMIMeter = AverageMeter()
    regNegEntMeter = AverageMeter()

    for _, (x, y) in enumerate(tqdm(node.train_data)):
    # for step in range(node.args.E):
        # x, y = next(iter(loader))
        x, y = x.to(node.device), y.to(node.device)
        z, (z_mu,z_sigma) = node.model.featurize(x,return_dist=True)
        logits = node.model.cls(z)
        loss = F.cross_entropy(logits,y)

        obj = loss
        regL2R = torch.zeros_like(obj)
        regCMI = torch.zeros_like(obj)
        regNegEnt = torch.zeros_like(obj)
        
        regL2R = z.norm(dim=1).mean()
        obj = obj + 0.01*regL2R

        r_sigma_softplus = F.softplus(node.r_sigma)
        r_mu = node.r_mu[y]
        r_sigma = r_sigma_softplus[y]
        z_mu_scaled = z_mu*node.C
        z_sigma_scaled = z_sigma*node.C
        regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
                (z_sigma_scaled**2+(z_mu_scaled-r_mu)**2)/(2*r_sigma**2) - 0.5
        regCMI = regCMI.sum(1).mean()
        obj = obj + 0.001*regCMI

        z_dist = distributions.Independent(distributions.normal.Normal(z_mu,z_sigma),1)
        mix_coeff = distributions.categorical.Categorical(x.new_ones(x.shape[0]))
        mixture = distributions.mixture_same_family.MixtureSameFamily(mix_coeff,z_dist)
        log_prob = mixture.log_prob(z)
        regNegEnt = log_prob.mean()


        node.optimizer.zero_grad()
        obj.backward()
        node.optimizer.step()

        acc = (logits.argmax(1)==y).float().mean()
        lossMeter.update(loss.data,x.shape[0])
        accMeter.update(acc.data,x.shape[0])
        regL2RMeter.update(regL2R.data,x.shape[0])
        regCMIMeter.update(regCMI.data,x.shape[0])
        regNegEntMeter.update(regNegEnt.data,x.shape[0])
    sw.add_scalar(f'Train-sr/acc/{node.num}', accMeter.average(), round*args.E+epo) # type: ignore
    sw.add_scalar(f'Train-sr/loss/{node.num}', lossMeter.average(), round*args.E+epo) # type: ignore
    sw.add_scalar(f'Train-sr/l2r/{node.num}', regL2RMeter.average(), round*args.E+epo) # type: ignore
    sw.add_scalar(f'Train-sr/cmi/{node.num}', regCMIMeter.average(), round*args.E+epo) # type: ignore
    sw.add_scalar(f'Train-sr/neg/{node.num}', regNegEntMeter.average(), round*args.E+epo) # type: ignore
    logger.info(f'train acc: {accMeter.average()}, loss: {lossMeter.average()}, regL2R: {regL2RMeter.average()}, regCMI: {regCMIMeter.average()}, regNegEnt: {regNegEntMeter.average()}')

# 来自FedSR
def train_fedadg(node, args, logger, round, sw = None, epo=None):
    node.model.train()
    node.model.to(node.device)
    lossMeter = AverageMeter()
    accMeter = AverageMeter()
    DlossMeter = AverageMeter()
    DaccMeter = AverageMeter()
    for _, (x, y) in enumerate(tqdm(node.train_data)):
    # for step in range(steps):
        # x, y = next(iter(loader))
        x, y = x.to(node.device), y.to(node.device)
        z, logits = node.model(x)
        loss = F.cross_entropy(logits, y)

        noise = torch.rand([x.shape[0], 10]).to(node.device)
        z_fake = node.G(noise)

        D_inp = torch.cat([z_fake,z])
        D_target = torch.cat([torch.zeros([x.shape[0],1]),torch.ones([x.shape[0],1])]).to(node.device)
        
        # Train D
        D_out = node.D(D_inp.detach())
        D_loss = ((D_target-D_out)**2).mean()

        node.D_optim.zero_grad()
        D_loss.backward()
        node.D_optim.step()

        # Train Net
        D_out = node.D(D_inp)
        D_loss_g = -((D_target-D_out)**2).mean()
        obj = loss + D_loss_g

        node.optimizer.zero_grad()
        obj.backward()
        node.optimizer.step()


        acc = (logits.argmax(1)==y).float().mean()
        D_acc = ((D_out>0.5).long() == D_target).float().mean()
        lossMeter.update(loss.data,x.shape[0])
        accMeter.update(acc.data,x.shape[0])
        DlossMeter.update(D_loss.data,x.shape[0])
        DaccMeter.update(D_acc.data,x.shape[0])

    sw.add_scalar(f'Train-adg/acc/{node.num}', accMeter.average(), round*args.E+epo) # type: ignore
    sw.add_scalar(f'Train-adg/loss/{node.num}', lossMeter.average(), round*args.E+epo) # type: ignore
    sw.add_scalar(f'Train-adg/dacc/{node.num}', DaccMeter.average(), round*args.E+epo) # type: ignore
    sw.add_scalar(f'Train-adg/dloss/{node.num}', DlossMeter.average(), round*args.E+epo) # type: ignore
    logger.info(f'acc: {accMeter.average()}, loss: {lossMeter.average()}, Dacc: {DaccMeter.average()}, Dloss: {DlossMeter.average()}')

def train_adv(node, args, logger, round, sw = None, epo=None):
    node.gen_model.to(node.device).train()
    node.model.to(node.device).eval()
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
            f_r, _ = node.model(real_images)
            real_outputs = node.disc_model(f_r, y)
            loss_real = criterion_BCE(real_outputs, real_labels)
            real_scores = nn.Sigmoid()(real_outputs)  # 得到真实图片的判别值，输出的值越接近1越好
            # TODO Generator里面只用self.gen(x)的时候
            # fake_images = gen(z).to(device)
            z = torch.randn(batchsize, args.latent_space).to(node.device)
            # fake_images = node.gen_model(real_images.view(batchsize, -1), y).detach()
            fake_images = node.gen_model(z, y).detach()
            # print(fake_images[0])
            fake_outputs = node.disc_model(fake_images, y)
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
                data_iter = iter(node.target_loader)
                target_image, tar_y = next(data_iter)
                target_image = target_image.to(node.device)
                bs_t = target_image.size(0)
            true_labels_t = torch.ones(bs_t, 1).to(node.device)
            f_t, _ = node.model(target_image)
            tar_outputs = node.disc_model2(f_t)
            loss_real_t = criterion_BCE(tar_outputs, true_labels_t)
            z = torch.randn(bs_t, args.latent_space).to(node.device)
            if bs_t >= batchsize:
                fake_images = node.gen_model(z[:batchsize], y).detach()
                # fake_images = node.gen_model(target_image.view(bs_t, -1)[:batchsize], y).detach()
                fake_labels_t = torch.zeros(batchsize, 1).to(node.device)
            elif bs_t < batchsize:
                fake_images = node.gen_model(z, y[:bs_t]).detach()
                # fake_images = node.gen_model(target_image.view(bs_t, -1), y[:bs_t]).detach()
                fake_labels_t = torch.zeros(bs_t, 1).to(node.device).detach()
            # print(z.shape)
            # print(y[:bs_t])
            
            fake_outputs_t = node.disc_model2(fake_images)
            loss_fake_t = criterion_BCE(fake_outputs_t, fake_labels_t)
            loss_disc2 = loss_real_t + loss_fake_t
            loss_disc2.backward()
            node.optm_disc2.step()
        d_loss_1 += loss_disc2.item()    
        labels_cuda = labels.to(node.device)
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
            out = node.model.cls(gen_images)
            loss_cls = CE_Loss(out, labels_cuda)
            # loss_gen = loss_gen + loss
            # loss_gen = torch.log(1.0 - (discriminator(gen_images)).detach()) 
            loss_g = loss_gen1 + loss_gen + loss_cls
            # loss_g = loss_gen1 + loss_gen
            loss_g.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(node.gen_model.parameters(), max_norm=1.0)
            # 梯度值裁剪
            # torch.nn.utils.clip_grad_value_(node.gen_model.parameters(), clip_value=0.5)
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
        # if iter_==len(train_loader)-1:
        #     fake_images = to_img(fake_images.cuda().data, args.dataset)
        #     save_image(fake_images, os.path.join(args.save_path+"/gen_images/", '{}_fake_images-{}.png'.format(str(node.num), round + 1)))
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
        save_path = os.path.join(args.save_path+'/save/model/', str(node.num)+args.global_model+'_model.pth')
        # gen_save_path = os.path.join(args.save_path+'/save/model/', str(node.num)+'_gen.pth')

        torch.save(node.model.state_dict(), save_path)
        # torch.save(node.gen_model.state_dict(), gen_save_path)

    node.model.eval()
    model = node.model.cpu()
    # 训练完encoder之后
    class_counts = torch.zeros(args.classes)
    dim = node.model.out_dim
    with torch.no_grad():
        prototypes = torch.zeros(args.classes, dim)
        # 遍历整个数据集
        for images, labels in node.train_data:
            # images = images.to(node.device)
            feature, out = model(images)
            # feature = feature.cpu()
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
    margin = 1
    dist = F.pairwise_distance(x, y)
    class_eq = class_eq.to(device)
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()

def ssl_loss(x, y, class_eq, device):
    margin = 1
    dist = F.pairwise_distance(x, y)
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()


def info_nce_loss(query, key, temperature=0.5):
    """
    Compute InfoNCE loss given query and key tensors.
    
    Args:
    - query: Tensor of shape [batch_size, embedding_dim]
    - key: Tensor of shape [batch_size, embedding_dim]
    - temperature: A scalar temperature value
    
    Returns:
    - loss: A scalar loss value
    """
    batch_size = query.shape[0]
    
    # Normalize the query and key vectors
    # query = F.normalize(query, dim=1)
    # key = F.normalize(key, dim=1)
    
    # Compute similarity scores
    similarity_matrix = torch.mm(query, key.t()) / temperature
    
    # Apply exp to the similarity scores
    exp_sim = torch.exp(similarity_matrix)
    
    # Calculate the log of the sum of exp similarity scores for each row
    log_sum_exp_sim = torch.log(exp_sim.sum(dim=1))
    
    # Calculate the similarity of positive examples (diagonal elements)
    positive_sim = torch.diag(similarity_matrix)
    
    # Calculate the negative log-likelihood
    nll_loss = -positive_sim + log_sum_exp_sim
    
    # Take the mean of the negative log-likelihood
    loss = nll_loss.mean()
    
    return loss


def train_ssl(node, args, logger, round, sw=None):
    node.model.to(node.device)
    node.model.train()
    train_loader = node.train_data
    dim = node.model.out_dim
    # 训练编码器
    for k in trange(args.simclr_e):
        ls = 0
        running_loss_ce_t, running_ls_norm, running_ls_norm1, running_loss_ssl, running_loss_ce_f = 0.0, 0.0, 0.0, 0.0, 0.0
        running_loss_ssl2, loss = 0.0, 0.0
        for iter_, (real_images, labels) in enumerate(train_loader):
            batchsize = real_images.size(0)
            real_images = real_images.to(node.device)
            labels = labels.to(node.device)
            node.optimizer.zero_grad()
            
            feature1, out_r = node.model(real_images)
            feature1 = feature1.cpu()
            ls_,ls_2,ou_loss_norm = 0.0, 0.0, 0.0
            ou_loss_norm1, loss_ce_f = 0.0, 0.0
            if round > args.warm:#args.R / 2:
                z = torch.randn(batchsize, args.latent_space).to(node.device)
                # lab_ = torch.randint(0, 10, (batchsize,))
                num_same = batchsize // 4
                num_random = batchsize - num_same
                lab_same = labels.cpu()[torch.randperm(batchsize)[:num_same]]
                # 生成剩余 num_random 个随机元素
                lab_random = torch.randint(0, args.classes, (num_random,))
                # 将两部分拼接起来
                lab_ = torch.cat((lab_same, lab_random))
                # 打乱顺序
                lab_ = lab_[torch.randperm(batchsize)]
                # print(lab_, labels)
                y = torch.eye(args.classes)[lab_].to(node.device)
                # print(lab_[:10])
                # y = torch.eye(args.classes)[labels].to(node.device)  # 将类别转换为one-hot编码
                fake_images_f = node.gen_model(z, y).detach()
                # fake_images = node.gen_model(real_images.view(batchsize, -1), y).detach()
                # fake_images = gen(z)
                # print(real_images.size())
                # print(fake_images.size())
                cc1 = torch.zeros(args.classes)
                proto1 = torch.zeros(args.classes, dim)
                cc2 = torch.zeros(args.classes)
                proto2 = torch.zeros(args.classes, dim)
                # 遍历z1对应的label
                for i in range(args.classes):  # 遍历每个类别
                    class_ind = (labels == i)  # 找到属于当前类别的样本的索引
                    class_outputs = feature1[class_ind.cpu()]  # 提取属于当前类别的样本的特征向量
                    # class_outputs = embeddings_orig[class_ind]  # 提取属于当前类别的样本的特征向量
                    proto1[i] += torch.sum(class_outputs, dim=0)  # 将当前类别的特征向量累加到原型矩阵中
                    cc1[i] += class_outputs.shape[0]  # 统计当前类别的样本数量
                    class_ind2 = (lab_ == i)  # 找到属于当前类别的样本的索引
                    fake_images_f_cpu = fake_images_f.cpu()
                    class_outputs2 = fake_images_f_cpu[class_ind2.cpu()]  # 提取属于当前类别的样本的特征向量
                    # class_outputs2 = embeddings_gen[class_ind2]  # 提取属于当前类别的样本的特征向量
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
                    ou_loss_norm = Norm_(proto1, node.prototypes.detach())
                    ou_loss_norm1 = Norm_(proto2, node.prototypes.detach())
                else:
                    ou_loss_norm = Norm_(proto1, node.prototypes_global.detach())
                    ou_loss_norm1 = Norm_(proto2, node.prototypes_global.detach())
            
                # pseudo_labels_f = pseudo_labels_f.detach()
                # loss_ce_f = CE_Loss(out_f, pseudo_labels_f)  # 使用伪标签计算交叉熵损失

                running_ls_norm += ou_loss_norm.item()
                running_ls_norm1 += ou_loss_norm1.item()

                # if args.method == 'simclr':
                #     ls_ = NT_XentLoss(embeddings_orig, embeddings_gen)
                # elif args.method == 'simsiam':
                #     ls_ = D(embeddings_orig, feature2) / 2 + D(embeddings_gen, feature1) / 2
                # elif args.method == 'ccsa':
                #     lab_ = lab_.to(node.device)
                #     ls_ = csa_loss(embeddings_orig, embeddings_gen, (labels == lab_).float(), args.device)
                if args.method == 'ssl':
                    lab_ = lab_.to(node.device)
                    # 1. 将张量 A 中的每一行重复 10 次，得到 A'
                    A_prime = proto2.repeat_interleave(args.classes, dim=0)
                    if node.prototypes_global is None:
                        B_prime = node.prototypes.repeat(args.classes, 1)
                    else:
                        # 2. 将张量 B 进行 10 次堆叠，得到 B'
                        B_prime = node.prototypes_global.repeat(args.classes, 1)

                    # 3. 创建索引张量 C，表示 A' 中每一行在原始张量 A 中的索引号
                    C = torch.arange(args.classes).repeat_interleave(args.classes)

                    # 4. 创建索引张量 D，表示 B' 中每一行在原始张量 B 中的索引号
                    D = torch.arange(args.classes).repeat(args.classes)
                    ls_ = ssl_loss(A_prime, B_prime, (C == D).float(), args.device)
                    running_loss_ssl += ls_.item()
                # elif args.method == 'infonce':
                    # if node.prototypes_global is None:
                    #     ls_ = info_nce_loss(proto2, node.prototypes.detach())
                    # else:
                    #     ls_ = info_nce_loss(proto2, node.prototypes_global.detach())
                if node.prototypes_global is None:
                    ls_2 = info_nce_loss(proto2, node.prototypes)
                else:
                    ls_2 = info_nce_loss(proto2, node.prototypes_global)
                
                running_loss_ssl2 += ls_2.item()

                out_f = node.model.cls(fake_images_f)
                loss_ce_f = CE_Loss(out_f, lab_.to(node.device))
            
            loss_ce_true = CE_Loss(out_r, labels)  # 使用logits计算交叉熵损失
            running_loss_ce_t += loss_ce_true.item()

            
            
            # running_loss_ce_f += loss_ce_f.item()

            # loss = ou_loss_f + loss_ce_f + loss_ce_true + ls_
            # loss = ou_loss_norm + ou_loss_norm1 + loss_ce_true + ls_
            # loss = loss_ce_true + loss_ce_f + ou_loss_norm + ou_loss_norm1 + ls_ + ls_2
            loss = loss_ce_true + loss_ce_f + ou_loss_norm + ls_ + ls_2 + ou_loss_norm1
            # loss = 0.1*loss_ce_true + 5*ls_ + 5*ls_2 + ou_loss_norm1
        
            loss.backward()
            nn.utils.clip_grad_norm_(node.model.parameters(), max_norm=1.0)
            node.optimizer.step()
            
            ls += loss.item()

        sw.add_scalar(f'Train-ssl/ce_t_loss/{node.num}', running_loss_ce_t / len(train_loader), round*args.simclr_e+k) # type: ignore
        sw.add_scalar(f'Train-ssl/sim_loss/{node.num}', running_ls_norm / len(train_loader), round*args.simclr_e+k) # type: ignore
        sw.add_scalar(f'Train-ssl/sim_loss1/{node.num}', running_ls_norm1 / len(train_loader), round*args.simclr_e+k) # type: ignore
        sw.add_scalar(f'Train-ssl/ce_f_loss/{node.num}', running_loss_ce_f / len(train_loader), round*args.simclr_e+k) # type: ignore
        sw.add_scalar(f'Train-ssl/ssl_ls/{node.num}', running_loss_ssl / len(train_loader), round*args.simclr_e+k) # type: ignore
        sw.add_scalar(f'Train-ssl/ssl2_ls/{node.num}', running_loss_ssl2 / len(train_loader), round*args.simclr_e+k) # type: ignore
        sw.add_scalar(f'Train-ssl/loss/{node.num}', ls / len(train_loader), round*args.simclr_e+k) # type: ignore
        logger.info('Epoch [%d/%d], node %d: Loss_ce_t: %.4f, Loss_sim: %.4f, Loss_sim1: %.4f, Loss_ce_f: %.4f, Loss_ssl: %.4f, Loss_ssl: %.4f, total loss: %.4f' % (k+1, args.simclr_e, node.num, running_loss_ce_t / len(train_loader), running_ls_norm / len(train_loader), running_ls_norm1 / len(train_loader), running_loss_ce_f / len(train_loader), running_loss_ssl / len(train_loader), running_loss_ssl2 / len(train_loader), ls/len(train_loader))) # type: ignore

        # # 更新学习率
        # node.scheduler.step()
        # # 打印当前学习率
        # current_lr = node.optimizer.param_groups[0]['lr']
        # logger.info(f'Epoch {round*args.simclr_e+k}/{args.R*args.simclr_e}, Learning Rate: {current_lr}')

    node.model.eval()
    model = node.model.cpu()
    # 训练完encoder之后
    class_counts = torch.zeros(args.classes)
    with torch.no_grad():
        prototypes = torch.zeros(args.classes, dim)
        # 遍历整个数据集
        for images, labels in node.train_data:
            # images = images.to(node.device)
            feature, _ = model(images)
            # feature = feature.cpu()
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

def evaluate_model(device, model, test_loader, logger):
    model.eval()
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    logger.info(f"Test Loss: {avg_loss} Test Accuracy: {accuracy}")
    return accuracy, avg_loss

def train_ce(node, args, logger, round, sw=None):
    node.model.to(node.device)
    node.model.train()
    train_loader = node.train_data

    class_counts = torch.zeros(args.classes)
    dim = node.model.out_dim
    prototypes = torch.zeros(args.classes, dim)

    for k in trange(args.ce_epochs):
        running_loss_ce_t = 0.0
        total_correct = 0
        total_samples = 0
        accuracy_train = 0
        for iter_, (real_images, labels) in enumerate(train_loader):
            real_images = real_images.to(node.device)
            labels = labels.to(node.device)
            node.optimizer.zero_grad()
            _, out_r = node.model(real_images)
            loss_ce_true = CE_Loss(out_r, labels)  # 使用logits计算交叉熵损失
            running_loss_ce_t += loss_ce_true.item()
            loss_ce_true.backward()
            node.optimizer.step()
            
            _, predicted = torch.max(out_r, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # 更新学习率
        # node.scheduler.step()
        # # 打印当前学习率
        # current_lr = node.optimizer.param_groups[0]['lr']
        # logger.info(f'CE Epoch {k}/{args.ce_epochs}, Learning Rate: {current_lr}')

        accuracy_train = total_correct / total_samples

        sw.add_scalar(f'Train-ce/loss/{node.num}', running_loss_ce_t / len(train_loader), round*args.ce_epochs+k) # type: ignore
        logger.info('Epoch [%d/%d], node %d: Loss_ce: %.4f, Acc: %.4f' % (k+1, args.ce_epochs, node.num, running_loss_ce_t / len(train_loader), accuracy_train)) # type: ignore
        sw.add_scalar(f'Train-ce/acc/{node.num}', accuracy_train, round*args.ce_epochs+k) # type: ignore

        acc_test, loss_test = evaluate_model(node.device, copy.deepcopy(node.model), node.test_data, logger) # type: ignore
        sw.add_scalar(f'Test-ce/acc/{node.num}', acc_test, round*args.ce_epochs+k) # type: ignore
        sw.add_scalar(f'Test-ce/loss/{node.num}', loss_test, round*args.ce_epochs+k) # type: ignore

    node.model.eval()
    model = node.model.cpu()
    with torch.no_grad():
        for images, labels in train_loader:
            # images = images.to(node.device)
            feature, _ = model(images)
            # feature = feature.cpu()
            for i in range(args.classes):  # 遍历每个类别
                class_indices = (labels == i)  # 找到属于当前类别的样本的索引
                if class_indices.sum() > 0:  # 如果当前类别存在样本
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
    # 清理缓存
    # torch.cuda.empty_cache()


def train_ssl1(node, args, logger, round, sw=None):
    node.model.to(node.device)
    node.model.train()
    train_loader = node.train_data
    # 训练编码器
    for k in trange(args.simclr_e):
        running_loss_ce_t = 0.0
        for iter_, (real_images, labels) in enumerate(train_loader):
            real_images = real_images.to(node.device)
            labels = labels.to(node.device)
            node.optimizer.zero_grad()
        
            feature1, out_r = node.model(real_images)
            
            loss_ce_true = CE_Loss(out_r, labels)  # 使用logits计算交叉熵损失
            running_loss_ce_t += loss_ce_true.item()

            loss_ce_true.backward()
            node.optimizer.step()

        sw.add_scalar(f'Train-ssl/ce_t_loss/{node.num}', running_loss_ce_t / len(train_loader), round*args.simclr_e+k) # type: ignore
        logger.info('Epoch [%d/%d], node %d: Loss_ce_t: %.4f' % (k+1, args.simclr_e, node.num, running_loss_ce_t / len(train_loader))) # type: ignore

        # # 更新学习率
        # node.scheduler.step()
        # # 打印当前学习率
        # current_lr = node.optimizer.param_groups[0]['lr']
        # logger.info(f'Epoch {round*args.simclr_e+k}/{args.R*args.simclr_e}, Learning Rate: {current_lr}')

    node.model.eval()
    model = node.model.cpu()
    # 训练完encoder之后
    class_counts = torch.zeros(args.classes)
    dim = node.model.out_dim
    prototypes = torch.zeros(args.classes, dim)
    with torch.no_grad():
        # 遍历整个数据集
        for images, labels in node.train_data:
            # images = images.to(node.device)
            feature, _ = model(images)
            # feature = feature.cpu()
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
    node.model.train()
    train_loader = node.train_data
    for epo in range(args.cls_epochs):
        running_loss_t, running_loss_f = 0.0, 0.0
        loss_ce = 0
        for images, labels in train_loader:
            node.optm_cls.zero_grad()
            images = images.to(node.device)
            z_ = torch.randn(images.size(0), args.latent_space).to(node.device)
            y_ = torch.eye(node.args.classes)[labels].to(node.device)  # 将类别转换为one-hot编码
            # 假样本特征向量
            fake_imgs = node.gen_model(z_, y_).detach()
            features_f = F.normalize(fake_imgs, p=2, dim=1)
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
            _, outputs = node.model(images)
            # print(outputs)
            loss_ce_true = CE_Loss(outputs, labels.to(node.device))  # 使用logits计算交叉熵损失
            loss_ce = args.alpha * loss_ce_f + (1-args.alpha) * loss_ce_true
            loss_ce.backward()
            node.optm_cls.step()
            running_loss_t += loss_ce_true.item()
        sw.add_scalar(f'Train-cls/t_loss/{node.num}', running_loss_t / len(train_loader), round*args.cls_epochs+epo) # type: ignore
        sw.add_scalar(f'Train-cls/f_loss/{node.num}', running_loss_f / len(train_loader), round*args.cls_epochs+epo) # type: ignore
        # sw.add_scalar(f'Train-cls/ce_loss/{node.num}', loss_ce.item() / len(train_loader), round*args.cls_epochs+epo) # type: ignore
        logger.info('Epoch [%d/%d], node %d: Loss_t: %.4f, Loss_f: %.4f' % (epo+1, args.cls_epochs, node.num, running_loss_t / len(train_loader), running_loss_f / len(train_loader)))
    # node.model = node.clser
    # if args.save_model:
    #     cls_save_path = os.path.join(args.save_path+'/save/model/', str(node.num)+'_clser.pth')
    #     torch.save(node.clser.state_dict(), cls_save_path)
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
    node.model.train()
    train_loader = node.train_data
    # 假设有测试数据生成器
    # train_dataset = SampleGenerator(4000, args.latent_space, [node.gen_model], 10, args.device)
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    for epo in range(args.cls_epochs):
        running_loss_t, running_loss_f = 0.0, 0.0
        loss_ce = 0
        for images, labels in train_loader:
            node.optm_fc.zero_grad()
            images = images.to(node.device)
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
            fea, out = node.model(images)
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
    # if args.save_model:
    #     cls_save_path = os.path.join(args.save_path+'/save/model/', str(node.num)+'_clser.pth')
    #     torch.save(node.clser.state_dict(), cls_save_path)