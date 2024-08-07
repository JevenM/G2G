import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Total
    parser.add_argument('--algorithm', type=str, default='fed_mutual',
                        help='Type of algorithms:{fed_mutual, fed_sr, fed_avg, normal, fed_adv}')
    parser.add_argument('--method', type=str, default='simclr',
                        help='{simclr, simsiam}')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device: {cuda, cpu}')
    parser.add_argument('--node_num', type=int, default=3,
                        help='Number of nodes')
    parser.add_argument('--R', type=int, default=50,
                        help='Number of rounds: R')
    parser.add_argument('--E', type=int, default=7,
                        help='Number of local epochs: E')
    parser.add_argument('--notes', type=str, default='',
                        help='Notes of Experiments')
    parser.add_argument('--pin', default=True, action='store_true',
                        help='pin-memory')

    # Model
    parser.add_argument('--global_model', type=str, default='ResNet50',
                        help='Type of global model: {LeNet5, MLP, SmallCNN, ResNet50,ResNet18,VGG16,Alexnet,Alexnet2}')
    parser.add_argument('--local_model', type=str, default='ResNet50',
                        help='Type of local model: {LeNet5, MLP, SmallCNN, ResNet50,ResNet18,VGG16,Alexnet,Alexnet2}')
    # parser.add_argument('--catfish', type=str, default=None,
    #                     help='Type of local model: {None, LeNet5, MLP, CNN2, ResNet50}')
    parser.add_argument('--discr_e', type=int, default=1,
                        help='')
    parser.add_argument('--disc_lr', type=float, default=0.0002,
                        help='')
    
    parser.add_argument('--gen_e', type=int, default=3,
                        help='')
    parser.add_argument('--gen_lr', type=float, default=0.0002,
                        help='')
    parser.add_argument('--mnist_subset', type=int, default=0,
                        help='')
    
    
    parser.add_argument('--simclr_e', type=int, default=10,
                        help='')
    
    parser.add_argument('--ce_epochs', type=int, default=30,
                        help='')
    parser.add_argument('--cls_epochs', type=int, default=150,
                        help='')
    parser.add_argument('--cls_lr', type=float, default=0.0001,
                        help='')
    parser.add_argument('--embedding_d', type=int, default=512,
                        help='')
    parser.add_argument('--latent_space', type=int, default=64,
                        help='')
    parser.add_argument('--save_model', default=False, action='store_true',
                        help='')
    parser.add_argument('--warm', type=int, default=10, 
                        help='number of epochs to want before ssl (10)')
    parser.add_argument('--server_e', type=int, default=10,
                        help='')
    
    # Data
    parser.add_argument('--dataset', type=str, default='vlcs',
                        help='datasets: {air_ori, air, pacs, vlcs, cifar100, cifar10, femnist, office-home, mnist, rotatedmnist}')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--iid', type=int, default=0,
                        help='data iid distribution')
    parser.add_argument('--split', type=int, default=5,
                        help='data split')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='val_ratio')
    parser.add_argument('--all_data', default=False, action='store_true',
                        help='use all train_set')
    parser.add_argument('--classes', type=int, default=7,
                        help='classes')

    # Optima
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer: {sgd, adam}')
    # cl_lr
    parser.add_argument('--lr', type=float, default=0.0008,
                        help='learning rate')
    # parser.add_argument('--lr_step', type=int, default=10,
    #                     help='learning rate decay step size')
    parser.add_argument('--stop_decay', type=int, default=50,
                        help='round when learning rate stop decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='local ratio of data loss')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='meme ratio of data loss')
    parser.add_argument('--workers', type=int, default=16,
                        help='num_workers')                    
    parser.add_argument('--pretrained', default=True, action='store_true')
    parser.add_argument('--factor', type=float, default=0.1, 
                        help='lr decreased factor (0.1)')
    parser.add_argument('--patience', type=int, default=20, 
                        help='number of epochs to want before reduce lr (20)')
    parser.add_argument('--lr-threshold', type=float, default=1e-4, 
                        help='lr schedular threshold')
    parser.add_argument('--ite-warmup', type=int, default=100, 
                        help='LR warm-up iterations (default:500)')
    # parser.add_argument('--label_smoothing', type=float, default=0.1, 
    #                     help='the rate of wrong label(default:0.2)')
    parser.add_argument('--save_path', type=str, default='.',
                        help='path for this run.')


    # for ALexnet2
    parser.add_argument('--lr0', type=float, default=0.0001, help='learning rate 0')
    parser.add_argument('--lr1', type=float, default=0.0007, help='learning rate 1')
    parser.add_argument('--weight-dec', type=float, default=1e-7, help='0.005 weight decay coefficient default 1e-5')
    parser.add_argument('--rp-size', type=int, default=1024, help='Random Projection size 1024')
    parser.add_argument('--hidden_size', type=int, default=4096, help='the size of hidden feature')
    parser.add_argument('--iteration', type=int, default=0, help='the iteration')

    parser.add_argument('--mix', type=float, default=0)
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--warm_up',type=bool, default=True)
    parser.add_argument('--lr_scheduler',type=bool, default=True)
    args = parser.parse_args()
    return args
