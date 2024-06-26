import torch
import h5py
import os
import numpy as np
from PIL.Image import BICUBIC
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset, Dataset, Subset, ConcatDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.abspath("/data/mwj/data")

class Data(object):
    def __init__(self, args, logger, ):
        self.args = args
        iteration = args.iteration
        client = None
        if args.dataset == 'rotatedmnist':
            self.trainset, self.testset = None, None
            if iteration == 0:
                print(iteration, 0000000000000)
                client = [0, 15, 30, 45, 60, 75]
            if iteration == 1:
                client = [15, 30, 45, 60, 75, 0]
            if iteration == 2:
                client = [30, 45, 60, 75, 0, 15]
            if iteration == 3:
                print(iteration, 333333333)
                client = [45, 60, 75, 0, 15, 30]
            if iteration == 4:
                client = [60, 75, 0, 15, 30, 45]
            if iteration == 5:
                client = [75, 0, 15, 30, 45, 60]
            if client is not None:
                self.train_loader, self.test_loader,self.target_loader, self.classes, self.class_to_idx = get_rmnist_loaders(args, client, logger) 

        if args.dataset == 'pacs':
            self.trainset, self.testset = None, None
            if iteration == 0:
                client = ['cartoon', 'sketch',  'art_painting','photo']
            if iteration == 1:
                client=[ 'photo','cartoon','sketch','art_painting']
            if iteration == 2:
                client=[ 'sketch','photo','art_painting','cartoon']
            if iteration == 3:
                client=[ 'photo','cartoon','art_painting','sketch']
            if client is not None:
                if args.iid == 0:
                    self.train_loader, self.test_loader, self.target_loader, self.classes, self.class_to_idx = get_pacs_loaders(args, client, logger) 
                elif args.iid == 1:
                    self.train_loader, self.test_loader, self.target_loader, self.classes, self.class_to_idx = partition_data(args, client, "iid", 0.1, logger)

        if args.dataset == 'vlcs':
            if iteration == 0:
                client=['SUN09', 'Caltech101','LabelMe','VOC2007']
            if iteration == 1:
                client=['Caltech101','VOC2007','SUN09', 'LabelMe']
            if iteration == 2:
                client=['SUN09', 'LabelMe','VOC2007','Caltech101']
            if iteration == 3:
                client=['LabelMe','Caltech101','VOC2007','SUN09']
            if client is not None:
                if args.iid == 0:
                    self.train_loader, self.test_loader, self.target_loader, self.classes, self.class_to_idx = get_vlcs_loaders(args, client, logger)
                elif args.iid == 1:
                    self.train_loader, self.test_loader, self.target_loader, self.classes, self.class_to_idx = partition_data(args, client, "iid", 0.1, logger)
                                                               
        if args.dataset == 'office-home':
            if iteration == 0:
                client=['Real World','Product','Clipart','Art']                                                                                                 
            if iteration == 1:
                client=[ 'Real World','Product','Art','Clipart']                                      
            if iteration == 2:
                client=['Real World','Art','Clipart', 'Product']                          
            if iteration == 3:
                client=['Clipart', 'Art','Product','Real World']
            if client is not None:
                self.train_loader, self.test_loader,self.target_loader = get_office_loaders(args,client)
        self.client = client
        logger.info('CLIENT_ORDER{}'.format(client))

class MultipleDomainDataset:
    EPOCHS = 100             # Default, if train with epochs, check performance every epoch.
    N_WORKERS = 4            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = datasets.MNIST(root, train=True, download=False)
        original_dataset_te = datasets.MNIST(root, train=False, download=False)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []
        self.classes = original_dataset_tr.classes
        self.class_to_idx = original_dataset_tr.class_to_idx

        # 每隔6个数据形成一个列表，将数组切片为6个子列表，每个列表根据environments[i]旋转相应角度
        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

class RotatedMNIST(MultipleEnvironmentMNIST):

    def __init__(self, root, environments):
        super(RotatedMNIST, self).__init__(root, environments,
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: x.rotate(angle, fill=(0,), resample=Image.BICUBIC)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)

class MultipleEnvironmentMNIST1000(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes, mnist_subset):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        # mnist_subset = np.random.choice(10)
        print('random select a batch of mnist_subset', mnist_subset)
        indices = np.load(os.path.join('datasets/VLCS/dataset/supervised_inds_' + str(mnist_subset) + '.npy'))

        original_dataset = datasets.MNIST(root, train=True, download=True)

        original_images = original_dataset.data

        original_labels = original_dataset.targets

        original_images = original_images[indices]
        original_labels = original_labels[indices]

        self.datasets = []
        self.classes = original_dataset.classes
        self.class_to_idx = original_dataset.class_to_idx

        for i in range(len(environments)):
            self.datasets.append(dataset_transform(original_images, original_labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

class RotatedMNIST1000(MultipleEnvironmentMNIST1000):
    def __init__(self, root, environments, mnist_subset):
        super(RotatedMNIST1000, self).__init__(root, environments,
                                           self.rotate_dataset, (1, 28, 28,), 10, mnist_subset)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: x.rotate(angle, fillcolor=(0,), resample=BICUBIC)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)

class Loader_dataset(Dataset):
    def __init__(self, path, tranforms = None):
        self.path = path
        self.dataset = datasets.ImageFolder(path, transform=tranforms)
        self.length = self.dataset.__len__()
        self.transform = tranforms
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data, label = self.dataset.__getitem__(idx)
        return data, label

'''
class UniqueLabelSampler(Sampler):
    def __init__(self, data_source, batch_size=10, num_classes=10):
        self.data_source = data_source
        self.num_classes = num_classes
        self.batch_size = batch_size
        if isinstance(data_source, Subset):
            self.labels = data_source.dataset.tensors[1][data_source.indices].numpy()
        elif isinstance(data_source, TensorDataset):
            self.labels = data_source.tensors[1].numpy()
        else:
            self.labels = data_source.targets.numpy()
        self.indices = list(range(len(self.labels)))
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
        # for label in self.label_to_indices.keys():
        #     print(len(self.label_to_indices[label]))
        
    def __iter__(self):
        label_to_indices = {label: indices[:] for label, indices in self.label_to_indices.items()}  # Copy the dictionary
        batches = []
        print(f"len {len(self.data_source)}")
        while len(batches) * self.batch_size < len(self.data_source):
            batch = []
            used_labels = set()
            label = 0
            while len(batch) < self.batch_size:
                if not label_to_indices:
                    print("No more labels available to sample from.")
                    break
                # label = random.choice(list(label_to_indices.keys()))
                # print(f"choose label {label}")
                if label_to_indices[label]:
                    if label not in used_labels:
                        idx = label_to_indices[label].pop()
                        batch.append(idx)
                        used_labels.add(label)
                        # if not label_to_indices[label]:
                        #     del label_to_indices[label]
                else:
                    if label not in used_labels:
                        idx = random.choice(list(self.label_to_indices[label]))
                        batch.append(idx)
                        used_labels.add(label)
                    
                    # for i, l in enumerate(used_labels):
                    #     label_to_indices[l].append(batch[i])
                    # break
                label += 1
                if len(used_labels) == self.num_classes or label == self.num_classes:
                    break

            if len(batch) == self.batch_size:
                batches.append(batch)
            else:
                print("Failed to form a complete batch, retrying...")
                break
        print(len(batches) * self.batch_size)
        random.shuffle(batches)
        # print(11111111111)
        return iter([idx for batch in batches for idx in batch])
    
    def __len__(self):
        return len(self.data_source)
'''

def get_rmnist_loaders(args, client, logger):
    # 参考FedSR中所说，only 1000 images are rotated to form the domain
    data = eval("RotatedMNIST1000")(root="/data/mwj/data/", environments=client, mnist_subset=args.mnist_subset)
    train_datas, train_loaders = {}, {}
    valid_datas, valid_loaders = {}, {}
    target_domain_idx = len(client)-1
    for s_domain_idx in range(target_domain_idx):
        dataset = data.datasets[s_domain_idx]
        train_len = int(len(dataset) * 0.9)
        test_len = len(dataset) - train_len
        # print(train_len, test_len)
        train_datas[s_domain_idx], valid_datas[s_domain_idx] = random_split(dataset, [train_len,test_len], generator=torch.Generator().manual_seed(0))
        # change the transform of test split 
        if hasattr(valid_datas[s_domain_idx].dataset,'transform'):
            import copy
            valid_datas[s_domain_idx].dataset = copy.copy(valid_datas[s_domain_idx].dataset)
            valid_datas[s_domain_idx].dataset.transform = data.transform
        # DataLoader
        # sampler = UniqueLabelSampler(train_datas[s_domain_idx])
        # train_loaders[s_domain_idx] = DataLoader(train_datas[s_domain_idx], batch_size=10, num_workers=args.workers,pin_memory=args.pin, sampler=sampler)
        train_loaders[s_domain_idx] = DataLoader(train_datas[s_domain_idx], batch_size=args.batch_size, shuffle=True, num_workers=args.workers,pin_memory=args.pin)
        valid_loaders[s_domain_idx] = DataLoader(valid_datas[s_domain_idx], batch_size=args.batch_size, shuffle=False, num_workers=args.workers,pin_memory=args.pin)

    logger.info(f"unseen domain: {client[target_domain_idx]}")
    target_data = data.datasets[target_domain_idx] 
    target_loader = DataLoader(target_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,pin_memory=args.pin)
    # logger.info(f'client list: {client}')
    return train_loaders, valid_loaders, target_loader, data.classes, data.class_to_idx


def get_vlcs_loaders(args, client, logger):
    path_root = './datasets/VLCS/VLCS/'
    trans0 = transforms.Compose([transforms.RandomResizedCrop(225, scale=(0.7, 1.0)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomGrayscale(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trans1 = transforms.Compose([transforms.Resize([225, 225]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_path, valid_path = {}, {}
    train_datas, train_loaders = {}, {}
    valid_datas, valid_loaders = {}, {}
    classes_name = None
    class_to_idx = None
    for i in range(3):
        train_path[i] = path_root + client[i] + '/train'
        train_datas[i] = Loader_dataset(path=train_path[i], tranforms=trans0)
        if classes_name is None:
            classes_name = train_datas[i].classes
            class_to_idx = train_datas[i].class_to_idx
        train_loaders[i] = DataLoader(train_datas[i], args.batch_size, True, num_workers=args.workers,pin_memory=args.pin)

        valid_path[i] = path_root + client[i] + '/val'
        valid_datas[i] = Loader_dataset(path=valid_path[i], tranforms=trans1)
        valid_loaders[i] = DataLoader(valid_datas[i], args.batch_size, False, num_workers=args.workers,pin_memory=args.pin)
    target_path = path_root + client[3] + '/val'
    logger.info(f"unseen domain: {client[3]}")
    target_data = Loader_dataset(target_path, trans1)
    target_loader = DataLoader(target_data, args.batch_size, False, num_workers=args.workers,pin_memory=args.pin)
    # logger.info(f'client list: {client}')
    return train_loaders, valid_loaders, target_loader, classes_name, class_to_idx

class Loader_dataset_pacs(Dataset):
    def __init__(self, path, tranforms):
        self.path = path
        hdf = h5py.File(self.path, 'r')
        self.length = len(hdf['labels'])   # <KeysViewHDF5 ['images', 'labels']>
        self.transform = tranforms
        hdf.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        hdf = h5py.File(self.path, 'r')
        y = hdf['labels'][idx]
        data_pil = Image.fromarray(hdf['images'][idx, :, :, :].astype('uint8'), 'RGB')
        hdf.close()
        data = self.transform(data_pil)
        return data, torch.tensor(y).long().squeeze()-1

def get_pacs_loaders(args, client, logger):
    path_root = DATA_PATH+'/PACS_hdf5/'
    trans0 = transforms.Compose([transforms.RandomResizedCrop(222, scale=(0.7, 1.0)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomGrayscale(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trans1 = transforms.Compose([transforms.Resize([222, 222]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_path, valid_path = {}, {}
    train_datas, train_loaders = {}, {}
    valid_datas, valid_loaders = {}, {}
    for i in range(3):
        train_path[i] = path_root + client[i] + '_train.hdf5'
        train_datas[i] = Loader_dataset_pacs(path=train_path[i], tranforms=trans0)
        train_loaders[i] = DataLoader(train_datas[i], args.batch_size, True, num_workers=args.workers, pin_memory=args.pin)
        valid_path[i] = path_root + client[i] + '_val.hdf5'
        valid_datas[i] = Loader_dataset_pacs(path=valid_path[i], tranforms=trans1)
        valid_loaders[i] = DataLoader(valid_datas[i], args.batch_size, True, num_workers=args.workers, pin_memory=args.pin)
    target_path = path_root + client[3] + '_test.hdf5'
    logger.info(f"unseen domain: {client[3]}")
    target_data = Loader_dataset_pacs(target_path, trans1)
    target_loader = DataLoader(target_data, args.batch_size, True, num_workers=args.workers, pin_memory=args.pin)
    class_to_idx = {
                    '0 - dog': 0,
                    '1 - elephant': 1,
                    '2 - giraffe': 2,
                    '3 - guitar': 3,
                    '4 - horse': 4,
                    '5 - house': 5,
                    '6 - person': 6,
                }
    classes = {
                '0 - dog',
                '1 - elephant',
                '2 - giraffe',
                '3 - guitar',
                '4 - horse',
                '5 - house',
                '6 - person',
            }
    return train_loaders, valid_loaders, target_loader, classes, class_to_idx

class PACS(Dataset):
    '''
    https://blog.csdn.net/qq_43827595/article/details/121345640
    '''
    def __init__(self, root_path, domain, train=True, transform=None, target_transform=None):
        self.root = f"{root_path}/{domain}"
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        label_name_list = os.listdir(self.root)
        self.label = []
        self.data = []
        self.label_name_2_index = {
                    'dog': 0,
                    'elephant': 1,
                    'giraffe': 2,
                    'guitar': 3,
                    'horse': 4,
                    'house': 5,
                    'person': 6,
                }
        self.classes = {
                    '0 - dog',
                    '1- elephant',
                    '2- giraffe',
                    '3- guitar',
                    '4- horse',
                    '5- house',
                    '6- person',
                }

        if not os.path.exists(f"{root_path}/precessed"):
            os.makedirs(f"{root_path}/precessed")
        if os.path.exists(f"{root_path}/precessed/{domain}_data.pt") and os.path.exists(
                f"{root_path}/precessed/{domain}_label.pt"):
            print(f"Load {domain} data and label from cache.")
            self.data = torch.load(f"{root_path}/precessed/{domain}_data.pt")
            self.label = torch.load(f"{root_path}/precessed/{domain}_label.pt")
        else:
            print(f"Getting {domain} datasets")
            for index, label_name in enumerate(label_name_list):
                images_list = os.listdir(f"{self.root}/{label_name}")
                for img_name in images_list:
                    img = Image.open(f"{self.root}/{label_name}/{img_name}").convert('RGB')
                    img = np.array(img)
                    self.label.append(self.label_name_2_index[label_name])
                    if self.transform is not None:
                        img = self.transform(img)
                    self.data.append(img)
            self.data = torch.stack(self.data)
            self.label = torch.tensor(self.label, dtype=torch.long)
            torch.save(self.data, f"{root_path}/precessed/{domain}_data.pt")
            torch.save(self.label, f"{root_path}/precessed/{domain}_label.pt")

    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

# 自己写的，参考https://blog.csdn.net/qq_43827595/article/details/121345640
# 功能可以替代get_pacs_loaders，读取原始图像，划分训练集(0.8,0.2)
def get_pacs_domain(args, domains, logger):
    root_path=f"{DATA_PATH}/PACS"
    transform = transforms.Compose([transforms.RandomResizedCrop(222, scale=(0.7, 1.0)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomGrayscale(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trans1 = transforms.Compose([transforms.Resize([222, 222]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # 一个domain的数据获取
    # all_data = PACS(root_path, domain, transform=transform)
    # train:test=8:2
    # x_train, x_test, y_train, y_test = train_test_split(all_data.data.numpy(), all_data.label.numpy(),
    #                                                     test_size=0.20, random_state=42)

    # return x_train, y_train, x_test, y_test
    target_domain_idx = len(domains)-1
    train_datas, train_loaders = {}, {}
    valid_datas, valid_loaders = {}, {}
    for s_domain_idx in range(3):
        dataset = PACS(root_path, domains[s_domain_idx], transform=transform)
        test_len = int(len(dataset) * 0.9)
        train_datas[s_domain_idx], valid_datas[s_domain_idx] = random_split(dataset, [1-test_len,test_len], generator=torch.Generator().manual_seed(0))
        train_loaders[s_domain_idx] = DataLoader(train_datas[s_domain_idx], args.batch_size, True, num_workers=args.workers, pin_memory=args.pin)
        valid_loaders[s_domain_idx] = DataLoader(valid_datas[s_domain_idx], args.batch_size, True, num_workers=args.workers, pin_memory=args.pin)

    logger.info(f"unseen domain: {domains[target_domain_idx]}")
    target_data = PACS(root_path, domains[target_domain_idx], transform=trans1)
    target_loader = DataLoader(target_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,pin_memory=args.pin)
    return train_loaders, valid_loaders, target_loader, target_data.classes, target_data.label_name_2_index

def get_office_loaders(args, client):
    path_root = 'datasets/OfficeHome/'
    trans0 = transforms.Compose([transforms.RandomResizedCrop(225, scale=(0.7, 1.0)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomGrayscale(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trans1 = transforms.Compose([transforms.Resize([225, 225]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_path, valid_path = {}, {}
    train_datas, train_loaders = {}, {}
    valid_datas, valid_loaders = {}, {}
    for i in range(3):
        train_path[i] = path_root + client[i] + '/train'
        train_datas[i] = Loader_dataset(path=train_path[i], tranforms=trans0)
        train_loaders[i] = DataLoader(train_datas[i], args.batch_size, True, num_workers=args.workers,pin_memory=args.pin)

        valid_path[i] = path_root + client[i] + '/val'
        valid_datas[i] = Loader_dataset(path=valid_path[i], tranforms=trans1)
        valid_loaders[i] = DataLoader(valid_datas[i], args.batch_size, True, num_workers=args.workers,pin_memory=args.pin)
    target_path = path_root + client[3] + '/val'
    target_data = Loader_dataset(target_path, trans1)
    target_loader = DataLoader(target_data, args.batch_size, True, num_workers=args.workers,pin_memory=args.pin)
    return train_loaders, valid_loaders, target_loader

# ------------------------------------------start 来自 Federated Generative Learning with Foundation Models--------------------------------------
def record_net_data_stats(y_train, net_dataidx_map, logger, classes):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():  # label:sets
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)  # 去除数组中的重复数字，并进行排序之后输出。
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        for i in range(classes):
            if i in tmp.keys():
                continue
            else:
                tmp[i] = 0

        net_cls_counts[net_i] = tmp

    logger.info(f'Data statistics: {str(net_cls_counts)}')

    return net_cls_counts

def get_dataset_VLCS(domains_name=[]):
    path_root = './datasets/VLCS/VLCS/'

    trans0 = transforms.Compose([transforms.RandomResizedCrop(225, scale=(0.7, 1.0)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomGrayscale(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trans1 = transforms.Compose([transforms.Resize([225, 225]),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_path, valid_path = {}, {}
    train_datas_list = []
    valid_datas_list = []
    classes_name = None
    class_to_idx = None
    for i in range(3):
        train_path[i] = path_root + domains_name[i] + '/train'
        train_datas_list.append(Loader_dataset(path=train_path[i], tranforms=trans0))
        if classes_name is None:
            classes_name = train_datas_list[i].classes
            class_to_idx = train_datas_list[i].class_to_idx

        valid_path[i] = path_root + domains_name[i] + '/val'
        valid_datas_list.append(Loader_dataset(path=valid_path[i], tranforms=trans1))

    target_path = path_root + domains_name[3] + '/val'
    print(f"unseen domain: {domains_name[3]}")
    target_datas = Loader_dataset(target_path, trans1)

    return train_datas_list, valid_datas_list, target_datas, classes_name, class_to_idx

def get_dataset_PACS(domains_name=[]):
    path_root = DATA_PATH+'/PACS_hdf5/'
    trans0 = transforms.Compose([transforms.RandomResizedCrop(222, scale=(0.7, 1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomGrayscale(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trans1 = transforms.Compose([transforms.Resize([222, 222]),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_path, valid_path = {}, {}
    train_datas_list = []
    valid_datas_list = []

    for i in range(3):
        train_path[i] = path_root + domains_name[i] + '_train.hdf5'
        train_datas_list.append(Loader_dataset_pacs(path=train_path[i], tranforms=trans0))
        valid_path[i] = path_root + domains_name[i] + '_val.hdf5'
        valid_datas_list.append(Loader_dataset_pacs(path=valid_path[i], tranforms=trans1))

    target_path = path_root + domains_name[3] + '_test.hdf5'
    print(f"unseen domain: {domains_name[3]}")
    target_datas = Loader_dataset_pacs(target_path, trans1)

    class_to_idx = {
                    '0 - dog': 0,
                    '1 - elephant': 1,
                    '2 - giraffe': 2,
                    '3 - guitar': 3,
                    '4 - horse': 4,
                    '5 - house': 5,
                    '6 - person': 6,
                }
    classes_name = {
                '0 - dog',
                '1 - elephant',
                '2 - giraffe',
                '3 - guitar',
                '4 - horse',
                '5 - house',
                '6 - person',
            }

    return train_datas_list, valid_datas_list, target_datas, classes_name, class_to_idx

def get_dataset(data_type='vlcs', domains=['SUN09', 'Caltech101', 'LabelMe', 'VOC2007']):
    classes_name, class_to_idx = None, None
    
    # get real VLCS dataset
    if data_type == "vlcs":
        trainset_list, valset_list, target_dataset, classes_name, class_to_idx = get_dataset_VLCS(domains_name=domains)
    # get real PACS dataset
    elif data_type == "pacs":
        trainset_list, valset_list, target_dataset, classes_name, class_to_idx = get_dataset_PACS(domains_name=domains)
    train_dataset = ConcatDataset(trainset_list)
    val_dataset = ConcatDataset(valset_list)

    return train_dataset, val_dataset, target_dataset, classes_name, class_to_idx

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # return image, label
        return torch.from_numpy(image).float(), torch.tensor(label)

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

# 改编来自 Federated Generative Learning with Foundation Models
def load_data(args, client):
    train_dataset, val_dataset, target_dataset, classes_name, class_to_idx = get_dataset(data_type=args.dataset, domains=client)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    testloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    X_train, y_train = [], []
    # for data in trainloader:
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        X_train.append(inputs.numpy())
        y_train.append(targets.numpy())
    
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    X_test, y_test = [], []
    # for data in trainloader:
    for batch_idx, (inputs, targets) in enumerate(testloader):
        X_test.append(inputs.numpy())
        y_test.append(targets.numpy())
    
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    return X_train, y_train, X_test, y_test, train_dataset, test_dataset, target_dataset, classes_name, class_to_idx

# 来自 Federated Generative Learning with Foundation Models
def partition_data(args, client, partition, beta=0.1, logger=None):
    _, y_train, _, y_test, train_dataset, val_dataset, target_dataset, classes_name, class_to_idx = load_data(args, client)
    data_size = y_train.shape[0]
    val_data_size = y_test.shape[0]

    n_parties = args.node_num
    if partition == "iid":
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        val_idxs = np.random.permutation(val_data_size)
        val_batch_idxs = np.array_split(val_idxs, n_parties)
        val_net_dataidx_map = {i: val_batch_idxs[i] for i in range(n_parties)}

    elif partition == "dirichlet":
        min_size = 0
        min_require_size = 10
        label = np.unique(y_test).shape[0]
        net_dataidx_map = {}
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(label):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)  # shuffle the label
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(   # 0 or x
                    [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logger, args.classes)

    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_loaders, valid_loaders = {}, {}
    for idx in range(n_parties):
        train_loaders[idx] = DataLoader(DatasetSplit(train_dataset, net_dataidx_map[idx]), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        valid_loaders[idx] = DataLoader(DatasetSplit(val_dataset, val_net_dataidx_map[idx]), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
       
    return train_loaders, valid_loaders, target_loader, classes_name, class_to_idx

# ------------------------------------------end 来自 Federated Generative Learning with Foundation Models--------------------------------------