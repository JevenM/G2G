import torch
import h5py
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision.transforms.functional import rotate
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import datasets, transforms
import random
from collections import defaultdict


class Data(object):
    def __init__(self, args, logger):
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
                self.train_loader, self.test_loader,self.target_loader, self.classes, self.class_to_idx = get_rmnist_loaders(args,client,logger) 

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
                self.train_loader, self.test_loader,self.target_loader = get_pacs_loaders(args,client) 


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
                self.train_loader, self.test_loader, self.target_loader, self.classes, self.class_to_idx = get_vlcs_loaders(args, client, logger)
                                                               
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

        # 定义数据变换
        transform = transforms.Compose([
            # transforms.RandomRotation(degrees=(0, 90)),  # 随机旋转-30到30度
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.1307,), (0.3081,))  # 归一化
        ])
        original_dataset_tr = datasets.MNIST(root, train=True, download=False, transform=transform)
        original_dataset_te = datasets.MNIST(root, train=False, download=False, transform=transform)

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
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,))),
                                            #    resample=Image.BICUBIC)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class Loader_dataset(data.Dataset):
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
    data = eval("RotatedMNIST")(root="/data/mwj/data/", environments=client)
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
    target_data = data.datasets[target_domain_idx] #30度
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
        valid_loaders[i] = DataLoader(valid_datas[i], args.batch_size, True, num_workers=args.workers,pin_memory=args.pin)
    target_path = path_root + client[3] + '/val'
    logger.info(f"unseen domain: {client[3]}")
    target_data = Loader_dataset(target_path, trans1)
    target_loader = DataLoader(target_data, args.batch_size, True, num_workers=args.workers,pin_memory=args.pin)
    # logger.info(f'client list: {client}')
    return train_loaders, valid_loaders, target_loader, classes_name, class_to_idx

class Loader_dataset_pacs(data.Dataset):
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

def get_pacs_loaders(args, client = ['cartoon', 'sketch',  'art_painting','photo']):
    path_root = 'datasets/PACS/'
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
    target_data = Loader_dataset_pacs(target_path, trans1)
    target_loader = DataLoader(target_data, args.batch_size, True, num_workers=args.workers, pin_memory=args.pin)
    return train_loaders, valid_loaders, target_loader


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