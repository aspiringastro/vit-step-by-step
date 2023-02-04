from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10


class CIFAR10DataSet():
    def __init__(self, data_dir="data/cifar10"):
        self.data_dir = data_dir
        self.dataset = CIFAR10(root=self.data_dir, download=True)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def  train_dataloader(self, batch_size=32, resize=224, p=0.5, mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), num_workers=4, shuffle=True):
        tf = T.Compose([
                T.RandomResizedCrop(size=resize),
                T.RandomHorizontalFlip(p=p),
                T.RandomVerticalFlip(p=p),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        ds = CIFAR10(root=self.data_dir, train=True, transform=tf)
        dl = DataLoader(ds,batch_size=batch_size, num_workers=num_workers,shuffle=shuffle, drop_last=True)
        return dl
    
    def test_dataloader(self, batch_size=32, mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225), num_workers=4):
        tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
            ]
        )
        ds = CIFAR10(root=self.data_dir, train=False, transform=tf)
        dl = DataLoader(ds,batch_size=batch_size, num_workers=num_workers, drop_last=True)
        return dl
    
    def get_batch(self, dataloader):
        return next(iter(dataloader))
    
    
    



