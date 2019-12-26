from torchvision import transforms, utils

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None,normalize=False,mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5],img_trans=transforms.Compose([])):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.normalize = normalize
        self.norm = transforms.Compose([transforms.Normalize(mean=mean,std=std)])
        self.img_trans = img_trans

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        if self.transform:
            #joint transformations        
            x,y = self.transform(x,y)
            #individual transformations on the image
            x = self.img_trans(x)
        if self.normalize:
            x= self.norm(x)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
