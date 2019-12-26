import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage import exposure
from torch.utils.data import DataLoader
from torchvision import transforms, utils

import os

#Network
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.optim as optim

import matplotlib.pyplot as plt
from IPython.display import clear_output

# internal utilities
from time import time
from model.utils.Evaluation import F1_score
from model.utils.data_augmentation import *
from model.utils.dataset import CustomTensorDataset
from model.model import UNet,SegNet


def train(model, opt, loss_fn, epochs, data_tr,data_val, data_testA,data_testB,device):

    criterion = nn.CrossEntropyLoss()
    X_val, Y_val = next(iter(data_val))
    X_testA, Y_testA = next(iter(data_testA))
    X_testB, Y_testB = next(iter(data_testB))
    unnorm = UnNormalize(mean,std)
    loss_hist={'train':[],'test':[]}
    F1_scores_macro = {'train_mean':[],'train_std':[],'val_mean':[],'val_std':[],'testA_mean':[],'testA_std':[],'testB_mean':[],'testB_std':[]}
    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        avg_loss_train = 0
        f1_train =[]
        model.train()  # train mode
        for X_batch, Y_batch in data_tr:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)
            loss = criterion(Y_pred,torch.argmax(Y_batch,dim=1).long())
            for idx_,y_pred in enumerate(Y_pred):
                f1_train.append(F1_score(torch.argmax(Y_batch[idx_].detach().cpu(),dim=0),torch.argmax(y_pred.detach().cpu(),dim=0)))

            # backward
            loss.backward()  # backward-pass
            opt.step()  # update weights
            # calculate metrics to show the user
            avg_loss += loss.detach().cpu().numpy()
        avg_loss = avg_loss/len(data_tr)
        toc = time()
        loss_hist['train'].append(avg_loss)
        
        f1_temp=np.array(f1_train)
        F1_scores_macro['train_mean'].append(np.mean(f1_temp))
        F1_scores_macro['train_std'].append(np.std(f1_temp))
        # show intermediate results
        
        model.eval()  # testing mode
        #calulating loss on data_val

        f1_val = []
        for X_batch_val, Y_batch_val in data_val:
            Y_pred = torch.sigmoid(model(X_batch_val.to(device))).detach().cpu()
            avg_loss_train += criterion(Y_pred,torch.argmax(Y_batch_val,dim=1).long()).detach().cpu().numpy()
            for idx_,y_pred in enumerate(Y_pred):
                f1_val.append(F1_score(torch.argmax(Y_batch_val[idx_],dim=0),torch.argmax(y_pred,dim=0)))

        avg_loss_train =avg_loss_train/len(data_val)
        loss_hist['test'].append(avg_loss_train)
        f1_temp=np.array(f1_val)
        F1_scores_macro['val_mean'].append(np.mean(f1_temp))
        F1_scores_macro['val_std'].append(np.std(f1_temp))
        #calulating F1 on testA
        f1_testA_ep_macro =[] # temp f1 scores for all images on this epoch
        
        for X_batch_val, Y_batch_val in data_testA:
            Y_pred = torch.sigmoid(model(X_batch_val.to(device))).detach().cpu()
            for idx_,y_pred in enumerate(Y_pred):

                f1_testA_ep_macro.append(F1_score(Y_batch_val[idx_],torch.argmax(y_pred,dim=0)))
        
  

        f1_temp2=np.array(f1_testA_ep_macro)
        F1_scores_macro['testA_mean'].append(np.mean(f1_temp2))
        F1_scores_macro['testA_std'].append(np.std(f1_temp2))
        
        #calculating F1 on testB
        f1_testB_ep_macro =[] # temp f1 scores for all images on this epoch
        
        for X_batch_val, Y_batch_val in data_testB:
            Y_pred = torch.sigmoid(model(X_batch_val.to(device))).detach().cpu()
            for idx_,y_pred in enumerate(Y_pred):
                f1_testB_ep_macro.append(F1_score(Y_batch_val[idx_],torch.argmax(y_pred,dim=0)))
                #f1_testB_ep_binary.append(F1_score(Y_batch_val[idx_],y_pred,avg='binary'))
        
        f1_temp2=np.array(f1_testB_ep_macro)
        
        F1_scores_macro['testB_mean'].append(np.mean(f1_temp2))
        F1_scores_macro['testB_std'].append(np.std(f1_temp2))
        
        
        
        '''
        Plotting after each epoch
        '''
        Y_hat = model(X_val.to(device)).detach().cpu()
        Y_hatA = model(X_testA.to(device)).detach().cpu()
        Y_hatB = model(X_testB.to(device)).detach().cpu()

      
        #print('Y_hat.shape',Y_hat.shape)
        clear_output(wait=True)
        plt.rcParams['figure.figsize'] = [18, 6]
        rows, cols = 3,7

        plt.subplot(1,cols,1)
        plt.plot(list(range(1,len(loss_hist['train'])+1)),loss_hist['train'],'-',color='cornflowerblue')
        plt.plot(list(range(1,len(loss_hist['test'])+1)),loss_hist['test'],'-',color='orange')

        plt.xlabel('epoch')
        plt.ylabel('loss')
        for k in range(2,4):
            plt.subplot(rows, cols, k)
            plt.imshow(np.rollaxis(unnorm(X_val[k].clone()).numpy(), 0, 3), cmap='gray')
            plt.title('Real - val')
            plt.axis('off')

            plt.subplot(rows, cols, k+7)
            plt.imshow(np.rollaxis(Y_val[k,1].numpy(), 0, 1), cmap='gray')
            plt.title('Original Mask')
            plt.axis('off')
            
            plt.subplot(rows, cols, k+14)
            plt.imshow(torch.argmax(Y_hat[k],dim=0), cmap='gray')
            plt.title('Prediction mask')
            plt.axis('off')
        for k in range(4,6):
            plt.subplot(rows, cols, k)
            plt.imshow(np.rollaxis(unnorm(X_testA[k].clone()).numpy(), 0, 3), cmap='gray')
            plt.title('Real - TestA')
            plt.axis('off')

            plt.subplot(rows, cols, k+7)
            plt.imshow(np.rollaxis(Y_testA[k,0].numpy(), 0, 1), cmap='gray')
            plt.title('Original Mask')
            plt.axis('off')

            plt.subplot(rows, cols, k+14)
            plt.imshow(torch.argmax(Y_hatA[k],dim=0), cmap='gray')
            plt.title('Prediction mask')
            plt.axis('off')
        for k in range(6,8):
            plt.subplot(rows, cols, k)
            plt.imshow(np.rollaxis(unnorm(X_testB[k].clone()).numpy(), 0, 3), cmap='gray')
            plt.title('Real - TestB')
            plt.axis('off')

            plt.subplot(rows, cols, k+7)
            plt.imshow(np.rollaxis(Y_testB[k,0].numpy(), 0, 1), cmap='gray')
            plt.title('Original Mask')
            plt.axis('off')
            
            plt.subplot(rows, cols, k+14)
            plt.imshow(torch.argmax(Y_hatB[k],dim=0), cmap='gray')
            plt.title('Prediction mask')
            plt.axis('off')
        plt.suptitle('%d / %d - loss: %f - time:%f - F1TestA:%f - F1TestB:%f' % (epoch+1, epochs, avg_loss,toc-tic,F1_scores_macro['testA_mean'][-1],F1_scores_macro['testB_mean'][-1]))
        plt.show()
    return F1_scores_macro
#Helper functions
def to_numpy(images,masks):
    X = np.array(images, np.float32)
    Y = np.array(masks, np.float32)
    print('Loaded %d images' % len(X))
    print('Loaded %d masks' % len(Y))
    return X,Y
def apply_transform(image,mask,sigma):
    trans_img = []
    trans_masks = []
    for idx,img in enumerate(image):
        #merge to apply elastic deformation
        im_merge = np.dstack((img, mask[idx]))
        im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08 * sigma, im_merge.shape[1] * 0.08 * sigma)
        #split it afterwards
        trans_img.append(im_merge_t[...,:-1])
        trans_masks.append(im_merge_t[...,-1]>0)
    return trans_img,trans_masks

if __name__ == "__main__":
    os.chdir('image root')
    #read images
    images_train,images_testA,images_testB = [],[],[]
    masks_train,masks_testA,masks_testB = [],[],[]


    read_size=300
    img_size = 256

    size_read=(read_size,read_size) # original size is 576x576 -> maybe increas it for later tests
    size=(img_size,img_size)
    for root, dirs, files in os.walk('warwick_data/'):
        for f in files:
            if not f.endswith('anno.bmp') and not f.endswith('csv') and f.startswith('train'):
                img_path = f
                images_train.append(resize(imread(os.path.join(root, img_path)), size_read, mode='constant',anti_aliasing=True))
                
                mask_path = f[:-4]+'_anno.bmp'
                masks_train.append(resize(imread(os.path.join(root, mask_path)), size_read, mode='constant',anti_aliasing=True)>0)
            elif not f.endswith('anno.bmp') and not f.endswith('csv') and f.startswith('testA'):
                img_path = f
                images_testA.append(resize(imread(os.path.join(root, img_path)), size, mode='constant',anti_aliasing=True))
                
                mask_path = f[:-4]+'_anno.bmp'
                masks_testA.append(resize(imread(os.path.join(root, mask_path)), size, mode='constant',anti_aliasing=True)>0)
            elif not f.endswith('anno.bmp') and not f.endswith('csv') and f.startswith('testB'):
                img_path = f
                images_testB.append(resize(imread(os.path.join(root, img_path)), size, mode='constant',anti_aliasing=True))
                
                mask_path = f[:-4]+'_anno.bmp'
                masks_testB.append(resize(imread(os.path.join(root, mask_path)), size, mode='constant',anti_aliasing=True)>0)
    
    #convert all the images to numpy arrays
    X_train,Y_train = to_numpy(images_train,masks_train)
    X_testA,Y_testA = to_numpy(images_testA,masks_testA)
    X_testB,Y_testB = to_numpy(images_testB,masks_testB)

    if True:
        X_deformed1,Y_deformed1 = apply_transform(X_train,Y_train,sigma=3)
        print('Elastic deformation 1 done')

        X_deformed2,Y_deformed2 = apply_transform(X_train,Y_train,sigma=2)
        print('Elastic deformation 2 done')

        X_train=np.concatenate((X_train,X_deformed1))
        Y_train=np.concatenate((Y_train,Y_deformed1))
        X_train=np.concatenate((X_train,X_deformed2))
        Y_train=np.concatenate((Y_train,Y_deformed2))

    #Transform masks into two channel images
    Y_train[Y_train>0] = 1
    Y_train_re = np.zeros((Y_train.shape[0], Y_train.shape[1], Y_train.shape[2], 2),np.float32)
    for i in range(2):
        Y_train_re[:,:, :, i][Y_train == i] = 1

    #Apply adaptive histogram equalization
    if True:
        X_train = np.array([exposure.equalize_adapthist(x, clip_limit=0.01) for x in X_train])
        X_testA = np.array([exposure.equalize_adapthist(x, clip_limit=0.01) for x in X_testA])
        X_testB = np.array([exposure.equalize_adapthist(x, clip_limit=0.01) for x in X_testB])

    #Create Dataloaders and apply data augmentation on the images

    batch_size = 16
    transform = Compose([           ToPILImage(),
                                    RandomHorizontalFlip(),
                                    RandomVerticalFlip(),
                                    RandomCrop(img_size),
                                    ToTensor()
                                ])
    img_trans = transforms.Compose([
                                    #transforms.ToPILImage(),
                                    #transforms.RandomGrayscale(),
                                    #transforms.ColorJitter(brightness=0, contrast=0.2, saturation=0.0, hue=.2),
                                    #transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                                    #transforms.ToTensor()

    ])
    mean = np.mean(X_train,axis=(0,1,2))
    std = np.std(X_train,axis=(0,1,2))

    ix = np.random.choice(len(X_train), len(X_train), False)
    #print(ix)
    tr, val = np.split(ix, [int(len(X_train)/4*3)])
    #transnorm = transforms.Compose([transforms.Normalize(mean=mean,std=std)])

    train = CustomTensorDataset(tensors=(torch.from_numpy(np.rollaxis(X_train[tr], 3, 1)), torch.from_numpy(np.rollaxis(Y_train_re[tr],3,1))),transform= transform, normalize=True, mean=mean, std=std,img_trans=img_trans)
    validation = CustomTensorDataset(tensors=(torch.from_numpy(np.rollaxis(X_train[val], 3, 1)), torch.from_numpy(np.rollaxis(Y_train_re[val],3,1))),transform= transform,normalize = True, mean = mean, std=std,img_trans=img_trans)

    data_tr = DataLoader(train, batch_size=batch_size, shuffle=True)
    data_val = DataLoader(validation, batch_size=batch_size, shuffle=False)

    testA = CustomTensorDataset(tensors=(torch.from_numpy(np.rollaxis(X_testA, 3, 1)), torch.from_numpy(np.rollaxis(Y_testA[...,np.newaxis],3,1))), normalize=True, mean=mean, std=std)
    data_testA = DataLoader(testA, batch_size=batch_size, shuffle=False)

    testB = CustomTensorDataset(tensors=(torch.from_numpy(np.rollaxis(X_testB, 3, 1)), torch.from_numpy(np.rollaxis(Y_testB[...,np.newaxis],3,1))), normalize=True, mean=mean, std=std)
    data_testB = DataLoader(testB, batch_size=batch_size, shuffle=False)

    #set device (eiter cpu or gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    use = 'segnet'
    if use=='segnet':
        model = SegNet().to(device)
        summary(model, (3, 256, 256))
    else:
        model = UNet().to(device)
        summary(model, (3, 256, 256))

    F1_scores=train(model, optim.AdamW(model.parameters(),lr=0.0001), 100, data_tr,data_val, data_testA,data_testB,device)

    plt.rcParams['figure.figsize'] = [18, 12]


ig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, sharex=True);
ax0.errorbar(range(len(F1_scores['train_mean'])), F1_scores['train_mean'], yerr=F1_scores['train_std'], fmt='-o')
ax0.set_title('F1-Score train')

ax1.errorbar(range(len(F1_scores['val_mean'])), F1_scores['val_mean'], yerr=F1_scores['val_std'], fmt='-o')
ax1.set_title('F1-Score val')

ax2.errorbar(range(len(F1_scores['testA_mean'])), F1_scores['testA_mean'], yerr=F1_scores['testA_std'], fmt='-o')
ax2.set_title('F1-Score TestA')

ax3.errorbar(range(len(F1_scores['testB_mean'])), F1_scores['testB_mean'], yerr=F1_scores['testB_std'], fmt='-o')
ax3.set_title('F1-Score TestB')
plt.show()
