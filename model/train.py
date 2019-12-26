import numpy as np

#Network
import torch
import torch.nn as nn

#plot
import matplotlib.pyplot as plt
from IPython.display import clear_output

#performance measurement
from time import time
from model.utils.Evaluation import F1_score

def train(model, opt, loss_fn, epochs, data_tr,data_val, data_testA,data_testB):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    #criterion = nn.BCELoss() # not possible due to multiple channels
    criterion = nn.CrossEntropyLoss()
    X_val, Y_val = next(iter(data_val))
    X_testA, Y_testA = next(iter(data_testA))
    X_testB, Y_testB = next(iter(data_testB))

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
            plt.imshow(np.rollaxis(X_val[k].numpy(), 0, 3), cmap='gray')
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
            plt.imshow(np.rollaxis(X_testA[k].numpy(), 0, 3), cmap='gray')
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
            plt.imshow(np.rollaxis(X_testB[k].numpy(), 0, 3), cmap='gray')
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
