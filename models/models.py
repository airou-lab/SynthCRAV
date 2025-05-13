#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------
import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import wandb

from models.models_utils.config import device, ndevice
from models.models_utils.models_visualizer import plot_confusion_mat

class Legacy_RadarNDet(pl.LightningModule):
    def __init__(self, n_channels, output_size, history, dropout_prob=0, lr=1e-3):
        super(RadarNDet, self).__init__()

        self.lr = lr
        self.loss_fct=nn.CrossEntropyLoss()
        self.train_step_outputs = {'acc':[],'loss':[]}
        self.validation_step_outputs = {'acc':[],'loss':[]}
        self.test_step_outputs = {'acc':[],'loss':[],'preds':[],'labels':[]}
        self.history = history

        # ------------------------------------------model------------------------------------------
        self.input = nn.Sequential(nn.Conv1d(in_channels=n_channels, out_channels=32, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob))

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob))

        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob))

        # self.conv3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
        #                             nn.ReLU(),
        #                             nn.Dropout1d(p=dropout_prob))

        
        self.fc1 = nn.Sequential(nn.Linear(128,64),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_prob))

        self.fc2 = nn.Sequential(nn.Linear(64,32),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_prob))

        self.head = nn.Sequential(nn.Linear(32,output_size))
        # no softmax because we use cross entropy loss

    def forward(self, x):
        x = self.input(x)

        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)

        x = torch.mean(x, dim=2)  # global average pooling

        x = self.fc1(x)
        x = self.fc2(x)
        
        x = self.head(x)

        return x

class test_RadarNDet(pl.LightningModule):
    def __init__(self, n_channels, output_size, history, dropout_prob=0, lr=1e-3):
        super(RadarNDet, self).__init__()

        self.lr = lr
        self.loss_fct=nn.CrossEntropyLoss()
        self.train_step_outputs = {'acc':[],'loss':[]}
        self.validation_step_outputs = {'acc':[],'loss':[]}
        self.test_step_outputs = {'acc':[],'loss':[],'preds':[],'labels':[]}
        self.history = history
        layer_size = 256

        # ------------------------------------------model------------------------------------------
        self.input = nn.Sequential(nn.Conv1d(in_channels=n_channels, out_channels=32, kernel_size=3),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob),
                                    nn.MaxPool1d(kernel_size=2, stride=2))
        layer_size=np.floor((layer_size-2)/2) 


        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob),
                                    nn.MaxPool1d(kernel_size=2, stride=2))
        layer_size=np.floor((layer_size-2)/2) 


        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob),
                                    nn.MaxPool1d(kernel_size=2, stride=2))
        layer_size=np.floor((layer_size-2)/2) 


        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob),
                                    nn.MaxPool1d(kernel_size=2, stride=2))
        layer_size=np.floor((layer_size-2)/2) 


        self.conv4 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob),
                                    nn.MaxPool1d(kernel_size=2, stride=2))
        layer_size=np.floor((layer_size-2)/2) 


        # self.conv3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
        #                             nn.ReLU(),
        #                             nn.Dropout1d(p=dropout_prob))

        
        # self.fc1 = nn.Sequential(nn.Linear(128,64),
        #                        nn.ReLU(),
        #                        nn.Dropout(p=dropout_prob))

        # self.fc2 = nn.Sequential(nn.Linear(64,32),
        #                        nn.ReLU(),
        #                        nn.Dropout(p=dropout_prob))

        self.head = nn.Sequential(nn.Flatten(), nn.Linear(int(32*layer_size),output_size))
        # no softmax because we use cross entropy loss

    def forward(self, x):
        x = self.input(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv3(x)

        # x = torch.mean(x, dim=2)  # global average pooling

        # x = self.fc1(x)
        # x = self.fc2(x)
        
        x = self.head(x)

        return x

## MODEL INSTANCES

# Radar noise classifier
class RadarNDet(pl.LightningModule):
    def __init__(self, n_channels, output_size, history, dropout_prob=0, lr=1e-3):
        super(RadarNDet, self).__init__()

        self.lr = lr
        self.loss_fct=nn.CrossEntropyLoss()
        self.train_step_outputs = {'acc':[],'loss':[]}
        self.validation_step_outputs = {'acc':[],'loss':[]}
        self.test_step_outputs = {'acc':[],'loss':[],'preds':[],'labels':[]}
        self.history = history

        # ------------------------------------------model------------------------------------------
        self.input = nn.Sequential(nn.Conv1d(in_channels=n_channels, out_channels=32, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob))

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob))

        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
                                    nn.ReLU(),
                                    nn.Dropout1d(p=dropout_prob))

        # self.conv3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
        #                             nn.ReLU(),
        #                             nn.Dropout1d(p=dropout_prob))

        
        self.fc1 = nn.Sequential(nn.Linear(128,64),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_prob))

        self.fc2 = nn.Sequential(nn.Linear(64,32),
                               nn.ReLU(),
                               nn.Dropout(p=dropout_prob))

        self.head = nn.Sequential(nn.Linear(32,output_size))
        # no softmax because we use cross entropy loss

    def forward(self, x):
        x = self.input(x)

        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)

        x = torch.mean(x, dim=2)  # global average pooling

        x = self.fc1(x)
        x = self.fc2(x)
        
        x = self.head(x)

        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-20)



    # train 
    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = self(x)

        #compute loss         
        loss = self.loss_fct(y_hat, y)  

        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1) 
        acc = (preds == y).float().mean()


        self.log('train_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', loss.item(), prog_bar=False, on_step=False, on_epoch=True)

        # self.log('train_acc', acc, prog_bar=True) 
        # self.log('train_acc_epoch', acc, prog_bar=False, on_step=False, on_epoch=True)

        self.train_step_outputs['loss'].append(loss.item())
        self.train_step_outputs['acc'].append(acc)

        return loss

    def on_train_epoch_end(self):
        avg_acc = torch.stack(self.train_step_outputs['acc']).mean()
        tot_loss = sum(self.train_step_outputs['loss'])
        self.log('train_acc', avg_acc, prog_bar=False)

        self.history['train_loss'].append(tot_loss)
        self.history['train_accuracy'].append(avg_acc)

        self.train_step_outputs['acc']=[]  # clear for next epoch
        self.train_step_outputs['loss']=[]  # clear for next epoch

        wandb.log({'loss': tot_loss, 'accuracy': avg_acc})



    # val
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = self(x)

        #compute loss         
        loss = self.loss_fct(y_hat, y)  

        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()

        if not self.trainer.sanity_checking:
            self.log('val_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=False)
            self.log('val_loss_epoch', loss.item(), prog_bar=False, on_step=False, on_epoch=True)

            # self.log('val_loss', loss, prog_bar=True)
            # self.log('val_acc', acc, prog_bar=True)

            self.validation_step_outputs['loss'].append(loss)
            self.validation_step_outputs['acc'].append(acc)
        

        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            avg_acc = torch.stack(self.validation_step_outputs['acc']).mean()
            tot_loss = sum(self.validation_step_outputs['loss'])
            self.log('val_acc', avg_acc, prog_bar=False)

            self.history['val_loss'].append(tot_loss)
            self.history['val_accuracy'].append(avg_acc)

            self.validation_step_outputs['acc']=[]  # clear for next epoch
            self.validation_step_outputs['loss']=[]  # clear for next epoch
        
            wandb.log({'val_loss': tot_loss, 'val_accuracy': avg_acc})


    # test
    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = self(x)

        #compute loss         
        loss = self.loss_fct(y_hat, y)  

        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()

        self.log('test_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=False)
        self.log('test_loss_epoch', loss.item(), prog_bar=False, on_step=False, on_epoch=True)

        # self.log('test_acc', acc, prog_bar=True)

        self.test_step_outputs['loss'].append(loss.item())
        self.test_step_outputs['acc'].append(acc)
        self.test_step_outputs['preds'].append(preds.cpu())
        self.test_step_outputs['labels'].append(y.cpu())

        return loss    

    def on_test_epoch_end(self):
        avg_acc = torch.stack(self.test_step_outputs['acc']).mean()
        tot_loss = sum(self.test_step_outputs['loss'])
        self.log('test_acc', avg_acc, prog_bar=False)

        self.history['test_loss'].append(tot_loss)
        self.history['test_accuracy'].append(avg_acc)

        all_preds = torch.cat(self.test_step_outputs['preds']).numpy()
        all_labels = torch.cat(self.test_step_outputs['labels']).numpy()

        plot_confusion_mat(all_preds, all_labels, os.path.join('./ckpt','radar_model_mat.png'))










# Camera noise classifier
# TODO : try using depthwise convolution layers for lighter network
class CameraNDet(pl.LightningModule):
    def __init__(self, image_shape, output_size, history, conv_k=3, dropout_prob=0, lr=1e-3):
        super(CameraNDet, self).__init__()
        
        self.lr = lr
        self.loss_fct=nn.CrossEntropyLoss()
        self.train_step_outputs = {'acc':[],'loss':[]}
        self.validation_step_outputs = {'acc':[],'loss':[]}
        self.test_step_outputs = {'acc':[],'loss':[],'preds':[],'labels':[]}
        self.history = history

        image_size = np.array(image_shape[:2]) # 1080 x 1920
        in_channels = image_shape[2] # 3

        # ------------------------------------------model------------------------------------------
        self.input = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size=np.floor((image_size-2)/2) 

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size=np.floor((image_size-2)/2) 
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size = np.floor((image_size-2)/2) 

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size = np.floor((image_size-2)/2) 


        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size = np.floor((image_size-2)/2) 

        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size = np.floor((image_size-2)/2) #


        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(conv_k,conv_k), \
                               stride=1, padding=0, bias=True, padding_mode = 'zeros'),
                               nn.ReLU(),
                               nn.Dropout2d(p=dropout_prob),
                               nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        image_size = np.floor((image_size-2)/2) #


        # self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(conv_k,conv_k), \
        #                        stride=1, padding=0, bias=True, padding_mode = 'zeros'),
        #                        nn.ReLU(),
        #                        nn.Dropout2d(p=dropout_prob),
        #                        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        # image_size=np.floor((image_size-2)/2) # 65x118



        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(int(8*image_size[0]*image_size[1]),output_size))

    def forward(self, x):
        x = self.input(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = self.head(x)
        return x
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-20)



    # train 
    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = self(x)

        #compute loss         
        loss = self.loss_fct(y_hat, y)  

        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1) 
        acc = (preds == y).float().mean()


        self.log('train_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_loss_epoch', loss.item(), prog_bar=False, on_step=False, on_epoch=True)

        # self.log('train_acc', acc, prog_bar=True) 
        # self.log('train_acc_epoch', acc, prog_bar=False, on_step=False, on_epoch=True)

        self.train_step_outputs['loss'].append(loss.item())
        self.train_step_outputs['acc'].append(acc)

        return loss

    def on_train_epoch_end(self):
        avg_acc = torch.stack(self.train_step_outputs['acc']).mean()
        tot_loss = sum(self.train_step_outputs['loss'])
        self.log('train_acc', avg_acc, prog_bar=False)

        self.history['train_loss'].append(tot_loss)
        self.history['train_accuracy'].append(avg_acc)

        self.train_step_outputs['acc']=[]  # clear for next epoch
        self.train_step_outputs['loss']=[]  # clear for next epoch

        wandb.log({'loss': tot_loss, 'accuracy': avg_acc})



    # val
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = self(x)

        #compute loss         
        loss = self.loss_fct(y_hat, y)  

        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()

        if not self.trainer.sanity_checking:
            self.log('val_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=False)
            self.log('val_loss_epoch', loss.item(), prog_bar=False, on_step=False, on_epoch=True)

            # self.log('val_loss', loss, prog_bar=True)
            # self.log('val_acc', acc, prog_bar=True)

            self.validation_step_outputs['loss'].append(loss)
            self.validation_step_outputs['acc'].append(acc)
        

        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            avg_acc = torch.stack(self.validation_step_outputs['acc']).mean()
            tot_loss = sum(self.validation_step_outputs['loss'])
            self.log('val_acc', avg_acc, prog_bar=False)

            self.history['val_loss'].append(tot_loss)
            self.history['val_accuracy'].append(avg_acc)

            self.validation_step_outputs['acc']=[]  # clear for next epoch
            self.validation_step_outputs['loss']=[]  # clear for next epoch
        
            wandb.log({'val_loss': tot_loss, 'val_accuracy': avg_acc})

    # test
    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_hat = self(x)

        #compute loss         
        loss = self.loss_fct(y_hat, y)  

        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()

        self.log('test_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=False)
        self.log('test_loss_epoch', loss.item(), prog_bar=False, on_step=False, on_epoch=True)

        # self.log('test_acc', acc, prog_bar=True)

        self.test_step_outputs['loss'].append(loss.item())
        self.test_step_outputs['acc'].append(acc)
        self.test_step_outputs['preds'].append(preds.cpu())
        self.test_step_outputs['labels'].append(y.cpu())

        return loss    

    def on_test_epoch_end(self):
        avg_acc = torch.stack(self.test_step_outputs['acc']).mean()
        tot_loss = sum(self.test_step_outputs['loss'])
        self.log('test_acc', avg_acc, prog_bar=False)

        self.history['test_loss'].append(tot_loss)
        self.history['test_accuracy'].append(avg_acc)

        all_preds = torch.cat(self.test_step_outputs['preds']).numpy()
        all_labels = torch.cat(self.test_step_outputs['labels']).numpy()

        plot_confusion_mat(all_preds, all_labels, os.path.join('./ckpt','camera_model_mat.png'))





# # train / test functions
# def train(model,args,df_train,df_val,optimizer,loss_fct,wandb):
#     # init loss self.history
#     self.history = {'train_loss':[],
#                 'val_loss':[],
#                 'test_loss':[],
#                 'train_accuracy':[],
#                 'val_accuracy':[],
#                 'test_accuracy':[],
#                 'test_results':[]}

#     print('Training')
#     for epoch in range(args.n_epochs):
#         print('Epoch:',epoch,'/',args.n_epochs)
#         ################Training################
#         model.train()   # set model in train mode
#         epoch_loss = 0
#         correct_predictions=0
#         total_predictions=0

#         for batch_idx, batch in enumerate(batch_generator(args,df_train)):
#             print(100*' ',end='\r')
#             print('batch:',batch_idx, '/', round(len(df_train)/args.batch_size),end='\r')
#             # Load data
#             X_train = torch.stack([x[0] for x in batch])
#             labels_train = torch.stack([x[1] for x in batch])
            
#             # generate predictions (i.e noise values)
#             pred = model(X_train)

#             # calculate loss
#             loss = loss_fct(pred,labels_train)

#             # backprop loss
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # Accumulate loss
#             epoch_loss += loss.item()

#             # Calculate accuracy
#             _, predicted = torch.max(pred, 1)
#             correct_predictions += (predicted == labels_train).sum().item()
#             total_predictions += labels_train.size(0)

#             # print('running loss: %0.3f'%(loss.item()),end='\r')

#         epoch_accuracy = 100 * (correct_predictions/total_predictions)
        
#         print('\nEpoch train loss: %0.3f \t | \taccuracy:%0.2f'%(epoch_loss, epoch_accuracy))

#         # clear up training tensor from mem
#         del X_train
#         torch.cuda.empty_cache()  # Clear cache after deletion

#         ################VALIDATION################
#         model.eval()        # eval mode
#         val_loss=0
#         correct_predictions_val=0
#         total_predictions_val=0

#         with torch.no_grad():
#             for batch_idx, val_batch in enumerate(batch_generator(args,df_val)):
#                 print(150*' ',end='\r')
#                 print('val batch:',batch_idx, '/', round(len(df_val)/args.batch_size),end='\r')
                
#                 # Load data
#                 X_val = torch.stack([x[0] for x in val_batch]).detach()
#                 labels_val = torch.stack([x[1] for x in val_batch]).detach()

#                 val_pred = model(X_val)
#                 val_loss += loss_fct(val_pred,labels_val).item()

#                 # Calculate accuracy
#                 _, predicted = torch.max(val_pred, 1)
#                 correct_predictions_val += (predicted == labels_val).sum().item()
#                 total_predictions_val += labels_val.size(0)

#             val_accuracy = 100 * (correct_predictions_val/total_predictions_val)

#         print('\nEpoch val loss: %0.3f \t\t | \taccuracy:%0.2f'%(val_loss, val_accuracy))


#         # scheduler step
#         if args.scheduler:
#             scheduler.step()

#         # output values
#         self.history['train_loss'].append(epoch_loss)
#         self.history['train_accuracy'].append(epoch_accuracy)
#         self.history['val_loss'].append(val_loss)
#         self.history['val_accuracy'].append(val_accuracy)

#         wandb.log({'train_loss': epoch_loss, 'train_accuracy': epoch_accuracy, 'val_loss': val_loss, 'val_accuracy': val_accuracy})

#         # Save model and results
#         if args.save_model:
#             torch.save(model.state_dict(), args.output_path+args.sensor_type+'_model.pth')

#         if args.save_self.hist:
#             with open(args.output_path+args.sensor_type+'_model_self.hist.pkl', 'wb') as f:
#                 pickle.dump(self.history,f)

#     print('\nFinal loss: \t train: %0.3f \t val: %0.3f'%(epoch_loss,val_loss))
#     print('Final accuracy: \t train: %0.2f \t val: %0.2f'%(epoch_accuracy,val_accuracy))

#     return model, self.history

# def test(model,args,df_test,loss_fct,self.history):
#     print('Testing')
#     ################TESTING################
#     model.eval()
#     test_loss=0
#     correct_predictions_test=0
#     total_predictions_test=0
#     all_preds = []
#     all_labels = []

#     start_t = time.perf_counter()

#     with torch.no_grad():
#         for batch_idx, test_batch in enumerate(batch_generator(args,df_test)):
#             print(150*' ',end='\r')
#             print('test batch:',batch_idx, '/', round(len(df_test)/args.batch_size),end='\r')
            
#             # Load data
#             X_test = torch.stack([x[0] for x in test_batch])
#             labels_test = torch.stack([x[1] for x in test_batch])

#             test_pred = model(X_test)
#             test_loss += loss_fct(test_pred,labels_test).item()

#              # Calculate accuracy
#             _, predicted = torch.max(test_pred, 1)
#             correct_predictions_test += (predicted == labels_test).sum().item()
#             total_predictions_test += labels_test.size(0)

#             all_preds.extend(predicted.cpu().numpy())  # Convert to numpy and store
#             all_labels.extend(labels_test.cpu().numpy())  # Convert to numpy and store
        
#     test_accuracy = 100 * (correct_predictions_test/total_predictions_test)

#     end_t = time.perf_counter()

#     print('Test loss: %0.3f \t | \taccuracy:%0.2f'%(test_loss,test_accuracy))
#     print('Total inference time:',end_t - start_t,'s across',len(df_test),'samples. =>',(end_t - start_t)/len(df_test),'s per sample.')

#     # Convert lists to numpy arrays
#     all_preds = np.array(all_preds)
#     all_labels = np.array(all_labels)

#     # output values
#     self.history['test_loss'].append(test_loss)
#     self.history['test_accuracy'].append(test_accuracy)

#     self.history['test_results'] = {'preds':all_preds,'labels':all_labels}

#     plot_confusion_mat(all_preds, all_labels,args.output_path+args.sensor_type+'_model_mat.png')

#     return self.history
