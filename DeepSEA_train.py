import scipy.io
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import time
import visdom 
import mkdir
import bestmodel
torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark=True

## Hyper Parameters
EPOCH = 100
BATCH_SIZE = 100
LR = 0.01
save_model_time = '0526'
# build model file
mkpath = 'model/model%s'% save_model_time
mkdir(mkpath) 

print('starting loading the data')
np_valid_data = scipy.io.loadmat('valid.mat')

validX_data = torch.FloatTensor(np_valid_data['validxdata'])
validY_data = torch.FloatTensor(np_valid_data['validdata'])

params = {'batch_size': 100,'num_workers': 2}

valid_loader = Data.DataLoader(
    dataset=Data.TensorDataset(validX_data, validY_data), 
    shuffle=False,
    **params)

vis = visdom.Visdom(env='DeepSEA')

win = vis.line(
    X=np.array([0]),
    Y=np.array([0]),
    opts=dict(
        title='LOSS-EPOCH(%s)' % save_model_time,
        showlegend=True,),
    name="train")
vis.line(
    X=np.array([0]),
    Y=np.array([0]),
    win=win,
    update="new",
    name="val",
)

print('compling the network')
class DeepSEA(nn.Module):
    def __init__(self, ):
        super(DeepSEA, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=320, out_channels=480, kernel_size=8)
        self.Conv3 = nn.Conv1d(in_channels=480, out_channels=960, kernel_size=8)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Drop2 = nn.Dropout(p=0.5)
        self.Linear1 = nn.Linear(53*960, 925)
        self.Linear2 = nn.Linear(925, 919)

    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop2(x)
        x = x.view(-1, 53*960)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x


deepsea = DeepSEA()
deepsea.cuda()
print(deepsea)
optimizer = optim.SGD(deepsea.parameters(), lr=LR,momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,verbose=1)
loss_func = nn.BCEWithLogitsLoss()

print('starting training')
# training and validating
since = time.time()

train_losses = []
valid_losses = []

for epoch in range(EPOCH):
    deepsea.train()
    train_loss = 0
    for i in range(1,11):
        trainX_data = torch.load('pt_data/%s.pt' % str(i))
        trainY_data = torch.load('pt_label/%s.pt' % str(i))
        train_loader = Data.DataLoader(dataset=Data.TensorDataset(trainX_data, trainY_data), shuffle=True, **params)
        for step, (train_batch_x, train_batch_y) in enumerate(train_loader):

            train_batch_x = train_batch_x.cuda()
            train_batch_y = train_batch_y.cuda()

            out = deepsea(train_batch_x)
            loss = loss_func(out, train_batch_y)
           # all_para = torch.cat([y.view(-1) for y in deepsea.Conv3.parameters()])
           # l2_reg = 0.9*torch.norm(all_para,2)
            #loss_sum = loss+l2_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
    i = 1

    if epoch % 5 == 0:
        torch.save(deepsea, 'model/model{save_model_time}/deepsea_net_{epoch}.pkl'.format(save_model_time=save_model_time,epoch=int(epoch/5)))
        torch.save(deepsea.state_dict(), 'model/model{save_model_time}/deepsea_net_params_{epoch}.pkl'.format(save_model_time=save_model_time,epoch=int(epoch/5)))
 
    deepsea.eval()

    for valid_step, (valid_batch_x, valid_batch_y) in enumerate(valid_loader):

        valid_batch_x = valid_batch_x.cuda()
        valid_batch_y = valid_batch_y.cuda()

        val_out = deepsea(valid_batch_x)
        val_loss = loss_func(val_out, valid_batch_y)
        valid_losses.append(val_loss.item())
        
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    
    scheduler.step(valid_loss)
    
    epoch_len = len(str(epoch))

    print_msg = (f'[{epoch:>{epoch_len}}/{EPOCH:>{epoch_len}}] ' +
                 f'train_loss: {train_loss:.5f} ' +
                 f'valid_loss: {valid_loss:.5f}')

    print(print_msg)
    
    vis.line(
        X=np.array([epoch]),
        Y=np.array([train_loss]),
        win=win,
        update="append",
        name="train"
    )
    vis.line(
        X=np.array([epoch]),
        Y=np.array([valid_loss]),
        win=win,
        update="append",
        name="val"
    )
    #save bestmodel
    bestmodel(deepsea,save_model_time,valid_loss)
    
    train_losses = []
    valid_losses = []

time_elapsed = time.time() - since
print('time:', time_elapsed)
torch.save(deepsea, 'model/model{save_model_time}/deepsea_net_final.pkl'.format(save_model_time=save_model_time))  # save entire net
torch.save(deepsea.state_dict(), 'model/model{save_model_time}/deepsea_net_params_final.pkl'.format(save_model_time=save_model_time))
