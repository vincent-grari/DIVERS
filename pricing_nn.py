from tqdm import tqdm 
from sklearn.utils import shuffle
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from tqdm import tqdm 

class Pricing_NN(torch.nn.Module): 

    def __init__(self,mod_h,mod_g,lr,batch_size,p_device,nbepoch): 
        super().__init__()
        self.lr = lr
        self.batch_size = int(batch_size)
        self.device = torch.device(p_device)
        self.nbepoch = int(nbepoch)
        self.model_h = mod_h()
        self.model_g = mod_g()
    def forward(self, x): 
        out = self.linear(x) 
        return out 

    def predict(self, X_train,G_train,E_train): 
        e_var = torch.FloatTensor(np.expand_dims(E_train,axis = 1)).to(self.device)
        x_var = Variable(torch.FloatTensor(X_train.values)).to(self.device)
        g_out = self.predict_g(G_train)
        yhat= self.model_h(torch.cat((x_var,g_out),1),e_var)
        return yhat

    def predict_g(self, G_train): 
        g_var = Variable((torch.FloatTensor(G_train.values))).to(self.device)
        g_out = self.model_g(g_var)
        return g_out

    def Poisson_Loss(self,yhat, y):
      #loss=torch.mean(torch.exp(xbeta)-y*xbeta)
      loss=torch.mean(yhat-y*torch.log(yhat))
      return loss

    def EDR_POIS(self, yhat, y):
    #loss=torch.mean(torch.exp(xbeta)-y*xbeta)
    #loss=torch.mean(yhat-y*torch.log(yhat))
      eps=0.000000000001
      res=1-np.mean((y*np.log((y+eps)/yhat)-(y-yhat)))/np.mean((y*np.log((y+eps)/np.mean(y))))
      return res
    
    def fit(self, X_train, y_train, G_train, E_train): 
        batch_no = len(X_train) // self.batch_size
        ##### PREDICTOR H #####
        self.model_h=NN_POISS()
        #criterion = nn.PoissonNLLLoss()
        criterion = self.Poisson_Loss
        #, eps=1e-8) #torch.nn.MSELoss() #reduction='mean'
        optimizer_h = torch.optim.Adam(self.model_h.parameters(), lr=self.lr)
        self.model_h.to(self.device)

        ##### PREDICTOR G #####
        self.model_g=NN_G()
        #criterion = torch.nn.MSELoss() #reduction='mean'
        optimizer_g = torch.optim.Adam(self.model_g.parameters(), lr=self.lr)
        self.model_g.to(self.device)
        for epoch in tqdm(range(1, self.nbepoch + 1), 'Epoch: ', leave=False):
            x_train, ytrain, g_train, e_train = shuffle(X_train.values,np.expand_dims(y_train,axis = 1),G_train.values,np.expand_dims(E_train,axis = 1))
            # Mini batch learning
            for i in range(batch_no):
                start = i * self.batch_size
                end = start + self.batch_size
                g_var = Variable(torch.FloatTensor(g_train[start:end])).to(self.device)
                e_var = Variable(torch.FloatTensor(e_train[start:end])).to(self.device)
                x_var = Variable(torch.FloatTensor(x_train[start:end])).to(self.device)
                y_var = Variable(torch.FloatTensor(ytrain[start:end])).to(self.device)
                # Forward + Backward + Optimize

                for l in range(1):
                  optimizer_g.zero_grad()
                  g_out = self.model_g(g_var)
                  ypred_var= self.model_h(torch.cat((x_var,g_out),1),e_var)
                  #ypred_var=  model_h(x_var,e_var)
                  #loss = F.poisson_nll_loss(ypred_var, y_var, reduction='none') 
                  #loss = torch.mean(loss)
                  loss = criterion(ypred_var, y_var)
                  loss.backward()
                  optimizer_g.step()
                  #print('epoch :',epoch,'loss', loss)
        
                optimizer_h.zero_grad()
                g_out = self.model_g(g_var)
                ypred_var= self.model_h(torch.cat((x_var,g_out),1),e_var)
                #ypred_var= model_h(x_var,e_var)
                #loss = F.poisson_nll_loss(ypred_var, y_var, reduction='none') 
                #loss = torch.mean(loss)
                loss = criterion(ypred_var, y_var)
                loss.backward()
                optimizer_h.step()  
                #print(loss)   
        yhat = self.predict(X_train, G_train, E_train).cpu().data.numpy()
        #print(yhat.shape)
        #print(y_train.shape)
        return yhat#print('DONE')
