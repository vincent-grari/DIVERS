import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from tqdm import tqdm 

class Pricing_NN(torch.nn.Module): 

    def __init__(self,lr,batch_size,p_device,nbepoch): 
        self.lr = lr
        self.batch_size = int(batch_size)
        self.device = torch.device(p_device)
        self.nbepoch = int(nbepoch)

    def forward(self, x): 
        out = self.linear(x) 
        return out 

    def Poisson_Loss(yhat, y):
      #loss=torch.mean(torch.exp(xbeta)-y*xbeta)
      loss=torch.mean(yhat-y*torch.log(yhat))
      return loss
  
    def fit(self, X_train, y_train, G_train, E_train,NN_POISS,NN_G): 

        batch_no = len(X_train) // self.batch_size
        ##### PREDICTOR H #####
        model_h = NN_POISS()
        #criterion = nn.PoissonNLLLoss()
        criterion = self.Poisson_Loss
        #, eps=1e-8) #torch.nn.MSELoss() #reduction='mean'
        optimizer_h = torch.optim.Adam(model_h.parameters(), lr=self.lr)
        model_h.to(self.device)

        ##### PREDICTOR G #####
        model_g = NN_G()
        #criterion = torch.nn.MSELoss() #reduction='mean'
        optimizer_g = torch.optim.Adam(model_g.parameters(), lr=self.lr)
        model_g.to(self.device)
        for epoch in tqdm(range(1, self.nbepoch + 1), 'Epoch: ', leave=False):
            x_train, ytrain, g_train, e_train = shuffle(X_train.values,np.expand_dims(y_train,axis = 1),G_train.values,np.expand_dims(E_train,axis = 1))
            # Mini batch learning
            epsilon=0.00000000000000001
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
                  g_out = model_g(g_var)
                  ypred_var= model_h(torch.cat((x_var,g_out),1),e_var)
                  #ypred_var=  model_h(x_var,e_var)
                  #loss = F.poisson_nll_loss(ypred_var, y_var, reduction='none') 
                  #loss = torch.mean(loss)
                  loss = criterion(ypred_var, y_var)
                  loss.backward()
                  optimizer_g.step()
                  #print('epoch :',epoch,'loss', loss)
        
                optimizer_h.zero_grad()
                g_out = model_g(g_var)
                ypred_var= model_h(torch.cat((x_var,g_out),1),e_var)
                #ypred_var= model_h(x_var,e_var)
                #loss = F.poisson_nll_loss(ypred_var, y_var, reduction='none') 
                #loss = torch.mean(loss)
                loss = criterion(ypred_var, y_var)
                loss.backward()
                optimizer_h.step()  
                #print(loss)     
        return self
