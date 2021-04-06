import torch
from torch.autograd import grad
import torch.nn as nn
from numpy import genfromtxt
import torch.optim as optim

class DINN(nn.Module):
    def __init__(self, t, S_data, I_data, tao_data, T_star_data, u):
        super(DINN, self).__init__()

        self.t = t
        self.S = S_data
        self.I = I_data
        self.taos = tao_data
        self.T_stars = T_star_data
        self.us = u

        #learnable parameters
        self.alpha1=torch.rand(1)
        self.alpha2=torch.rand(1)
        self.mu=torch.rand(1)
        self.beta=torch.rand(1)

        #predictions
        #self.s_pred=net_x(self.t, self.S, self.I)[0] ##### need to fix with loader or something
        #self.i_pred=net_x(self.t, self.S, self.I)[1]
        #self.f_pred = net_f(self.alpha1, self.alpha2, self.beta)
        
        #loss
        #self.loss = (torch.mean(torch.square(self.S-self.s_pred))+torch.mean(torch.square(self.I-self.i_pred)) + #S, I 
        #torch.mean(torch.square(self.f_pred[0])) + torch.mean(torch.square(self.f_pred[1]))) #f1, f2

        #nets
        def net_x(t, u):            
            self.fc1=nn.Linear(2, 20) #takes t, u
            self.fc2=nn.Linear(20, 20)
            self.out=nn.Linear(20, 4) #outputs S, I, tao, T*

            def forward(self, x):
                x=self.fc1(x)
                x=self.fc2(x)
                x=self.out(x)
                return x
        

        def net_f(t, u):
            S, I, tao, T_star = net_x(t, u)                
            S_t = grad(net_x, S)
            I_t = grad(net_x, I)
            
            f1 = S_t + self.beta * S * I + u * (t>tao) * self.alpha1
            f2 = I_t - self.beta * S * I + self.mu * I + u * (t>tao) * self.alpha2

            return f1, f2
    

        def train(n_epochs):
            print('starting training...')
            losses = []
            learning_rate = 0.01
            momentum = 0.5
            optimizer = optim.SGD(net_x.parameters(), lr = learning_rate,momentum = momentum)

            for epoch in n_epochs:
                for time_step in range(len(self.t)):
                    t_value, u_value = self.t[time_step], self.u[time_step]
                    
                    optimizer.zero_grad()
                    S_pred, I_pred, tao_pred, T_star_pred = net_x(t_value, u_value)
                    f1, f2 = net_f(t_value, u_value)

                    loss = (torch.mean(torch.square(self.S[time_step]-S_pred))+torch.mean(torch.square(self.I[time_step]-I_pred)) + #S,I 
                    torch.mean(torch.square(f1)) + torch.mean(torch.square(f2)) + #f1, f2
                    torch.mean(torch.square(self.taos[time_step]-tao_pred)) + torch.mean(torch.square(self.T_stars[time_step]-T_star_pred))) #tao, T_star

                    loss.backward()
                    optimizer.step()
                    losses.append(loss)

        



if __name__ == "__main__":

    tSI_data = genfromtxt('tSI_data.csv', delimiter=',') #in the form of [t, S, I]
    tao_data = genfromtxt('tao_data.csv', delimiter=',') #in the form of [t, tao_star, T_star, us]

    epochs = 10

    model = DINN(tSI_data[0], tSI_data[1], tSI_data[2], tao_data[1], tao_data[2], tao_data[3]) #t, S_data, I_data, tao_data, T_star_data, u
    model.train(epochs)




    

