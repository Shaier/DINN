import torch
from torch.autograd import Variable, grad
import torch.nn as nn
from numpy import genfromtxt

class DINN(nn.Module):
    def __init__(self):
        super(DINN, self).__init__()

        self.t=sir_data[0]
        self.S=sir_data[1]
        self.I=sir_data[2]

        self.alpha1=torch.rand(1)
        self.alpha2=torch.rand(1)

        #predictions
        self.s_pred=net_x(self.t, self.S, self.I)[0] ##### need to fix with loader or something
        self.i_pred=net_x(self.t, self.S, self.I)[1]
        self.f_pred = net_f(self.alpha1, self.alpha2)
        
        #loss
        self.loss = (torch.square(self.S-self.s_pred)+torch.square(self.I-self.i_pred) + torch.square(self.f_pred))

        #nets
        def net_x(t, s, i):
            """[summary]

            Args:
                t ([type]): [description]
                s ([type]): [description]
                i ([type]): [description]

            Returns:
                [type]: [description]
            """
            self.fc1=nn.Linear(3, 20) #takes t, S, I
            self.fc2=nn.Linear(20, 20)
            self.out=nn.Linear(20, 2) #outputs S, I

            def forward(self, x):
                x=self.fc1(x)
                x=self.fc2(x)
                x=self.out(x)
                return x
        
        def net_f(alpha1, alpha2):
            """[summary]

            Args:
                alpha1 ([type]): [description]
                alpha2 ([type]): [description]
            """
            pass
            #return equation
    
        def callback(self, loss):
            """prints loss

            Args:
                loss (float): [loss of the function to minimize]
            """
            print(f'Loss: {self.loss}')

        def train(n_epochs):
            """[summary]

            Args:
                n_epochs ([type]): [description]
            """
            for epoch in n_epochs:



if __name__ == "__main__":

    sir_data = genfromtxt('SIR_data.csv', delimiter=',') #in the form of [t, S, I]