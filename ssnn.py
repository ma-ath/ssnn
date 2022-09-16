import torch
import math

class SSNN(torch.nn.Module):
    """
    x[n+1] = Ax[n] + Bu[n]
    y[n]   = Cx[n] + Du[n]
    """

    def __init__(self, u_len:int, x_len:int, y_len:int):
        """State-space neural network model

        Parameters
        ----------
        u_len : int
            input size
        x_len : int
            space vector size
        y_len : int
            output size
        """
        super(SSNN, self).__init__()

        # system matrices
        self.A = torch.nn.parameter.Parameter(torch.Tensor(x_len, x_len), requires_grad=True)
        self.B = torch.nn.parameter.Parameter(torch.Tensor(x_len, u_len), requires_grad=True)
        self.C = torch.nn.parameter.Parameter(torch.Tensor(y_len, x_len), requires_grad=True)
        self.D = torch.nn.parameter.Parameter(torch.Tensor(y_len, u_len), requires_grad=True)

        self.u_len = u_len
        self.x_len = x_len
        self.y_len = y_len

        # Initialize this layer
        self.__init_layer()

    def forward(self, u, x0 = None):
        batch_size, sequence_size, input_size = u.size()
        y_seq = []

        if input_size != self.u_len:
            raise ValueError("Size of input different than model input size")

        # state vector initialization
        if x0 is None:
            x = torch.zeros(batch_size, self.x_len).to(u.device)
            x = x.t() # column vectors
        else:
            x = x0.to(u.device)

        for t in range(sequence_size):
            u_t = u[:,t,:]
            u_t = u_t.t() # work with column vectors
       
            x = torch.matmul(self.A, x) + torch.matmul(self.B, u_t)
            y = torch.matmul(self.C, x) + torch.matmul(self.D, u_t)
            #y = torch.tanh(y)
            y_seq.append(y.unsqueeze(0))

        y_seq = torch.cat(y_seq, dim=0) #(seq, feat, batch)
        y_seq = y_seq.permute(2, 0, 1)  #(batch, seq, feat)

        return y_seq

    def __init_layer(self):
        stdv = 1.0 / math.sqrt(self.x_len)
        for matrix in self.parameters():
            matrix.data.uniform_(-stdv, stdv)


if __name__ == '__main__':       
    import numpy as np
    import matplotlib.pyplot as plt

    X_train = np.arange(0,100,0.5) 
    y_train = np.sin(X_train)
    X_test = np.arange(100,200,0.5) 
    y_test = np.sin(X_test)

    train_series = torch.from_numpy(y_train.reshape((len(y_train), 1)).reshape((10,20,1))).float()
    test_series  = torch.from_numpy(y_test.reshape((len(y_test), 1)).reshape((10,20,1))).float()

    model = SSNN(1, 2, 1)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(100):  # loop over the dataset multiple times
        #print(epoch)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(train_series)
        loss = criterion(outputs, train_series)
        #print("loss:", loss.item())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            outputs = model(test_series)
            loss = criterion(outputs, test_series)
            #print("eval:", loss.item())