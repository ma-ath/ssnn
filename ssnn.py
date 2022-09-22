import torch
import math

class SSNN(torch.nn.Module):
    """
    x[n+1] = Ax[n] + Bu[n]
    y[n]   = Cx[n] + Du[n]
    """

    def __init__(self, u_len:int, x_len:int, y_len:int, **kwargs):
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
            # y = torch.tanh(y)
            y_seq.append(y.unsqueeze(0))

        y_seq = torch.cat(y_seq, dim=0) #(seq, feat, batch)
        y_seq = y_seq.permute(2, 0, 1)  #(batch, seq, feat)

        return y_seq

    def __init_layer(self):
        stdv = 1.0 / math.sqrt(self.x_len)
        for matrix in self.parameters():
            matrix.data.uniform_(-stdv, stdv)