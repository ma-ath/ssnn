import torch
import math

class SSNN(torch.nn.Module):
    """
    x[n+1] = Ax[n] + Bu[n]
    y[n]   = Cx[n] + Du[n]
    """

    def __init__(self, u_len:int, x_len:int, y_len:int, x0 = None):
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

        # state vector
        self.x = torch.Tensor(x_len)

        self.u_len = u_len
        self.x_len = x_len
        self.y_len = y_len

        # Initialize this layer
        self.__init_layer(x0)

    def forward(self, u, x0 = None):
        batch_size, sequence_size, input_size = u.size()
        y_seq = []

        if input_size != self.u_len:
            raise ValueError("Size of input different than model input size")

        for batch in range(batch_size):
            for t in range(sequence_size):
                u_t = u[batch,t,:]
          
                self.x = torch.mv(self.A, self.x) + torch.mv(self.B, u_t)
                y = torch.mv(self.C, self.x) + torch.mv(self.D, u_t)
                y = torch.tanh(y)
                y_seq.append(y.unsqueeze(0))

        y_seq = torch.cat(y_seq, dim=0)
        y_seq = y_seq.transpose(0, 1).contiguous()

        return y_seq

    def __init_layer(self, x0):
        stdv = 1.0 / math.sqrt(self.x_len)
        for matrix in self.parameters():
            matrix.data.uniform_(-stdv, stdv)
        
        if x0 is None:
            self.x = torch.zeros(self.x_len).to(self.x.device)
        else:
            self.x = x0


if __name__ == '__main__':
    model = SSNN(3, 3, 3)

    u = torch.ones(4,4,3)

    y = model(u)

    print(u)
    print(y.shape)