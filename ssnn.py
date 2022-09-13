import torch

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
        self.A = torch.Tensor(x_len, x_len)
        self.B = torch.Tensor(x_len, u_len)
        self.C = torch.Tensor(y_len, x_len)
        self.D = torch.Tensor(y_len, u_len)

        # state vector
        self.x = torch.Tensor(x_len)

        self.u_len = u_len


    def forward(self, u):
        if len(u) != self.u_len:
            raise ValueError("Size of input different than model input size")

        self.x = torch.mv(self.A, self.x) + torch.mv(self.B, u)
        y = torch.mv(self.C, self.x) + torch.mv(self.D, u)

        return torch.tanh(y)




if __name__ == '__main__':
    model = SSNN(4, 3, 2)

    u = torch.ones(4)

    y = model(u)

    print(u)
    print(y)