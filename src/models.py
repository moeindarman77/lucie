import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def numpy_to_cuda(arr):
    return torch.from_numpy(arr).float().cuda()

def cuda_to_numpy(arr):
    return arr.cpu().detach().numpy()

def get_num_parameters(net):
    return sum(p.numel() for p in net.parameters())

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x
    
## https://github.com/pytorch/examples/blob/main/super_resolution/model.py

def cuda_memory_info():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print(f"reserved: {r/10e6}")
    print(f"allocated: {a/10e6}")
    print(f"free: {f/10e6}")

class CNN2D(nn.Module):
    def __init__(self, channelsin = 3, channelsout = 3):
        super(CNN2D, self).__init__()

        self.lrelu1 = nn.LeakyReLU()
        self.lrelu2 = nn.LeakyReLU()
        self.lrelu3 = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(channelsin, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, channelsout, (3, 3), (1, 1), (1, 1))
        self._initialize_weights()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.conv4(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('leaky_relu'))
        init.orthogonal_(self.conv4.weight)

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    @staticmethod
    def compl_mul2d(input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights) ## batch, input, lat, lon EIN TENSOR PRODUCT input, output, lat, lon

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, padding = 8, channels = 3, channelsout = 3):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic
        self.p = nn.Linear(channels+2, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.dconv0 = MLP(self.width, self.width, self.width)
        self.dconv1 = MLP(self.width, self.width, self.width)
        self.dconv2 = MLP(self.width, self.width, self.width)
        self.dconv3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, channelsout, self.width * 4) # output channel is 1: u(x, y)
        
    ## this is not working....bring back the grid addition in this step if cant fix...done
    def forward(self, x):
        # adds grid coordinates to the end of the input array for FNO prediction x,y coordinates
        grid = self.get_grid(x.shape, x.device)
        #grid = get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        #x = F.pad(x, [0,self.padding, 0, 0]) # pad the domain if input is non-periodic
        x = x.permute(0, 3, 1, 2)
        
        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.dconv0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.dconv1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.dconv2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.dconv3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        #print(x.shape)
        x = self.q(x) # why does the resulting shape have the same?
        #print(x.shape)
        x = x.permute(0, 2, 3, 1)
        #x = x[:, :-self.padding, :, :] # pad the domain if input is non-periodic
        return x

    def get_grid(self, shape, device):
        # normalized from 0 to 1?
        ## this is a pretty strange way to implement for grid free prediction
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class FNO2d_ds(nn.Module): ## issues with small predictions for some reason
    def __init__(self, modes1, modes2, width, padding = 8, channels = 3, channelsout = 3):
        super(FNO2d_ds, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = padding # pad the domain if input is non-periodic
        self.p = nn.Linear(channels+2, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.dconv0 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, channelsout, self.width * 4) # output channel is 1: u(x, y)
        
    ## this is not working....bring back the grid addition in this step if cant fix...done
    def forward(self, x):
        # adds grid coordinates to the end of the input array for FNO prediction x,y coordinates
        grid = self.get_grid(x.shape, x.device)
        #grid = get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        #x = F.pad(x, [0,self.padding, 0, 0]) # pad the domain if input is non-periodic
        x = x.permute(0, 3, 1, 2)
        
        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.dconv0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        
        #print(x.shape)
        x = self.q(x) # why does the resulting shape have the same?
        #print(x.shape)
        x = x.permute(0, 2, 3, 1)
        #x = x[:, :-self.padding, :, :] # pad the domain if input is non-periodic
        return x

    def get_grid(self, shape, device):
        # normalized from 0 to 1?
        ## this is a pretty strange way to implement for grid free prediction
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class FNO2D_grid(nn.Module):
    def __init__(self, modes1, modes2, width, padding = 8, channels = 3, channelsout = 3, gridx=[0,1], gridy=[0,1]):
        super(FNO2D_grid, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.gridx = gridx
        self.gridy = gridy
        self.padding = 8 # pad the domain if input is non-periodic
        self.p = nn.Linear(channels+2, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.sconv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.sconv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.sconv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.sconv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.dconv0 = MLP(self.width, self.width, self.width)
        self.dconv1 = MLP(self.width, self.width, self.width)
        self.dconv2 = MLP(self.width, self.width, self.width)
        self.dconv3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, channelsout, self.width * 4) # output channel is 1: u(x, y)
        
    ## this is not working....bring back the grid addition in this step if cant fix...done
    def forward(self, x):
        # adds grid coordinates to the end of the input array for FNO prediction x,y coordinates
        grid = self.get_grid(x.shape, x.device)
        #grid = get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        #x = F.pad(x, [0,self.padding, 0, 0]) # pad the domain if input is non-periodic
        x = x.permute(0, 3, 1, 2)
        
        x1 = self.norm(self.sconv0(self.norm(x)))
        x1 = self.dconv0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.sconv1(self.norm(x)))
        x1 = self.dconv1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.sconv2(self.norm(x)))
        x1 = self.dconv2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.sconv3(self.norm(x)))
        x1 = self.dconv3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        #print(x.shape)
        x = self.q(x) # why does the resulting shape have the same?
        #print(x.shape)
        x = x.permute(0, 2, 3, 1)
        #x = x[:, :-self.padding, :, :] # pad the domain if input is non-periodic
        
        return x

    def get_grid(self, shape, device):
        ## modifications to include lat/lon coordinates being inputted into the function
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(self.gridx, dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(self.gridy, dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class FNO2D_grid_pad(nn.Module):
    def __init__(self, modes1 = 20, modes2 = 20, width = 10, padding = 8, channels = 3, channelsout = 3, gridx=[0,1], gridy=[0,1], padval = 0, pbias = False):
        super(FNO2D_grid_pad, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.gridx = gridx
        self.gridy = gridy
        self.padding = padding
        self.padval = padval# pad the domain if input is non-periodic
        ## changing so grid is added after raising. 
        ##  notice, grid is added after raising linear p, so width will be two less than final width
        self.p = nn.Linear(channels+2, self.width, bias = pbias) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        # self.p = nn.Linear(channels, self.width-2, bias = pbias) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.sconv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.sconv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.sconv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.sconv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.dconv0 = MLP(self.width, self.width, self.width)
        self.dconv1 = MLP(self.width, self.width, self.width)
        self.dconv2 = MLP(self.width, self.width, self.width)
        self.dconv3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1) ## not sure this is correct
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, channelsout, self.width * 4) # output channel is 1: u(x, y)
        
    ## this is not working....bring back the grid addition in this step if cant fix...done
    def forward(self, x):
        # adds grid coordinates to the end of the input array for FNO prediction x,y coordinates
        grid = self.get_grid(x.shape, x.device)
        #grid = get_grid(x.shape, x.device)
        
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        
        ## flipping these next two steps...shouldn't the grid dimensions not effect the raising?
        # x = self.p(x)
        # x = torch.cat((x, grid), dim=-1)
        
        ## permutation before, last 2 dimensions
        x = x.permute(0, 3, 1, 2)
        
        if self.padding is not None:
            ## to try, replicate pad? right now it is set to zeros...
            ## or manually create pad from predefined boundary pad, calculated from mean channel values at boundary
            x = F.pad(x, [self.padding,self.padding,self.padding,self.padding], mode = "constant", value = self.padval) # pad the domain if input is non-periodic
        
        # print(x.shape)
        
        x1 = self.norm(self.sconv0(self.norm(x)))
        x1 = self.dconv0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.sconv1(self.norm(x)))
        x1 = self.dconv1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.sconv2(self.norm(x)))
        x1 = self.dconv2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.sconv3(self.norm(x)))
        x1 = self.dconv3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        #print(x.shape)
        x = self.q(x) # why does the resulting shape have the same?
        #print(x.shape)
        x = x.permute(0, 2, 3, 1)
        
        if self.padding is not None:
            x = x[:, self.padding:-self.padding, self.padding:-self.padding, :] # pad the domain if input is non-periodic
        
        return x

    def get_grid(self, shape, device):
        ## modifications to include lat/lon coordinates being inputted into the function
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(self.gridx, dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(self.gridy, dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
    # def get_grid(self, shape, device):
        # # normalized from 0 to 1?
        # ## this is a pretty strange way to implement for grid free prediction
        # batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        # print(size_x)
        # print(size_y)
        # print(self.gridx.shape)
        # print(self.gridy.shape)
        # gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        # gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        # gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        # gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        # return torch.cat((gridx, gridy), dim=-1).to(device)

class FNO2D_grid_pad2(nn.Module):
    def __init__(self, 
                 modes1 = 20, 
                 modes2 = 20, 
                 width = 10, 
                 padding = 8, 
                 channels = 3, 
                 channelsout = 3, 
                 gridx=[0,1], 
                 gridy=[0,1], 
                 padval = 0, 
                 pbias = True,
                 fourier_layers = 4
                 ):
        super(FNO2D_grid_pad2, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.gridx = gridx
        self.gridy = gridy
        self.padding = padding
        self.padval = padval # pad the domain if input is non-periodic
        ## changing so grid is added after raising. 
        ##  notice, grid is added after raising linear p, so width will be two less than final width
        self.p = nn.Linear(channels+2, self.width, bias = pbias) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        # self.p = nn.Linear(channels, self.width-2, bias = pbias) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.spectral_layers = nn.ModuleList([])

        for layer in range(fourier_layers):
            self.spectral_layers.append(nn.ModuleDict({
                                                        "sconv" : SpectralConv2d(self.width, self.width, self.modes1, self.modes2),
                                                        "mlp" : MLP(self.width, self.width, self.width), ## 2 layer linear, gelu activation
                                                        "w" : nn.Conv2d(self.width, self.width, kernel_size=1, bias=True), ## use Conv2d so you dont have to permute the tensors dimensions
                                                        #"c" : nn.Conv2d(self.width, self.width, kernel_size=10, padding="same", padding_mode="replicate", bias=True), ## same with kernel=1 ## bias not needed since mlp already has bias term in it
                                                     })
                                       )

        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, channelsout, self.width) # output channel is 1: u(x, y)
        
    ## this is not working....bring back the grid addition in this step if cant fix...done
    def forward(self, x):
        # adds grid coordinates to the end of the input array for FNO prediction x,y coordinates
        grid = self.get_grid(x.shape, x.device)
        #grid = get_grid(x.shape, x.device)
        
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        
        ## flipping these next two steps...shouldn't the grid dimensions not effect the raising?
        # x = self.p(x)
        # x = torch.cat((x, grid), dim=-1)
        
        ## permutation before, expected for torch tensor operations: batch number x channels x xdim x ydim
        x = x.permute(0, 3, 1, 2)
        
        if self.padding is not None:
            ## to try, replicate pad? right now it is set to zeros...
            ## or manually create pad from predefined boundary pad, calculated from mean channel values at boundary
            x = F.pad(x, [self.padding,self.padding,self.padding,self.padding], mode = "constant", value = self.padval) # pad the domain if input is non-periodic
        
        for spectral_layer in self.spectral_layers:
            x1 = spectral_layer["mlp"](self.norm(spectral_layer["sconv"](self.norm(x))))
            # x1 = spectral_layer["dconv"](spectral_layer["sconv"](x))
            x2 = spectral_layer["w"](x)  # linear transform
            # x3 = spectral_layer["c"](x)  # convolution 
            x = F.gelu(x1 + x2)
            # x = F.gelu(x1 + x2 + x3)

        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        
        if self.padding is not None:
            x = x[:, self.padding:-self.padding, self.padding:-self.padding, :] # pad the domain if input is non-periodic
        
        return x

    def get_grid(self, shape, device):
        ## modifications to include lat/lon coordinates being inputted into the function
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(self.gridx, dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(self.gridy, dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

##########################
## UNET2 implementation ##
##########################

#utils
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, 
            in_channels // 2, 
            kernel_size=2, 
            stride=2
        )
        
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        ## need padding to match dimensionality of x1 and x2 tensors
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1, 
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )
    
    def forward(self, x):
        return self.conv(x)

#model  
class UNET2(nn.Module): ## modified from https://github.com/kyegomez/SimpleUnet/blob/main/unet/model.py
    def __init__(
        self,
        channelsin,
        channelsout,
        ## bilinear = True
    ):
        super(UNET2, self).__init__()
        self.channelsin = channelsin
        self.channelsout = channelsout
        # self.bilinear = bilinear

        self.inc = DoubleConv(channelsin, channelsin*4)
        
        self.down1 = Down(channelsin*4, channelsin*8)
        self.down2 = Down(channelsin*8, channelsin*16)
        self.down3 = Down(channelsin*16, channelsin*32)
        self.down4 = Down(channelsin*32, channelsin*64)

        # factor = 2 if bilinear else 1

        # self.down4 = Down(512, 1024 // factor)
        
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)

        self.up1 = Up(channelsin*64, channelsin*32)
        self.up2 = Up(channelsin*32, channelsin*16)
        self.up3 = Up(channelsin*16, channelsin*8)
        
        self.up4 = Up(channelsin*8, channelsin*4)
        self.outc = OutConv(channelsin*4, channelsout)
    
    # def forward(self, x):
    #     x = x.permute(0,3,1,2)
    #     print(x.shape)
    #     x1 = self.inc(x)
    #     print(x1.shape)
    #     x2 = self.down1(x1)
    #     print(x2.shape)
    #     x3 = self.down2(x2)
    #     print(x3.shape)
    #     x4 = self.down3(x3)
    #     print(x4.shape)
    #     x5 = self.down4(x4)
    #     print(x5.shape)

    #     x = self.up1(x5, x4)
    #     print(x.shape)
    #     x = self.up2(x, x3)
    #     print(x.shape)
    #     x = self.up3(x, x2)
    #     print(x.shape)

    #     x = self.up4(x, x1)
    #     print(x.shape)

    #     logits = self.outc(x)
    #     print(logits.shape)

    #     logits = logits.permute(0,2,3,1)
    #     return logits

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)

        x = self.up4(x, x1)

        logits = self.outc(x)

        logits = logits.permute(0,2,3,1)
        return logits


def get_loss_cond(model, x_0, t, label_batch, loss_func):  #???
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return  loss_func((x_noisy-noise_pred),label_batch ,wavenum_init,lamda_reg)

def regular_loss(output, target):
    loss = torch.mean((output-target)**2)
    return loss

def ocean_loss(output, target, ocean_grid):
    loss = (torch.sum((output-target)**2))/ocean_grid
    return loss

def mseloss_mask(output, target, mask = None, loss_weights = [1,1,1]):
    run_loss = torch.zeros(1).float().cuda()
    loss_weights_norm = np.array(loss_weights)/np.linalg.norm(loss_weights)

    for iw, w in enumerate(loss_weights_norm,0):
        loss_all = (output[...,iw]-target[...,iw])**2
        run_loss += w*torch.mean((output[...,iw]-target[...,iw])**2)

    return run_loss

def spectral_sqr_abs(output, 
                               target, 
                               grid_valid_size = None,
                               wavenum_init_lon = 1, 
                               wavenum_init_lat = 1, 
                               lambda_fft = .5,
                               lat_lon_bal = .5,
                               channels = "all",
                               fft_loss_scale = 1./110.):
        
    """
    Grid and spectral losses, both with mse
    """
    # loss1 = torch.sum((output-target)**2)/ocean_grid + torch.sum((output2-target2)**2)/ocean_grid
    
    ## loss from grid space
    if grid_valid_size is None: 
        grid_valid_size = output.flatten().shape[0]
        
    loss_grid = torch.sum((output-target)**2)/(grid_valid_size*len(channels))
    # loss_grid = torch.mean((output-target)**2)
    # loss1 = torch.abs((output-tnparget))/ocean_grid

    run_loss_run = torch.zeros(1).float().cuda()
    
    if channels == "all":
        num_spectral_chs = output.shape[-1]
        channels = [["_",i,1./num_spectral_chs] for i in np.arange(num_spectral_chs)]
    
    totcw = 0
    for [cname,c,cw] in channels:
        ## it makes sense for me to take fft differences before...if you take mean, you lose more important differences at the equator?
        ## losses from fft, both lat (index 1) and lon (index 2) directions
        ## lat
        out_fft_lat = torch.abs(torch.fft.rfft(output[:,:,:,c],dim=1))[:,wavenum_init_lon:,:]
        target_fft_lat = torch.abs(torch.fft.rfft(target[:,:,:,c],dim=1))[:,wavenum_init_lon:,:]
        loss_fft_lat = torch.mean((out_fft_lat - target_fft_lat)**2)
        ## lon
        out_fft_lon = torch.abs(torch.fft.rfft(output[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
        target_fft_lon = torch.abs(torch.fft.rfft(target[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
        loss_fft_lon = torch.mean((out_fft_lon - target_fft_lon)**2)

        run_loss_run += ((1-lat_lon_bal)*loss_fft_lon + lat_lon_bal*loss_fft_lat)*cw
        totcw+=cw
        
    ## fft_loss_scale included, so the lambda_fft is more intuitive (lambda_fft = .5 --> around half weighted shared between grid and fft loss)
    loss_fft = run_loss_run/totcw*fft_loss_scale
    loss_fft_weighted = lambda_fft*loss_fft
    loss_grid_weighted = ((1-lambda_fft))*loss_grid
    loss = loss_grid_weighted + loss_fft_weighted
    
    # return loss, loss_grid, loss_fft
    return loss

def spectral_sqr_abs2(output, 
                        target, 
                        grid_valid_size = None,
                        wavenum_init_lon = 1, 
                        wavenum_init_lat = 1, 
                        lambda_fft = .5,
                        lat_lon_bal = .5,
                        channels = "all",
                        fft_loss_scale = 1./110.):
        
    """
    Grid and spectral losses, both with mse
    """

    loss_types = {}

    ## loss from grid space
    if grid_valid_size is None: 
        grid_valid_size = output.flatten().shape[0]
        
    loss_grid = torch.sum((output-target)**2)/(grid_valid_size*len(channels))

    loss_types["loss_grid"] = loss_grid.item()

    run_loss_run = torch.zeros(1).float().cuda()
    
    if channels == "all":
        num_spectral_chs = output.shape[-1]
        channels = [["_",i,1./num_spectral_chs] for i in np.arange(num_spectral_chs)]
    
    totcw = 0
    
    ## for valid periodic fft
    ## times x lat x lon x channels
    output2lat = torch.cat([output, torch.flip(output,[1])],dim=1)
    output2lon = torch.cat([output, torch.flip(output,[2])],dim=2)
    target2lat = torch.cat([target, torch.flip(target,[1])],dim=1)
    target2lon = torch.cat([target, torch.flip(target,[2])],dim=2)
    
    for [cname,c,cw] in channels:
        ## it makes sense for me to take fft differences before...if you take mean, you lose more important differences at the equator?
        ## losses from fft, both lat (index 1) and lon (index 2) directions
        if cw != 0:
            ## lat
            out_fft_lat = torch.abs(torch.fft.rfft(output2lat[:,:,:,c],dim=1))[:,wavenum_init_lon:,:]
            target_fft_lat = torch.abs(torch.fft.rfft(target2lat[:,:,:,c],dim=1))[:,wavenum_init_lon:,:]
            loss_fft_lat = torch.mean((out_fft_lat - target_fft_lat)**2)
            ## lon
            out_fft_lon = torch.abs(torch.fft.rfft(output2lon[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
            target_fft_lon = torch.abs(torch.fft.rfft(target2lon[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
            loss_fft_lon = torch.mean((out_fft_lon - target_fft_lon)**2)

            run_loss_run += ((1-lat_lon_bal)*loss_fft_lon + lat_lon_bal*loss_fft_lat)*cw
            totcw+=cw

            loss_types[f"loss_fft_lat_{cname}"] = loss_fft_lat.item()
            loss_types[f"loss_fft_lon_{cname}"] = loss_fft_lon.item()
    
    ## fft_loss_scale included, so the lambda_fft is more intuitive (lambda_fft = .5 --> around half weighted shared between grid and fft loss)
    loss_fft = run_loss_run/totcw*fft_loss_scale
    loss_types[f"loss_fft"] = loss_fft.item()
    loss_fft_weighted = lambda_fft*loss_fft
    loss_grid_weighted = ((1-lambda_fft))*loss_grid
    loss = loss_grid_weighted + loss_fft_weighted
    
    # print(loss_types)
    # return loss, loss_grid, loss_fft
    return loss

def spectral_sqr_lonMean(output, 
                               target, 
                               grid_valid_size = None,
                               wavenum_init_lon = 1, 
                               wavenum_init_lat = 1, 
                               lambda_fft = .5,
                               lat_lon_bal = .5,
                               channels = "all",
                               fft_loss_scale = 1./110.):
        
    """
    Grid and spectral losses, both with mse
    """
    # loss1 = torch.sum((output-target)**2)/ocean_grid + torch.sum((output2-target2)**2)/ocean_grid
    
    ## loss from grid space
    if grid_valid_size is None: 
        grid_valid_size = output.flatten().shape[0]
        
    loss_grid = torch.sum((output-target)**2)/(grid_valid_size*len(channels))
    # loss_grid = torch.mean((output-target)**2)
    # loss1 = torch.abs((output-tnparget))/ocean_grid

    run_loss_run = torch.zeros(1).float().cuda()
    
    if channels == "all":
        num_spectral_chs = output.shape[-1]
        channels = [["_",i,1./num_spectral_chs] for i in np.arange(num_spectral_chs)]
    
    totcw = 0
    for [cname,c,cw] in channels:
        ## it makes sense for me to take fft differences before...if you take mean, you lose more important differences at the equator?
        ## losses from fft, both lat (index 1) and lon (index 2) directions
        ## lon
        out_fft_lon = torch.abs(torch.fft.rfft(output[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
        out_fft_lon = torch.mean(out_fft_lon,dim=1)
        target_fft_lon = torch.abs(torch.fft.rfft(target[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
        target_fft_lon = torch.mean(target_fft_lon,dim=1)
        
        loss_fft_lon = torch.mean((out_fft_lon - target_fft_lon)**2)

        run_loss_run += loss_fft_lon*cw
        totcw += cw
        
    ## fft_loss_scale included, so the lambda_fft is more intuitive (lambda_fft = .5 --> around half weighted shared between grid and fft loss)
    loss_fft = run_loss_run/totcw*fft_loss_scale
    loss_fft_weighted = lambda_fft*loss_fft
    loss_grid_weighted = ((1-lambda_fft))*loss_grid
    loss = loss_grid_weighted + loss_fft_weighted
    
    # return loss, loss_grid, loss_fft
    return loss

def spectral_sqr_phase(output, 
                       target, 
                       grid_valid_size = None,
                       wavenum_init_lon = 1, 
                       wavenum_init_lat = 1, 
                       lambda_fft = .5,
                       lat_lon_bal = .5,
                       channels = "all",
                       fft_loss_scale = 1./110.):
        
    """
    Grid and spectral losses, both with mse
    Takes into account sinusoidal phase as well, as opposed to complex norm
    """

    ## loss from grid space
    if grid_valid_size is None: 
        grid_valid_size = output.flatten().shape[0]
        
    loss_grid = torch.sum((output-target)**2)/(grid_valid_size*len(channels))

    run_loss_run = torch.zeros(1).float().cuda()
    
    if channels == "all":
        num_spectral_chs = output.shape[-1]
        channels = [["_",i,1./num_spectral_chs] for i in np.arange(num_spectral_chs)]
    
    totcw = 0
    for [cname,c,cw] in channels:
        ## it makes sense for me to take fft differences before...if you take mean, you lose more important differences at the equator?
        ## losses from fft, both lat (index 1) and lon (index 2) directions
        ## lat
        out_fft_lat = torch.fft.rfft(output[:,:,:,c],dim=1)[:,wavenum_init_lon:,:]
        target_fft_lat = torch.fft.rfft(target[:,:,:,c],dim=1)[:,wavenum_init_lon:,:]
        loss_fft_lat = torch.mean(torch.abs(out_fft_lat - target_fft_lat)**2)
        ## lon
        out_fft_lon = torch.fft.rfft(output[:,:,:,c],dim=2)[:,:,wavenum_init_lon:]
        target_fft_lon = torch.fft.rfft(target[:,:,:,c],dim=2)[:,:,wavenum_init_lon:]
        loss_fft_lon = torch.mean(torch.abs(out_fft_lon - target_fft_lon)**2)

        run_loss_run += ((1-lat_lon_bal)*loss_fft_lon + lat_lon_bal*loss_fft_lat)*cw
        totcw+=cw
        
    ## fft_loss_scale included, so the lambda_fft is more intuitive (lambda_fft = .5 --> around half weighted shared between grid and fft loss)
    loss_fft = run_loss_run/totcw*fft_loss_scale
    loss_fft_weighted = lambda_fft*loss_fft
    loss_grid_weighted = ((1-lambda_fft))*loss_grid
    loss = loss_grid_weighted + loss_fft_weighted
    
    # return loss, loss_grid, loss_fft
    return loss

## takes incredibly long with forecasting to train
def RK4step(net, input_batch):
    output_1 = net(input_batch.cuda())
    output_2 = net(input_batch.cuda()+0.5*output_1)
    output_3 = net(input_batch.cuda()+0.5*output_2)
    output_4 = net(input_batch.cuda()+output_3)

    return input_batch.cuda() + (output_1+2*output_2+2*output_3+output_4)/6
 
def Eulerstep(net, input_batch, delta_t = 1.0):
    output_1 = net(input_batch.cuda())
    return input_batch.cuda() + delta_t*(output_1)

def Eulerstep_reg(net, input_batch, delta_t = 1.0):
    output_1 = net(input_batch.cuda())
    return input_batch.cuda() + delta_t*(output_1)

def PECstep(net, input_batch, delta_t = 1.0):
    output_net = net(input_batch.cuda())
    #assert output_net.shape == input_batch.shape 
    output_1 = delta_t*output_net + input_batch.cuda() ## delta_t variances, to force jacobian of output to have smaller eigenvalues -> store matrix for single timestep, the can do eigenvalue decomp on bigger cluster.
    ## torch will return tensor rank 6 -> reshape to rank 2 (128x128x3)x(128x128x3)
    #print(net(input_batch.cuda()).shape, input_batch.cuda().shape)
    return input_batch.cuda() + delta_t*0.5*(net(input_batch.cuda())+net(output_1))

def directstep(net, input_batch):
    output_1 = net(input_batch.cuda())
    return output_1        

STEPS_SCHEMES = {
                "directstep" : directstep,
                "RK4step" : RK4step,
                "PECstep" : PECstep, 
                "Eulerstep" : Eulerstep,
               }

LOSS_FUNCTIONS = {
                  "ocean" : ocean_loss,
                  "spectral_abs" : spectral_sqr_abs,
                  "spectral_abs2" : spectral_sqr_abs2,
                  "spectral_phase" : spectral_sqr_phase,
                  "spectral_abs_lonMean" : spectral_sqr_lonMean,
                 }
                 
MODEL_ARCHITECTURES = {
                      "FNO2D" : FNO2d,
                      "CNN2D" : CNN2D,
                      "FNO2D_grid" : FNO2D_grid,
                      "FNO2D_grid_pad" : FNO2D_grid_pad,
                      "FNO2D_grid_pad2" : FNO2D_grid_pad2,
                      "Unet2" : UNET2,
                      }
# loss testing
# input = torch.from_numpy(lr[[0]]).float().cuda()
# target = torch.from_numpy(lr[[1]]).float().cuda()
# output = torch.from_numpy(lr[[10]]).float().cuda()
# out_fft_lon = torch.abs(torch.fft.rfft(output[:,:,:,0],dim=2))[:,:,1:]
# out_fft_lon = torch.mean(out_fft_lon,dim=1)
# target_fft_lon = torch.abs(torch.fft.rfft(target[:,:,:,0],dim=2))[:,:,1:]
# target_fft_lon = torch.mean(target_fft_lon,dim=1)