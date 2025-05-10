import torch
import torch.nn as nn
import torch.optim as optim
import math

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding='same')
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, 
                 in_channels = 2, 
                 out_channels = 2, 
                #  down_channels = (128, 128, 128, 128, 128), 
                 down_channels = (128, 128, 128, 128), 
                #  up_channels = (128, 128, 128, 128, 128), 
                 up_channels = (128, 128, 128, 128), 
                 time_emb_dim = 64):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # down_channels = (64, 64, 64, 64, 64)
        # up_channels = (64, 64, 64, 64, 64)
        # down_channels = (128, 128, 128, 128, 128)
        # up_channels = (128, 128, 128, 128, 128)
        # # time_emb_dim = 32
        # time_emb_dim = 64
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels+3, down_channels[0], 3, padding='same') # 3 beccause 2 hres + 1 orography

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        
        self.circular_padding = torch.nn.CircularPad2d((0, 0, 3, 4))
        self.conv_reshape = nn.Conv2d(up_channels[-1], 
                                      up_channels[-1], 
                                      kernel_size=(8,1), 
                                      stride=(1,1), 
                                      padding=(0,0)) # From 728x1440 to 721x1440


        self.output = nn.Conv2d(up_channels[-1], out_channels, 1)

    def forward(self, x, timestep, cond=None):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.circular_padding(x)
        cond = self.circular_padding(cond)
        x = torch.cat((x, cond), dim=1)
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        x = self.conv_reshape(x) # Reshape it back to 721x1440
        return self.output(x)

class SimpleUnet2(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, 
                 in_channels = 3, 
                 out_channels = 3, 
                 down_channels = (128, 128, 128, 128, 128), 
                 up_channels = (128, 128, 128, 128, 128), 
                 time_emb_dim = 64):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding='same')

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])


        self.output = nn.Conv2d(up_channels[-1], out_channels, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)

