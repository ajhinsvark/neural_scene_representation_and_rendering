import torch
from torch import nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class VAEDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        
        h_dim = 256 + 7
        z_dim = 32
        self.relu = nn.LeakyReLU()

        self.fcMu = nn.Linear(h_dim, z_dim)
        self.fcLogVar = nn.Linear(h_dim, z_dim)
        self.fcExpand = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256 + 7, 128, kernel_size=5, stride=2),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Conv2d(32, 3, kernel_size=1)
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar).to(device)
        eps = torch.randn_like(std).to(device)
        return eps.mul(std).add_(mu)

    def get_normal_samples(self, encoding):
        # Expects input to be from pyramid 1 x 1 x 256
        flat_encoding = encoding.view(-1, 256+7)
        mu = self.fcMu(flat_encoding)
        logvar = self.fcLogVar(flat_encoding)
        return self.reparameterize(mu, logvar)
    
    def decode(self, z):
        # deconv_input = z.view(-1,256+7, 1, 1)

        return self.decoder(z)
        
    def forward(self, representation, view):
        # print(representation.shape, view.shape)
        z = self.get_normal_samples(torch.cat((representation, view[:,:,None,None]), dim=1))
        z = self.fcExpand(z)
        decode_input = z[:,:,None,None] # torch.cat([z.view(-1, 256, 1, 1), view[:,:,None,None]], dim=1)
        image = self.decode(decode_input)
        image = image.permute([0, 2, 3, 1])
        return image
