import torch
from torch import nn

class VAEDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

        self.fcMu = nn.Linear(256, 256)
        self.fcLogVar = nn.Linear(256, 256)

        self.deconv1 = nn.ConvTranspose2d(256+7, 128, kernel_size=16, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv1   = nn.Conv2d(512, 3, kernel_size=1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def get_normal_samples(self, encoding):
        # Expects input to be from pyramid 1 x 1 x 256
        flat_encoding = encoding.view(-1, 256)
        mu = self.fcMu(flat_encoding)
        logvar = self.fcMu(flat_encoding)
        return self.reparameterize(mu, logvar)
    
    def decode(self, z):
        deconv_input = z.view(-1,256+7, 1, 1)

        x = self.relu(self.bn1(self.deconv1(deconv_input)))
        # print("128, 16, 16", x.shape)
        x = self.relu(self.bn2(self.deconv2(x)))
        # print("512, 32, 32", x.shape)
        x = self.relu(self.bn3(self.deconv3(x)))
        # print("512, 64, 64", x.shape)
        x = self.relu(self.conv1(x))

        return x
        
    def forward(self, representation, view):
        z = self.get_normal_samples(representation)
        decode_input = torch.cat([z, view], dim=1)
        image = self.decode(decode_input)
        image = image.permute([0, 2, 3, 1])
        return image