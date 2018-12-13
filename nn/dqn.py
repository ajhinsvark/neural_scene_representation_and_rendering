import torch
import torch.nn as nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DQN(nn.Module):
    """
    Tower Architecture of DQN
    """

    def __init__(self, architecture='pyramid'):
        super(DQN, self).__init__()

        self.architecture = architecture

        if self.architecture == 'tower':
            self.relu = nn.ReLU()

            self.res1 = nn.Conv2d(256, 256, kernel_size=1, stride=2, bias=False)
            self.bnr_1 = nn.BatchNorm2d(256)
            self.res2 = nn.Conv2d(256+7, 256, kernel_size=1, stride=1, bias=False)
            self.bnr_2 = nn.BatchNorm2d(256)
            

            self.conv1_1 = nn.Conv2d(3, 256, kernel_size=2, stride=2, bias=False)
            self.bn1 = nn.BatchNorm2d(256)
            self.conv1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv1_3 = nn.Conv2d(128, 256, kernel_size=2, stride=2, bias=False)
            self.bn3 = nn.BatchNorm2d(256)

            self.conv2_1 = nn.Conv2d(256+7, 128, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2_1 = nn.BatchNorm2d(128)
            self.conv2_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2_2 = nn.BatchNorm2d(256)
            self.conv2_3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
            self.bn2_3 = nn.BatchNorm2d(256)
        elif self.architecture == 'pyramid':
            
            self.conv1 = nn.Sequential(
                nn.Conv2d(10,32, kernel_size=2, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(32,64, kernel_size=2, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )

            self.conv3 = nn.Sequential(
                nn.Conv2d(64,128, kernel_size=2, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )

            self.conv4 = nn.Sequential(
                nn.Conv2d(128,256, kernel_size=8, stride=8),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        
    def _tower(self, views, frames):
        views = views.permute([1, 0, 2])
        frames = frames.permute([1, 0, 4, 2, 3])
        
        out = torch.zeros([len(views[0]), 256, 16, 16]).to(device)
        
        for i in range(len(views)):
            view = views[i].repeat([16, 16, 1, 1]).permute([2,3,0,1])
            frame = frames[i]

            x = self.conv1_1(frame)
            x = self.bn1(x)
            x = self.relu(x)

            residual = x
            x = self.conv1_2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.conv1_3(x)
            x = self.bn3(x)
            
            residual = self.res1(residual)
            residual = self.bnr_1(residual)
            x += residual
            x = self.relu(x)

            x = torch.cat([x, view], dim=1)

            residual = x

            x = self.conv2_1(x)
            x = self.bn2_1(x)
            x = self.relu(x)
            x = self.conv2_2(x)
            x = self.bn2_2(x)

            residual = self.res2(residual)
            residual = self.bnr_2(residual)
            x += residual
            x = self.relu(x)

            x = self.conv2_3(x)
            x = self.bn2_3(x)
            x = self.relu(x)
            out += x

        return out
    
    def _pyramid(self, views, frames):
        batch, context, view_size = views.shape
        _, _, h, w, chans = frames.shape
        
        views = views[:,:,None,None,:].repeat([1,1,h,w,1]).permute([0, 1, 4, 2, 3])         # Batch Context length, h w
        frames = frames.permute([0, 1, 4, 2, 3]) # Batch Context Chan, H, W

        contexts = torch.cat((views, frames), dim=2)  # concat along the channels
        
        contexts = contexts.view(batch * context, chans + view_size, h, w) # reshape for convolutions

        contexts = self.conv1(contexts)
        contexts = self.conv2(contexts)
        contexts = self.conv3(contexts)
        contexts = self.conv4(contexts)

        contexts = contexts.view(batch, context, 256, 1, 1)

        out = torch.sum(contexts, dim=1)

        return out
        # Sum all representations
        # out = torch.zeros([len(views[0]), 256, 1, 1]).to(device)
        # for i in range(len(views)):
        #     view = views[i].repeat([64, 64, 1, 1]).permute([2,3,0,1])
        #     frame = frames[i]

        #     x = torch.cat([view, frame], dim=1)

        #     x = self.conv1(x)
        #     # print(x.shape)
        #     x = self.conv2(x)
        #     # print(x.shape)
        #     x = self.conv3(x)
        #     # print(x.shape)
        #     x = self.conv4(x)

        #     out += x



    def forward(self, views, frames):
        if self.architecture == 'tower':
            return self._tower(views, frames)
        else:
            return self._pyramid(views, frames)
