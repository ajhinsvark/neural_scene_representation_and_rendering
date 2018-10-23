import torch
import torch.nn as nn

class Core(nn.Module):

    def __init__(self, in_channels):
        super(Core, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.lstm_i = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1)
        self.lstm_f = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1)
        self.lstm_tanh= nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1)
        self.lstm_o = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1)
        self.deconv_h = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=4)


    def forward(self, prev_hg, prev_cg, prev_u, prev_z, v, r):
        # NCHW
        lstm_in = torch.concat([prev_hg, v, r, prev_z], axis=1)
        
        forget_gate = self.sigmoid(self.lstm_i(lstm_in))
        input_gate = self.sigmoid(self.lstm_f(lstm_in))

        next_c = prev_cg * forget_gate + input_gate * self.tanh(self.lstm_tanh(lstm_in))
        next_h = self.sigmoid(self.lstm_o(lstm_in)) * self.tanh(next_c)
        next_u = self.deconv_h(next_h) + prev_u

        return next_h, next_c, next_u

class Prior(nn.Module):
    def __init__(self, channels_h, channels_z):
        super(Prior, self).__init__()
        self.mean_z = nn.Conv2d(channels_h, channels_z, kernel_size=5, stride=1, padding=2, bias=False)
        nn.init.kaiming_normal(self.mean_z.weight, a=0.1)
        
        self.ln_var_z = nn.Conv2d(channels_h, channels_z, kernel_size=5, stride=1, padding=2, bias=False)
        nn.init.kaiming_normal(self.ln_var_z.weight, a=0.1)

    def compute_mean_z(self, h):
        return self.mean_z(h)

    def compute_ln_var_z(self, h):
        return self.ln_var_z(h)

    def sample_z(self, h):
        mean = self.compute_mean_z(h)
        ln_var = self.compute_ln_var_z(h)
        return torch.normal(mean, ln_var)


class ObservationDistribution(nn.Module):
    def __init__(self, channels_u):
        super(ObservationDistribution, self).__init__()
        self.mean_x = nn.Conv2d(channels_u, 3, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.kaiming_normal(self.mean_x.weight, a=0.1)

    def compute_mean_x(self, u):
        return self.mean_x(u)

    def sample_x(self, u, ln_var):
        mean = self.compute_mean_x(u)
        return torch.normal(mean, ln_var)

class Generative(nn.Module):

    def __init__(self):
        super(Generative, self).__init__()

        self.core1 = Core(100)
        

    def forward(self, inputs):
        self.core(torch.zeros())