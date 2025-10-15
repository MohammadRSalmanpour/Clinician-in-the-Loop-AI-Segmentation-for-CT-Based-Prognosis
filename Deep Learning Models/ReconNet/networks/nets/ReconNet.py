import torch
import torch.nn as nn

class ReconNet(nn.Module):

    def initial_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 2, 3, padding=1),
            nn.BatchNorm3d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def consecutive_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def consecutive_conv_up(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def __init__(self, num_channels, num_latents):
        super(ReconNet, self).__init__()

        print('num_channels:', num_channels)
        print('num_latents:', num_latents)

        self.conv_initial = self.initial_conv(1, num_channels)
        self.conv_final = nn.Conv3d(num_channels, 1, 3, padding=1)

        self.conv_rest_x_64 = self.consecutive_conv(num_channels, num_channels * 2)
        self.conv_rest_x_32 = self.consecutive_conv(num_channels * 2, num_channels * 4)
        self.conv_rest_x_16 = self.consecutive_conv(num_channels * 4, num_channels * 8)

        self.conv_rest_u_32 = self.consecutive_conv_up(num_channels * 8 + num_channels * 4, num_channels * 4)
        self.conv_rest_u_64 = self.consecutive_conv_up(num_channels * 4 + num_channels * 2, num_channels * 2)
        self.conv_rest_u_128 = self.consecutive_conv_up(num_channels * 2 + num_channels, num_channels)

        self.contract = nn.MaxPool3d(2, stride=2)
        self.expand = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x_128 = self.conv_initial(x)  # conv_initial 1->num_channels
        x_64 = self.contract(x_128)
        x_64 = self.conv_rest_x_64(x_64)  # rest num_channels->num_channels*2
        x_32 = self.contract(x_64)
        x_32 = self.conv_rest_x_32(x_32)  # rest num_channels*2->num_channels*4
        x_16 = self.contract(x_32)
        x_16 = self.conv_rest_x_16(x_16)  # rest num_channels*4->num_channels*8

        u_32 = self.expand(x_16)
        u_32 = self.conv_rest_u_32(torch.cat((x_32, u_32), 1))  # rest num_channels*8+num_channels*4->num_channels*4
        u_64 = self.expand(u_32)
        u_64 = self.conv_rest_u_64(torch.cat((x_64, u_64), 1))  # rest num_channels*4+num_channels*2->num_channels*2
        u_128 = self.expand(u_64)
        u_128 = self.conv_rest_u_128(torch.cat((x_128, u_128), 1))  # rest num_channels*2+num_channels->num_channels
        u_128 = self.conv_final(u_128)

        return u_128

if __name__ == '__main__':
    x = torch.randn((1, 1, 64, 64, 64))
    model = ReconNet(32, 1)
    x = model(x)
    print(x.shape)
