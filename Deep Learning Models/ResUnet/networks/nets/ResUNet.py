import torch
import torch.nn as nn
import torch.nn.functional as F

class ResUNet(nn.Module):
    def __init__(self):
        super(ResUNet, self).__init__()

        # Encoder stages
        self.encoder_stage1 = self.conv_block(1, 32)
        self.encoder_stage2 = self.conv_block(32, 64)
        self.encoder_stage3 = self.conv_block(64, 128)
        self.encoder_stage4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder stages
        self.upsample1 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder_stage1 = self.conv_block(256, 256)  # No concatenation
        
        self.upsample2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder_stage2 = self.conv_block(128, 128)  # No concatenation
        
        self.upsample3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder_stage3 = self.conv_block(64, 64)    # No concatenation
        
        self.upsample4 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder_stage4 = self.conv_block(32, 32)    # No concatenation

        # Final Convolution
        self.final_conv = nn.Conv3d(32, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.encoder_stage1(x)
        e2 = self.encoder_stage2(F.max_pool3d(e1, kernel_size=2, stride=2))
        e3 = self.encoder_stage3(F.max_pool3d(e2, kernel_size=2, stride=2))
        e4 = self.encoder_stage4(F.max_pool3d(e3, kernel_size=2, stride=2))

        # Bottleneck
        bn = self.bottleneck(F.max_pool3d(e4, kernel_size=2, stride=2))

        # Decoder
        d1 = self.upsample1(bn)
        d1 = self.decoder_stage1(d1)

        d2 = self.upsample2(d1)
        d2 = self.decoder_stage2(d2)

        d3 = self.upsample3(d2)
        d3 = self.decoder_stage3(d3)

        d4 = self.upsample4(d3)
        d4 = self.decoder_stage4(d4)

        # Final Convolution
        out = self.final_conv(d4)
        return out
