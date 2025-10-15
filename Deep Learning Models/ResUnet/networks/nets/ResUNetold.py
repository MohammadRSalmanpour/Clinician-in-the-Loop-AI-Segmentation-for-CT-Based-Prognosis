import torch
import torch.nn as nn
import torch.nn.functional as F

class ResUNet(nn.Module):
    def __init__(self, training=True):
        super().__init__()

        self.training = training
        self.dorp_rate = 0.2

        # Encoder stages
        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(1, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(),

            nn.Conv3d(32, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(),

            nn.Conv3d(64, 128, 2, 2),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(),

            nn.Conv3d(128, 256, 2, 2),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(),

            nn.Conv3d(256, 512, 2, 2),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(),
        )

        # Decoder stages
        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(512, 256, 3, 1, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(256 + 256, 128, 3, 1, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(128 + 128, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(64 + 64, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(),
        )

        # Downsampling layers
        self.down_conv1 = nn.Sequential(
            nn.Conv3d(64, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(),
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(128, 256, 2, 2),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(),
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(256, 512, 2, 2),
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(),
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 2, 2),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(),
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(),
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(),
        )

        # 1x1x1 Convs to match channels before addition
        self.match_channels1 = nn.Conv3d(128, 64, 1, 1)
        self.match_channels2 = nn.Conv3d(64, 128, 1, 1)  # Adjust channels before encoder_stage3
        self.match_channels3 = nn.Conv3d(128, 256, 1, 1)  # Adjust channels before addition with long_range3
        self.match_channels4 = nn.Conv3d(512, 256, 1, 1)  # Adjust channels before addition with long_range4

        # New layer to adjust channels before decoder_stage1
        self.adjust_channels_for_decoder1 = nn.Conv3d(512, 256, 1, 1)

    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs)
        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1)
        
        # Adjust the number of channels of long_range2 to match short_range1
        long_range2 = self.match_channels1(long_range2)
        
        # Ensure spatial dimensions match
        if long_range2.shape[2:] != short_range1.shape[2:]:
            long_range2 = F.interpolate(long_range2, size=short_range1.shape[2:], mode='trilinear', align_corners=True)
        
        long_range2 = long_range2 + short_range1
        long_range2 = F.dropout(long_range2, self.dorp_rate, self.training)

        # Adjust channels before passing to encoder_stage3
        long_range2 = self.match_channels2(long_range2)

        # Pass through encoder_stage3
        long_range3 = self.encoder_stage3(long_range2)

        # Adjust channels of long_range2 to match long_range3
        long_range2 = self.match_channels3(long_range2)

        # Ensure spatial dimensions match before adding long_range2 to long_range3
        if long_range3.shape[2:] != long_range2.shape[2:]:
            long_range2 = F.interpolate(long_range2, size=long_range3.shape[2:], mode='trilinear', align_corners=True)
        
        long_range3 = long_range3 + long_range2
        
        long_range3 = F.dropout(long_range3, self.dorp_rate, self.training)
        short_range3 = self.down_conv3(long_range3)

        # Adjust channels of short_range3 before passing to encoder_stage4
        short_range3 = self.match_channels4(short_range3)

        # Pass through encoder_stage4
        long_range4 = self.encoder_stage4(short_range3)
        
        # Ensure spatial dimensions match before adding short_range3 to long_range4
        if long_range4.shape[2:] != short_range3.shape[2:]:
            short_range3 = F.interpolate(short_range3, size=long_range4.shape[2:], mode='trilinear', align_corners=True)
            
        # Adjust the number of channels of short_range3 to match long_range4
        conv_layer = nn.Conv3d(256, 512, 1).to(short_range3.device)
        short_range3 = conv_layer(short_range3)
        
        long_range4 = long_range4 + short_range3
        long_range4 = F.dropout(long_range4, self.dorp_rate, self.training)
        
        # Adjust the number of channels back to match decoder_stage1
        long_range4 = self.adjust_channels_for_decoder1(long_range4)

        # Decoder starts here
        outputs = self.decoder_stage1(long_range4) + long_range4
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        short_range6 = self.up_conv2(outputs)
        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1))

        # Ensure spatial dimensions match before adding short_range6 to outputs
        if outputs.shape[2:] != short_range6.shape[2:]:
            short_range6 = F.interpolate(short_range6, size=outputs.shape[2:], mode='trilinear', align_corners=True)
            
        outputs = outputs + short_range6
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        short_range7 = self.up_conv3(outputs)
        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1))
        
        # Ensure spatial dimensions match before adding short_range7 to outputs
        if outputs.shape[2:] != short_range7.shape[2:]:
            short_range7 = F.interpolate(short_range7, size=outputs.shape[2:], mode='trilinear', align_corners=True)

        outputs = outputs + short_range7
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        short_range8 = self.up_conv4(outputs)
        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1))
        
        # Ensure spatial dimensions match before adding short_range8 to outputs
        if outputs.shape[2:] != short_range8.shape[2:]:
            short_range8 = F.interpolate(short_range8, size=outputs.shape[2:], mode='trilinear', align_corners=True)
            
        outputs = outputs + short_range8

        return outputs
