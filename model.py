import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1),
            nn.BatchNorm2d(n_coefficients)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1),
            nn.BatchNorm2d(n_coefficients)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out

class Model(nn.Module):
    def __init__(self, img_ch=3, output_ch=19):
        super(Model, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Doubling pattern in encoder
        self.Conv1 = ConvBlock(img_ch, 16)
        self.Conv2 = ConvBlock(16, 32)
        self.Conv3 = ConvBlock(32, 64)
        self.Conv4 = ConvBlock(64, 128)
        self.Conv5 = ConvBlock(128, 256)
        self.Conv6 = ConvBlock(256, 512)
        self.Conv7 = ConvBlock(512, 1024)
        self.Conv8 = ConvBlock(1024, 2048)

        # Halving pattern in decoder
        self.Up8 = UpConv(2048, 1024)
        self.Att8 = AttentionBlock(F_g=1024, F_l=1024, n_coefficients=512)
        self.UpConv8 = ConvBlock(2048, 1024)

        self.Up7 = UpConv(1024, 512)
        self.Att7 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv7 = ConvBlock(1024, 512)

        self.Up6 = UpConv(512, 256)
        self.Att6 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv6 = ConvBlock(512, 256)

        self.Up5 = UpConv(256, 128)
        self.Att5 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv5 = ConvBlock(256, 128)

        self.Up4 = UpConv(128, 64)
        self.Att4 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv4 = ConvBlock(128, 64)

        self.Up3 = UpConv(64, 32)
        self.Att3 = AttentionBlock(F_g=32, F_l=32, n_coefficients=16)
        self.UpConv3 = ConvBlock(64, 32)

        self.Up2 = UpConv(32, 16)
        self.Att2 = AttentionBlock(F_g=16, F_l=16, n_coefficients=8)
        self.UpConv2 = ConvBlock(32, 16)

        self.FinalConv = nn.Conv2d(16, output_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.Conv1(x)
        
        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)
        
        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)
        
        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)
        
        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)
        
        e6 = self.MaxPool(e5)
        e6 = self.Conv6(e6)
        
        e7 = self.MaxPool(e6)
        e7 = self.Conv7(e7)
        
        e8 = self.MaxPool(e7)
        e8 = self.Conv8(e8)
        
        # Start decoder
        d8 = self.Up8(e8)
        s7 = self.Att8(gate=d8, skip_connection=e7)
        d8 = torch.cat((s7, d8), dim=1)
        d8 = self.UpConv8(d8)
        
        d7 = self.Up7(d8)
        s6 = self.Att7(gate=d7, skip_connection=e6)
        d7 = torch.cat((s6, d7), dim=1)
        d7 = self.UpConv7(d7)
        
        d6 = self.Up6(d7)
        s5 = self.Att6(gate=d6, skip_connection=e5)
        d6 = torch.cat((s5, d6), dim=1)
        d6 = self.UpConv6(d6)
        
        d5 = self.Up5(d6)
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1)
        d5 = self.UpConv5(d5)
        
        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)
        
        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)
        
        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)
        
        out = self.FinalConv(d2)

        return out