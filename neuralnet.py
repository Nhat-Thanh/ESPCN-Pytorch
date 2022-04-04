import torch
import torch.nn as nn

class ESPCN_model(nn.Module):
    def __init__(self, scale : int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_1.bias)

        self.tanh = nn.Tanh()

        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_2.bias)

        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=(3 * scale * scale), kernel_size=3, padding=1)
        nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        nn.init.zeros_(self.conv_3.bias)

        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, X_in):
        X = self.tanh(self.conv_1(X_in))
        X = self.tanh(self.conv_2(X))
        X = self.conv_3(X)
        X = self.pixel_shuffle(X)
        X_out = torch.clip(X, 0.0, 1.0)
        return X_out
