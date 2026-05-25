import torch
import torch.nn as nn


class FSRCNN(nn.Module):
    """FSRCNN with configurable upsampling backend.

    Parameters
    ----------
    upsample_mode : str
        ``'deconv'``  – original ConvTranspose2d (default, used during training)
        ``'pixelshuffle'`` – Conv2d + PixelShuffle (ONNX/Web-friendly)

    The two modes have **different weight layouts**, so a conversion step is
    required when switching from ``'deconv'`` to ``'pixelshuffle'`` after
    training.  See ``scripts/convert_fsrcnn_to_pixelshuffle.py``.
    """

    def __init__(self, scale=2, in_channels=3, d=56, s=12, m=4,
                 upsample_mode='deconv'):
        super().__init__()
        self.scale = scale
        self.upsample_mode = upsample_mode

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels, d, 5, padding=2),
            nn.PReLU(d)
        )
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, 1),
            nn.PReLU(s)
        )
        self.mapping = nn.Sequential(
            *sum([[nn.Conv2d(s, s, 3, padding=1), nn.PReLU(s)] for _ in range(m)], [])
        )
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, 1),
            nn.PReLU(d)
        )

        if upsample_mode == 'deconv':
            self.upsample = nn.ConvTranspose2d(
                d, in_channels, 9, stride=scale, padding=4,
                output_padding=scale - 1
            )
        elif upsample_mode == 'pixelshuffle':
            self.upsample = nn.Sequential(
                nn.Conv2d(d, in_channels * scale * scale, 9, padding=4),
                nn.PixelShuffle(scale)
            )
        else:
            raise ValueError(f"Unknown upsample_mode: {upsample_mode}")

        self._initialize_weights()

    def forward(self, x):
        x = self.feature(x)
        x = self.shrink(x)
        x = self.mapping(x)
        x = self.expand(x)
        return self.upsample(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.PReLU):
                nn.init.constant_(m.weight, 0.25)

