from torch import nn, optim


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.layers(x)


class Generator(nn.Module):
    def __init__(self, repeat_num=4):
        super(Generator, self).__init__()

        layers = list()
        layers.append(nn.Conv2d(3, 64, kernel_size=9, padding=4, bias=True))
        layers.append(nn.ReLU())

        for _ in range(repeat_num):
            layers.append(ResidualBlock(64))

        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(64, 3, kernel_size=9, padding=4, bias=True))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) * 0.58 + 0.5


class Discriminator(nn.Module):
    def __init__(self, in_channels, image_size=100):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=11, padding=5, stride=4, bias=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(48, 128, kernel_size=5, padding=2, stride=2, bias=True),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 192, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(192, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm2d(192, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(192, 128, kernel_size=3, padding=1, stride=2, bias=True),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2)
        )

        feature_size = (image_size + 15) // 16
        in_features = 128 * feature_size * feature_size

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)
        return self.fc_layers(h)


class WESPE(nn.Module):
    def __init__(self, config):
        super(WESPE, self).__init__()

        self.is_train = config.is_train
        self.gen_g = Generator()

        if self.is_train:
            self.gen_f = Generator()
            self.dis_c = Discriminator(in_channels=3)
            self.dis_t = Discriminator(in_channels=1)

            self.mse_criterion = nn.MSELoss()
            self.bce_criterion = nn.BCEWithLogitsLoss()

            gen_params = list(self.gen_g.parameters()) + list(self.gen_f.parameters())
            dis_params = list(self.dis_c.parameters()) + list(self.dis_t.parameters())
            self.gen_optimizer = optim.Adam(gen_params, lr=config.gen_lr)
            self.dis_optimizer = optim.Adam(dis_params, lr=config.dis_lr)

    def forward(self, x):
        y = self.gen_g(x)
        if self.is_train:
            x_rec = self.gen_f(y)
            return y, x_rec
        else:
            return y, None
