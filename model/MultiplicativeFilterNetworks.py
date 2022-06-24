import numpy as np
import skimage
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from typing import List
from tqdm import tqdm

class SineLayer(nn.Module):
    """
    SineLayer Implementation
    """

    def __init__(self, omega_0):
        super(SineLayer, self).__init__()
        self.omega_0 = torch.tensor(omega_0)

    def forward(self, input):
        self._check_input(input)
        return torch.sin(self.omega_0 * input)

    @staticmethod
    def _check_input(input):
        if not isinstance(input, torch.Tensor):
            raise TypeError("input must be a torch.xTensor")

class SIREN(nn.Module):
    """
    SIREN main class implementation
    
    ...............................
    :param layers: list of number of neurons in each hidden layer
    :type layers: List[int]
    :param 
    : number of input features
    :type in_features: int
    :param out_features: number of final output features
    :type out_features: int
    :param w0: w0 in the activation step `act(x; omega_0) = sin(omega_0 * x)`.
              defaults to 1.0
    :type w0: float, optional
    :param w0_initial: `w0` of first layer. defaults to 30 (as used in the
              paper)
    :type w0_initial: float, optional
    """
    def __init__(
        self,
        layers: List[int],
        omega_0_initial=38,
        omega_0=1,
        in_features=2,
        hidden_features=256,
        out_features=1,
    ):
        super(SIREN, self).__init__()

        self.layers = [
            nn.Linear(in_features, layers[0]),
            SineLayer(omega_0=omega_0_initial),
        ]

        for index in range(len(layers) - 1):
            self.layers.extend(
                [
                    nn.Linear(layers[index], layers[index + 1]),
                    SineLayer(omega_0=omega_0),
                ]
            )

        self.layers.append(nn.Linear(layers[-1], out_features))
        self.net = nn.Sequential(*self.layers)

    def forward(self, input):
        return self.net(input)

class GaborLayer(nn.Module):
    def __init__(self, in_dim, out_dim, padding, alpha, beta=1.0, bias=False):
        super(GaborLayer, self).__init__()

        self.mu = nn.Parameter(torch.rand((out_dim, in_dim)) * 2 - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_dim, )))
        self.linear = torch.nn.Linear(in_dim, out_dim)
        #self.padding = padding

        self.linear.weight.data *= 128. * torch.sqrt(self.gamma.unsqueeze(-1))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

        # Bias parameters start in zeros
        #self.bias = nn.Parameter(torch.zeros(self.responses)) if bias else None

    def forward(self, input):
        norm = (input ** 2).sum(dim=1).unsqueeze(-1) + (self.mu ** 2).sum(dim=1).unsqueeze(0) - 2 * input @ self.mu.T
        return torch.exp(- self.gamma.unsqueeze(0) / 2. * norm) * torch.sin(self.linear(input))


class GaborNet(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=256, out_dim=1, k=4):
        super(GaborNet, self).__init__()

        self.k = k
        self.gabon_filters = nn.ModuleList([GaborLayer(in_dim, hidden_dim, 0, alpha=6.0 / k) for _ in range(k)])
        self.linear = nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(k - 1)] + [torch.nn.Linear(hidden_dim, out_dim)])

        for lin in self.linear[:k - 1]:
            lin.weight.data.uniform_(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim))

    def forward(self, x):

        # Recursion - Equation 3
        zi = self.gabon_filters[0](x)  # Eq 3.a
        for i in range(self.k - 1):
            zi = self.linear[i](zi) * self.gabon_filters[i + 1](x)  # Eq 3.b

        return self.linear[self.k - 1](zi)  # Eq 3.c


def train(model, optim, nb_epochs=15000):
    psnrs = []
    for _ in tqdm(range(nb_epochs)):
        model_output = model(pixel_coordinates)
        loss = ((model_output - pixel_values) ** 2).mean()
        psnrs.append(20 * np.log10(1.0 / np.sqrt(loss.item())))

        optim.zero_grad()
        loss.backward()
        optim.step()

    return psnrs, model_output


if __name__ == "__main__":
    device = 'cuda'
    siren = SIREN(layers=[256, 256, 256, 256, 256]).to(device)
    gabor_net = GaborNet().to(device)

    # Target
    img = ((torch.from_numpy(skimage.data.grass()) - 127.5) / 127.5)
    img = torchvision.transforms.Resize(256)(img.unsqueeze(0))[0]
    pixel_values = img.reshape(-1, 1).to(device)

    # Input
    resolution = img.shape[0]
    tmp = torch.linspace(-1, 1, steps=resolution)
    x, y = torch.meshgrid(tmp, tmp)
    pixel_coordinates = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).to(device)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Ground Truth', fontsize=13)

    for i, model in enumerate([siren, gabor_net]):
        # Training
        optim = torch.optim.Adam(lr=1e-4 if (i == 0) else 1e-2, params=model.parameters())
        psnrs, model_output = train(model, optim, nb_epochs=1000)

        axes[i + 1].imshow(model_output.cpu().view(resolution, resolution).detach().numpy(), cmap='gray')
        axes[i + 1].set_title('SIREN' if (i == 0) else 'GaborNet', fontsize=13)
        axes[4].plot(psnrs, label='SIREN' if (i == 0) else 'GaborNet', c='green' if (i == 0) else 'purple')
        axes[4].set_xlabel('Iterations', fontsize=14)
        axes[4].set_ylabel('PSNR', fontsize=14)
        axes[4].legend(fontsize=13)

    for i in range(4):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    axes[3].axis('off')
    plt.savefig('Grass.png')
    plt.close()
