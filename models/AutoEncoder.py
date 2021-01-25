import torch
from torch import nn


class AutoEncoder(nn.Module):
	def __init__(self):
		super(AutoEncoder, self).__init__()
		# self.l1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
		# self.l2 = nn.ReLU(True)
		# self.l3 = nn.Conv2d(6, 16, kernel_size=5)
		# self.l4 = nn.ReLU(True)

		self.encoder = nn.Sequential(
			nn.Conv2d(1, 10, kernel_size=5),
			nn.ReLU(True),
			nn.Conv2d(10, 25, kernel_size=5),
			nn.ReLU(True))
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(25, 10, kernel_size=5),
			nn.ReLU(True),
			nn.ConvTranspose2d(10, 1, kernel_size=5),
			nn.ReLU(True))

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x
