import torch
import torch.nn as nn


class ConvBlock(nn.Module):
	def __init__(self, in_channels, num_filters):
		super(ConvBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(num_filters)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(num_filters)
		self.relu2 = nn.ReLU(inplace=True)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu2(x)
		return x












