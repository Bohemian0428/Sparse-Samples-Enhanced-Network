import torch
import torch.nn as nn
import math

class DenseConv(nn.Module):
	"""创建DenseNet中的一个卷积单元，包含BatchNormalization、ReLU激活和卷积层。

	Args:
		in_channels: 输入通道数。
		growth_rate: Growth rate定义了每个卷积层的输出通道数。
	"""
	def __init__(self, in_channels, growth_rate):
		super(DenseConv, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_channels)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, stride=1, padding=0)
		
		self.bn2 = nn.BatchNorm2d(4 * growth_rate)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1)
	
	def forward(self, x):
		out = self.bn1(x)
		out = self.relu1(out)
		out = self.conv1(out)
		
		out = self.bn2(out)
		out = self.relu2(out)
		out = self.conv2(out)
		return out


class DenseBlock(nn.Module):
	def __init__(self, in_channels, output_channels, growth_rate=4):
		super(DenseBlock, self).__init__()
		self.in_channels = in_channels
		self.output_channels = output_channels
		self.growth_rate = growth_rate
		
		num_layers = math.ceil((output_channels - in_channels) / growth_rate)
		
		self.layers = nn.ModuleList()
		current_channels = in_channels
		for i in range(num_layers):
			self.layers.append(DenseConv(current_channels, growth_rate))
			current_channels += growth_rate
		
		# 添加一个1x1卷积来调整通道数到指定的output_channels
		self.transition = nn.Conv2d(current_channels, output_channels, kernel_size=1, bias=False)
	
	def forward(self, x):
		features = [x]
		for layer in self.layers:
			new_features = layer(torch.cat(features, dim=1))
			features.append(new_features)
		out = torch.cat(features, dim=1)
		out = self.transition(out)  # 调整到正确的输出通道数
		return out


