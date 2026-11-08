import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
	def __init__(self, in_channels, ratio=8):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)
		
		self.fc1 = nn.Linear(in_channels, in_channels // ratio, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.fc2 = nn.Linear(in_channels // ratio, in_channels, bias=False)
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, x):
		b, c, h, w = x.size()
		
		# 平均池化分支
		avg_out = self.avg_pool(x).view(b, c)
		avg_out = self.fc1(avg_out)
		avg_out = self.relu(avg_out)
		avg_out = self.fc2(avg_out)
		
		# 最大池化分支
		max_out = self.max_pool(x).view(b, c)
		max_out = self.fc1(max_out)
		max_out = self.relu(max_out)
		max_out = self.fc2(max_out)
		
		# 相加并通过sigmoid
		out = avg_out + max_out
		out = self.sigmoid(out).view(b, c, 1, 1)
		
		return x * out


class SpatialAttention(nn.Module):
	def __init__(self, kernel_size=7):
		super(SpatialAttention, self).__init__()
		padding = (kernel_size - 1) // 2
		self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, x):
		# 计算通道维度的平均值和最大值
		avg_out = torch.mean(x, dim=1, keepdim=True)
		max_out, _ = torch.max(x, dim=1, keepdim=True)
		
		# 拼接并卷积
		out = torch.cat([avg_out, max_out], dim=1)
		out = self.conv(out)
		out = self.sigmoid(out)
		
		return x * out


class CBAMBlock(nn.Module):
	def __init__(self, in_channels, ratio=8, kernel_size=7):
		super(CBAMBlock, self).__init__()
		self.channel_attention = ChannelAttention(in_channels, ratio)
		self.spatial_attention = SpatialAttention(kernel_size)
	
	def forward(self, x):
		x = self.channel_attention(x)
		x = self.spatial_attention(x)
		return x
