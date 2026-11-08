import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score, f1_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from CBAM import CBAMBlock
from DenseBlock import DenseBlock
from ConvBlock import ConvBlock


class EncoderBlock(nn.Module):
	def __init__(self, in_channels, output_channels, ifdense, cbam, ratio=4):
		super(EncoderBlock, self).__init__()
		if ifdense == 1:
			self.encoder = DenseBlock(in_channels, output_channels)
		else:
			self.encoder = ConvBlock(in_channels, output_channels)
		
		self.use_cbam = (cbam == 1)
		if self.use_cbam:
			self.cbam = CBAMBlock(output_channels, ratio=ratio)
		
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
	
	def forward(self, x):
		encoder = self.encoder(x)
		if self.use_cbam:
			encoder = self.cbam(encoder)
		encoder_pool = self.pool(encoder)
		return encoder_pool, encoder


class MidBlock(nn.Module):
	def __init__(self, in_channels, output_channels, ifdense, cbam, ratio=4):
		super(MidBlock, self).__init__()
		if ifdense == 1:
			self.encoder = DenseBlock(in_channels, output_channels)
		else:
			self.encoder = ConvBlock(in_channels, output_channels)
		
		self.use_cbam = (cbam == 1)
		if self.use_cbam:
			self.cbam = CBAMBlock(output_channels, ratio=ratio)
	
	def forward(self, x):
		encoder = self.encoder(x)
		if self.use_cbam:
			encoder = self.cbam(encoder)
		return encoder


class DecoderBlock(nn.Module):
	def __init__(self, in_channels, output_channels, ifdense, cbam, ratio=4):
		super(DecoderBlock, self).__init__()
		self.upconv = nn.ConvTranspose2d(in_channels, output_channels, kernel_size=2, stride=2)
		# BatchNorm应该对拼接后的通道数进行操作
		self.bn = nn.BatchNorm2d(output_channels * 2)
		self.relu = nn.ReLU(inplace=True)
		
		# 注意：concat后通道数翻倍
		if ifdense == 1:
			self.decoder = DenseBlock(output_channels * 2, output_channels)
		else:
			self.decoder = ConvBlock(output_channels * 2, output_channels)
		
		self.use_cbam = (cbam == 1)
		if self.use_cbam:
			self.cbam = CBAMBlock(output_channels, ratio=ratio)
	
	def forward(self, x, concat_tensor):
		decoder = self.upconv(x)
		decoder = torch.cat([concat_tensor, decoder], dim=1)
		decoder = self.bn(decoder)
		decoder = self.relu(decoder)
		decoder = self.decoder(decoder)
		if self.use_cbam:
			decoder = self.cbam(decoder)
		return decoder


class UNetModel(nn.Module):
	def __init__(self, in_channels, num_classes, mode, ifdense, num_layers, channel, ifcbam):
		super(UNetModel, self).__init__()
		self.num_layers = num_layers
		self.mode = mode
		
		ifdenseen = int(ifdense[0])
		ifdensede = int(ifdense[1])
		ifcbamen = int(ifcbam[0])
		ifcbamde = int(ifcbam[1])
		
		channel = sorted(channel)
		
		# 构建编码器
		self.encoders = nn.ModuleList()
		self.mid = None
		self.decoders = nn.ModuleList()
		
		if num_layers == 1:
			self.encoders.append(EncoderBlock(in_channels, int(channel[0]), ifdenseen, ifcbamen, int(channel[0])//4))
			self.mid = MidBlock(int(channel[0]), int(channel[1]), ifdenseen, ifcbamen, int(channel[1])//4)
			self.decoders.append(DecoderBlock(int(channel[1]), int(channel[0]), ifdensede, ifcbamde, int(channel[0])//4))
			
		elif num_layers == 2:
			self.encoders.append(EncoderBlock(in_channels, int(channel[0]), ifdenseen, ifcbamen, int(channel[0])//4))
			self.encoders.append(EncoderBlock(int(channel[0]), int(channel[1]), ifdenseen, ifcbamen, int(channel[1])//4))
			self.mid = MidBlock(int(channel[1]), int(channel[2]), ifdenseen, ifcbamen, int(channel[2])//4)
			self.decoders.append(DecoderBlock(int(channel[2]), int(channel[1]), ifdensede, ifcbamde, int(channel[1])//4))
			self.decoders.append(DecoderBlock(int(channel[1]), int(channel[0]), ifdensede, ifcbamde, int(channel[0])//4))
			
		elif num_layers == 3:
			self.encoders.append(EncoderBlock(in_channels, int(channel[0]), ifdenseen, ifcbamen, int(channel[0])//4))
			self.encoders.append(EncoderBlock(int(channel[0]), int(channel[1]), ifdenseen, ifcbamen, int(channel[1])//4))
			self.encoders.append(EncoderBlock(int(channel[1]), int(channel[2]), ifdenseen, ifcbamen, int(channel[2])//4))
			self.mid = MidBlock(int(channel[2]), int(channel[3]), ifdenseen, ifcbamen, int(channel[3])//4)
			self.decoders.append(DecoderBlock(int(channel[3]), int(channel[2]), ifdensede, ifcbamde, int(channel[2])//4))
			self.decoders.append(DecoderBlock(int(channel[2]), int(channel[1]), ifdensede, ifcbamde, int(channel[1])//4))
			self.decoders.append(DecoderBlock(int(channel[1]), int(channel[0]), ifdensede, ifcbamde, int(channel[0])//4))
			
		elif num_layers == 4:
			self.encoders.append(EncoderBlock(in_channels, int(channel[0]), ifdenseen, ifcbamen, int(channel[0])//4))
			self.encoders.append(EncoderBlock(int(channel[0]), int(channel[1]), ifdenseen, ifcbamen, int(channel[1])//4))
			self.encoders.append(EncoderBlock(int(channel[1]), int(channel[2]), ifdenseen, ifcbamen, int(channel[2])//4))
			self.encoders.append(EncoderBlock(int(channel[2]), int(channel[3]), ifdenseen, ifcbamen, int(channel[3])//4))
			self.mid = MidBlock(int(channel[3]), int(channel[4]), ifdenseen, ifcbamen, int(channel[4])//4)
			self.decoders.append(DecoderBlock(int(channel[4]), int(channel[3]), ifdensede, ifcbamde, int(channel[3])//4))
			self.decoders.append(DecoderBlock(int(channel[3]), int(channel[2]), ifdensede, ifcbamde, int(channel[2])//4))
			self.decoders.append(DecoderBlock(int(channel[2]), int(channel[1]), ifdensede, ifcbamde, int(channel[1])//4))
			self.decoders.append(DecoderBlock(int(channel[1]), int(channel[0]), ifdensede, ifcbamde, int(channel[0])//4))
			
		else:  # num_layers == 5
			self.encoders.append(EncoderBlock(in_channels, int(channel[0]), ifdenseen, ifcbamen, int(channel[0])//4))
			self.encoders.append(EncoderBlock(int(channel[0]), int(channel[1]), ifdenseen, ifcbamen, int(channel[1])//4))
			self.encoders.append(EncoderBlock(int(channel[1]), int(channel[2]), ifdenseen, ifcbamen, int(channel[2])//4))
			self.encoders.append(EncoderBlock(int(channel[2]), int(channel[3]), ifdenseen, ifcbamen, int(channel[3])//4))
			self.encoders.append(EncoderBlock(int(channel[3]), int(channel[4]), ifdenseen, ifcbamen, int(channel[4])//4))
			self.mid = MidBlock(int(channel[4]), int(channel[5]), ifdenseen, ifcbamen, int(channel[5])//4)
			self.decoders.append(DecoderBlock(int(channel[5]), int(channel[4]), ifdensede, ifcbamde, int(channel[4])//4))
			self.decoders.append(DecoderBlock(int(channel[4]), int(channel[3]), ifdensede, ifcbamde, int(channel[3])//4))
			self.decoders.append(DecoderBlock(int(channel[3]), int(channel[2]), ifdensede, ifcbamde, int(channel[2])//4))
			self.decoders.append(DecoderBlock(int(channel[2]), int(channel[1]), ifdensede, ifcbamde, int(channel[1])//4))
			self.decoders.append(DecoderBlock(int(channel[1]), int(channel[0]), ifdensede, ifcbamde, int(channel[0])//4))
		
		# 输出层
		if mode == 1:
			self.output = nn.Conv2d(int(channel[0]), num_classes, kernel_size=1)
		else:
			self.output = nn.Conv2d(int(channel[0]), 1, kernel_size=1)
	
	def forward(self, x):
		# 编码阶段
		encoder_outputs = []
		for encoder in self.encoders:
			x, encoder_output = encoder(x)
			encoder_outputs.append(encoder_output)
		
		# 中间层
		x = self.mid(x)
		
		# 解码阶段
		for i, decoder in enumerate(self.decoders):
			encoder_output = encoder_outputs[-(i+1)]
			x = decoder(x, encoder_output)
		
		# 输出层
		x = self.output(x)
		
		if self.mode == 1:
			x = F.softmax(x, dim=1)
		
		return x



# 分类交叉熵损失
class CCELoss(nn.Module):
	def __init__(self):
		super(CCELoss, self).__init__()
	
	def forward(self, y_pred, y_true):
		# y_true shape: (B, H, W), y_pred shape: (B, C, H, W)
		# 找到所有有效像素（标签在 0-5 范围内，即 < 6）
		# 排除无效标签（如 -1, 15 等背景/填充值）
		mask = (y_true >= 0) & (y_true < 6)
		
		if not mask.any():
			return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
		
		# 提取有效位置的真实标签和预测值
		y_true_valid = y_true[mask].long()
		y_pred_valid = y_pred.permute(0, 2, 3, 1)[mask]  # (N, C)
		
		# 计算交叉熵
		y_pred_log = torch.log(y_pred_valid + 1e-7)
		loss = -torch.sum(y_pred_log[torch.arange(y_pred_log.size(0)), y_true_valid])
		
		# 归一化损失（除以有效样本数）
		loss = loss / y_true_valid.size(0)
		
		return loss


# 回归MAE损失
class CustomizedMAE(nn.Module):
	def __init__(self):
		super(CustomizedMAE, self).__init__()
	
	def forward(self, y_pred, y_true):
		# y_true shape: (B, H, W), y_pred shape: (B, 1, H, W)
		y_pred = y_pred.squeeze(1)  # (B, H, W)
		
		# 找到有效位置（>=0）
		mask = (y_true >= 0)
		if not mask.any():
			return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
		
		y_true_valid = y_true[mask]
		y_pred_valid = y_pred[mask]
		
		loss = torch.mean(torch.abs(y_true_valid - y_pred_valid))
		return loss


def get_model(BANDS, class_num, mode, ifdense, num_layers, channel, ifcbam):
	channel = np.sort(channel)
	in_channels = len(BANDS) - 1
	
	model = UNetModel(in_channels, class_num, mode, ifdense, num_layers, channel, ifcbam)
	
	return model


