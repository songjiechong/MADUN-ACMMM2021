import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
import glob
from time import time
import math
from torch.nn import init
import copy
import cv2
from skimage.measure import compare_ssim as ssim
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser(description='MADUN')
parser.add_argument('--epoch_num', type=int, default=401, help='epoch number of model')
parser.add_argument('--layer_num', type=int, default=25, help='phase number of MADUN')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cs_ratio', type=int, default=30, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--channels', type=int, default=32, help='feature number')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')
parser.add_argument('--algo_name', type=str, default='MADUN', help='log directory')

args = parser.parse_args()

epoch_num = args.epoch_num
learning_rate = args.learning_rate
layer_num = args.layer_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
test_name = args.test_name
channels = args.channels

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}
n_input = ratio_dict[cs_ratio]
n_output = 1089

# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']

# Initialization model
def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)

# Define ConvLSTM
class ConvLSTM(nn.Module):
    def __init__(self, inp_dim, oup_dim, kernel):
        super().__init__()
        
        pad_x = 1
        self.conv_xf = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)
        self.conv_xi = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)
        self.conv_xo = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)
        self.conv_xj = nn.Conv2d(inp_dim, oup_dim, kernel, padding=pad_x)
        pad_h = 1
        self.conv_hf = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hi = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_ho = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)
        self.conv_hj = nn.Conv2d(oup_dim, oup_dim, kernel, padding=pad_h)

    def forward(self, x, h, c):
        
        if h is None and c is None:
            i = F.sigmoid(self.conv_xi(x))
            o = F.sigmoid(self.conv_xo(x))
            j = F.tanh(self.conv_xj(x))
            c = i * j
            h = o * c
        else:
            f = F.sigmoid(self.conv_xf(x) + self.conv_hf(h))
            i = F.sigmoid(self.conv_xi(x) + self.conv_hi(h))
            o = F.sigmoid(self.conv_xo(x) + self.conv_ho(h))
            j = F.tanh(self.conv_xj(x) + self.conv_hj(h))
            c = f * c + i * j
            h = o * F.tanh(c)
    
        return h, h, c

# Define RB
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):

        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        res = x
        x = res + input
        return x

# Define MADUN Stage
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(channels, channels+1, 3, 3)))
        self.RB1 = ResidualBlock(channels, channels, 3, bias=True)
        self.RB2 = ResidualBlock(channels, channels, 3, bias=True)
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, channels, 3, 3)))
        self.ConvLSTM = ConvLSTM(channels, channels, 3)
        
    def forward(self, x, z, PhiWeight, PhiTWeight, PhiTb, h, c):
        
        x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
        x_input = x + self.lambda_step * PhiTb
        x_a = torch.cat([x_input, z], 1)
        x_D = F.conv2d(x_a, self.conv_D, padding=1)
        x = self.RB1(x_D)
        x, h, c = self.ConvLSTM(x, h, c)
        x_backward = self.RB2(x)
        x_G = F.conv2d(x_backward, self.conv_G, padding=1)
        x_pred = x_input + x_G

        return x_pred, x_backward, h, c

# Define MADUN
class MADUN(torch.nn.Module):
    def __init__(self, LayerNo):
        super(MADUN, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)
        self.fe = nn.Conv2d(1, channels, 3, padding=1, bias=True)

    def forward(self, Phix, Phi):

        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
        PhiTWeight = Phi.t().contiguous().view(n_output, n_input, 1, 1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None) # 64*1089*3*3 
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)
        x = PhiTb
        [h, c] = [None, None]
        z = self.fe(x)

        for i in range(self.LayerNo):
            x, z, h, c = self.fcs[i](x, z, PhiWeight, PhiTWeight, PhiTb, h, c)

        x_final = x

        return x_final

model = MADUN(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

num_params = 0
for para in model.parameters():
    num_params += para.numel()
print("total para num: %d\n" %num_params)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/CS_%s_channels_%d_layer_%d_ratio_%d" % (args.model_dir, args.algo_name, channels, layer_num, cs_ratio)

# Load pre-trained model with epoch number
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num)))

def imread_CS_py(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape
    return [Iorg, row, col, Ipad, row_new, col_new]

def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

test_dir = os.path.join(args.data_dir, test_name)
if test_name=='Set11':
    filepaths = glob.glob(test_dir + '/*.tif')
else:
    filepaths = glob.glob(test_dir + '/*.png')

result_dir = os.path.join(args.result_dir, test_name)
result_dir = os.path.join(result_dir, 'MADUN')
result_dir = os.path.join(result_dir, ('%d' % args.cs_ratio))
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.to(device)

results_csv=[]

print('\n')
print("CS Reconstruction Start")

with torch.no_grad():
    for img_no in range(ImgNum):

        imgName = filepaths[img_no]
        img_index = imgName.split('/')[-1].split('.')[0]
        Img = cv2.imread(imgName, 1)

        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
        Img_rec_yuv = Img_yuv.copy()
        Iorg_y = Img_yuv[:,:,0]
        Iorg = Iorg_y.copy()
        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
        Img_output = Ipad.reshape(1, 1, Ipad.shape[0], Ipad.shape[1])/255.0
        torch.cuda.synchronize()
        start = time()

        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)
        
        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
        Phix = F.conv2d(batch_x, PhiWeight, padding=0, stride=33, bias=None)

        x_output = model(Phix, Phi)
        torch.cuda.synchronize()
        end = time()

        Prediction_value = x_output.cpu().data.numpy().squeeze()
        row = Iorg.shape[0]
        col = Iorg.shape[1]

        X_rec = np.clip(Prediction_value[0:row, 0:col], 0, 1)

        rec_PSNR = psnr(X_rec*255, Iorg.astype(np.float64))
        rec_SSIM = ssim(X_rec*255, Iorg.astype(np.float64), data_range=255)

        print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM))

        Img_rec_yuv[:,:,0] = X_rec*255

        im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
        im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

        cv2.imwrite("%s/%s_MADUN_PSNR_%.2f_SSIM_%.4f.png" % (result_dir, img_index, rec_PSNR, rec_SSIM), im_rec_rgb)

        del x_output
        
        result_csv = [img_index] + [rec_PSNR] + [rec_SSIM]
        results_csv.append(result_csv)

        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM

print('\n')
output_data = "CS ratio is %d, Avg PSNR/SSIM for %s is %.2f/%.4f, Epoch number of model is %d \n" % (cs_ratio, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), epoch_num)
print(output_data)
print("CS Reconstruction End")
