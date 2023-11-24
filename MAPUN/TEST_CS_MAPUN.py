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

parser = ArgumentParser(description='ISTA-Net-plus')

parser.add_argument('--epoch_num', type=int, default=401, help='epoch number of model')
parser.add_argument('--start_epoch', type=int, default=400, help='epoch number of start training')
parser.add_argument('--layer_num', type=int, default=20, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=30, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--noise', type=float, default=0, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--patch_size', type=int, default=99)
parser.add_argument('--finetune_psize', type=int, default=132)
parser.add_argument('--channels', type=int, default=32, help='1 for gray, 3 for color')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')
parser.add_argument('--train_name', type=str, default='train400', help='name of test set')
parser.add_argument('--test_name', type=str, default='Urban100', help='name of test set')
parser.add_argument('--algo_name', type=str, default='MAPUN', help='log directory')

args = parser.parse_args()

epoch_num = args.epoch_num
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
test_name = args.test_name
channels = args.channels
noise = args.noise

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = ratio_dict[cs_ratio]
n_output = 1089
nrtrain = 88912
batch_size = 64

# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input = Phi_data['phi']
Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.to(device)

def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(33)(temp)

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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, res_scale=1):

        super(ResidualBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
#         print("x[:(x.shape(0)//2), ...]", x[:(x.shape(0)//2), ...].shape)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        res = x
        x = res + input
        return x

class SFT(nn.Module):
    def __init__(self, channels):
        
        super(SFT, self).__init__()
        self.RB1 = ResidualBlock(channels, channels, 3, bias=True, res_scale=1)
        self.RB2 = ResidualBlock(channels, channels, 3, bias=True, res_scale=1)
        self.RB3 = ResidualBlock(channels, channels, 3, bias=True, res_scale=1)
        self.RB4 = ResidualBlock(channels, channels, 3, bias=True, res_scale=1)
        
    def forward(self, x):
        ch = x.shape[1]
        x1 = x[:, :(ch//2), :, :]
#         print("x1:", x1.shape)
        x2 = x[:, (ch//2):, :, :]
        x_beta = self.RB1(x2)
        x_gamma = self.RB2(x_beta)
        x1 = self.RB3(x1*x_beta)
        x1 = self.RB4(x1+x_gamma)
        x = torch.cat((x1, x_gamma), 1)
        return x
        
    
# Define ISTA-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        rb_num = 2
        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(channels, channels+1, 3, 3)))
        self.SFT1 = SFT(channels//2)
        self.SFT2 = SFT(channels//2)
        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, channels, 3, 3)))
        self.ConvLSTM = ConvLSTM(channels, channels, 3)
        
    def forward(self, x, z, PhiWeight, PhiTWeight, PhiTb, h, c):
        x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
        x_input = x + self.lambda_step * PhiTb
        x_a = torch.cat([x_input, z], 1)

        x_D = F.conv2d(x_a, self.conv_D, padding=1)
        x = self.SFT1(x_D)
        x, h, c = self.ConvLSTM(x, h, c)
        x_backward = self.SFT2(x)
        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = x_input + x_G

        return x_pred, x_backward, h, c

# Define ISTA-Net-plus
class ISTANetplus(torch.nn.Module):
    def __init__(self, LayerNo, Phi):
        super(ISTANetplus, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
        self.PhiTWeight = Phi.t().contiguous().view(n_output, n_input, 1, 1)

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

#         self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        self.fcs = nn.ModuleList(onelayer)
        self.fe = nn.Conv2d(1, channels, 3, padding=1, bias=True)

    def forward(self, Phix):

#         PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
#         PhiTWeight = Phi.t().contiguous().view(n_output, n_input, 1, 1)
        PhiTb = F.conv2d(Phix, self.PhiTWeight.to(Phix.device), padding=0, bias=None) # 64*1089*3*3 
        PhiTb = torch.nn.PixelShuffle(33)(PhiTb)
        x = PhiTb
        [h, c] = [None, None]
        z = self.fe(x)

        for i in range(self.LayerNo):
            x, z, h, c = self.fcs[i](x, z, self.PhiWeight, self.PhiTWeight, PhiTb, h, c)
#             x = x_dual[:, :1, :, :]
#             z = x_dual[:, 1:, :, :]

        x_final = x

        return x_final

model = ISTANetplus(layer_num, Phi)
model = nn.DataParallel(model)
model = model.to(device)

num_params = 0
for para in model.parameters():
    num_params += para.numel()
print("total para num: %d\n" %num_params)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load pre-trained model with epoch number
model.load_state_dict(torch.load('./%s/CS_%s_ratio_%d.pkl' % (args.model_dir, args.algo_name, cs_ratio)))


def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)

# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)

def imread_CS_py(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = np.zeros([block_size**2, block_num])
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec


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

result_dir = os.path.join(args.result_dir, args.algo_name)
result_dir = os.path.join(result_dir, test_name)
result_dir = os.path.join(result_dir, ('%d' % args.cs_ratio))
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)


results_csv=[]

# Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
# Qinit = Qinit.to(device)

print('\n')
print("CS Reconstruction Start")

with torch.no_grad():
    for img_no in range(ImgNum):

        imgName = filepaths[img_no]
        if test_name=='bsd68': 
            img_index = imgName.split('_')[-1].split('.')[0][-2:]
        elif test_name=='Urban100':
            img_index = imgName.split('_')[-1].split('.')[0][-3:]
        else:
            img_index = imgName.split('/')[-1].split('.')[0]
        Img = cv2.imread(imgName, 1)

        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
        Img_rec_yuv = Img_yuv.copy()

        Iorg_y = Img_yuv[:,:,0]
        Iorg = Iorg_y.copy()

        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
#         Icol = img2col_py(Ipad, 33).transpose()/255.0

        Img_output = Ipad.reshape(1, 1, Ipad.shape[0], Ipad.shape[1])/255.0
        torch.cuda.synchronize()
        start = time()

        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(torch.FloatTensor)
#         _, _, h, w = batch_x.shape
        batch_x = batch_x.to(device)
#         perm = torch.randperm(h*w)
        
        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
        Phix = F.conv2d(batch_x, PhiWeight, padding=0, stride=33, bias=None)
#         Phi_norm = Phi.clone()
#         PhiTWeight = Phi_norm.t().contiguous().view(n_output, n_input, 1, 1)
#         Phix = Phi_fun(batch_x, PhiWeight, PhiTWeight, perm)
#         Phix = F.conv2d(batch_x, PhiWeight, padding=0, stride=33, bias=None)
        noise_sigma = noise/255.0 * torch.randn_like(Phix)
        Phix = Phix + noise_sigma
#         Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))

        x_output = model(Phix)
#         print("R_y:", R_y.shape, R_y.cpu().data.numpy())
        torch.cuda.synchronize()
        end = time()

#         x_output = x_output
        Prediction_value = x_output.cpu().data.numpy().squeeze()
        row = Iorg.shape[0]
        col = Iorg.shape[1]

#         X_rec = np.clip(col2im_CS_py(Prediction_value.transpose(), row, col, row_new, col_new), 0, 1)
        X_rec = np.clip(Prediction_value[0:row, 0:col], 0, 1)

        rec_PSNR = psnr(X_rec*255, Iorg.astype(np.float64))
        rec_SSIM = ssim(X_rec*255, Iorg.astype(np.float64), data_range=255)

        print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM))

        Img_rec_yuv[:,:,0] = X_rec*255

        im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
        im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

#         resultName = imgName.replace(args.data_dir, args.result_dir)
        cv2.imwrite("%s/%s_%s_layer_%d_ratio_%d_epoch_%d_PSNR_%.2f_SSIM_%.4f.png" % (result_dir, img_index, args.algo_name, layer_num, cs_ratio, epoch_num, rec_PSNR, rec_SSIM), im_rec_rgb)

        del x_output
        
        result_csv = [img_index] + [rec_PSNR] + [rec_SSIM]
        results_csv.append(result_csv)

        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM

print('\n')
output_data = "CS ratio is %d, Avg PSNR/SSIM for %s is %.2f/%.4f, Epoch number of model is %d \n" % (cs_ratio, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), epoch_num)
print(output_data)

# output_file_name = "./%s/PSNR_SSIM_Results_CS_SI_ASSB_noconv_alpha_dataloader_deblock_layer_%d_group_%d_lr_%.4f.txt" % (args.log_dir, layer_num, group_num, learning_rate)
# output_file_name = "./%s/PSNR_SSIM_Results_CS_%s_layer_%d_group_%d_lr50_le-5.txt" % (args.log_dir, layer_num, group_num)

# output_file = open(output_file_name, 'a')
# output_file.write(output_data)
# output_file.close()
# test_result = pd.DataFrame(results_csv)
# test_name = "./%s/PSNR_SSIM_Results_CS_%s_%s_epoch_%d_layer_%d_ratio_%d_lr_%.4f.csv" % (args.log_dir, args.algo_name, test_name, epoch_num, layer_num, cs_ratio, learning_rate)
# test_result.to_csv(test_name, index=False)
print("CS Reconstruction End")
