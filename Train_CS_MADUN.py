import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
import csdata_fast

parser = ArgumentParser(description='MADUN')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=400, help='epoch number of end training')
parser.add_argument('--finetune', type=int, default=10, help='epoch number of finetuning')
parser.add_argument('--layer_num', type=int, default=25, help='stage number of MADUN')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cs_ratio', type=int, default=30, help='from {10, 25, 30, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--patch_size', type=int, default=33)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--rgb_range', type=int, default=1, help='value range 1 or 255')
parser.add_argument('--n_channels', type=int, default=1, help='1 for gray, 3 for color')
parser.add_argument('--channels', type=int, default=32, help='feature number')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--train_name', type=str, default='train400', help='name of train set')
parser.add_argument('--ext', type=str, default='.png', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--algo_name', type=str, default='MADUN', help='log directory')
parser.add_argument('--data_copy', type=int, default=200, help='training data directory')

args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
channels = args.channels
finetune = args.finetune
batch_size = args.batch_size

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio_dict = {10:0, 25:1, 30:2, 40:3, 50:4}
n_input_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

n_input = n_input_dict[cs_ratio]
n_output = 1089

# Load CS Sampling Matrix: phi
Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, 10)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input10 = Phi_data['phi']

Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, 25)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input25 = Phi_data['phi']

Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, 30)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input30 = Phi_data['phi']

Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, 40)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input40 = Phi_data['phi']

Phi_data_Name = './%s/phi_0_%d_1089.mat' % (args.matrix_dir, 50)
Phi_data = sio.loadmat(Phi_data_Name)
Phi_input50 = Phi_data['phi']

# Initialization model
def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0,stride=33, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    temp = torch.nn.PixelShuffle(33)(temp)
    return temp

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

print_flag = 1  # print parameter number

if print_flag:
    num_count = 0
    num_params = 0
    for para in model.parameters():
        num_count += 1
        num_params += para.numel()
        print('Layer %d' % num_count)
        print(para.size())
    print("total para num: %d" % num_params)

training_data = csdata_fast1.SlowDataset(args)

if (platform.system() =="Windows"):
    rand_loader = DataLoader(dataset=training_data, batch_size=batch_size, num_workers=0,
                             shuffle=True)
else:
    rand_loader = DataLoader(dataset=training_data, batch_size=batch_size, num_workers=8,
                             shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/CS_%s_channels_%d_layer_%d_ratio_%d" % (args.model_dir, args.algo_name, channels, layer_num, cs_ratio)
log_file_name = "./%s/Log_CS_%s_channels_%d_layer_%d_ratio_%d.txt" % (args.log_dir, args.algo_name, channels, layer_num, cs_ratio)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))

Phi10 = torch.from_numpy(Phi_input10).type(torch.FloatTensor).to(device)
Phi25 = torch.from_numpy(Phi_input25).type(torch.FloatTensor).to(device)
Phi30 = torch.from_numpy(Phi_input30).type(torch.FloatTensor).to(device)
Phi40 = torch.from_numpy(Phi_input40).type(torch.FloatTensor).to(device)
Phi50 = torch.from_numpy(Phi_input50).type(torch.FloatTensor).to(device)
Phi_matrix = {0: Phi10, 1: Phi25, 2: Phi30, 3: Phi40, 4: Phi50}

media_epoch = end_epoch
if finetune > 0:
    end_epoch = end_epoch + finetune
    patch_size1 = 99
    
# Training loop
for epoch_i in range(start_epoch + 1, end_epoch + 1):

    if epoch_i > media_epoch:
        args.patch_size = patch_size1
    
    for data in rand_loader:
        batch_x = data
        batch_x = batch_x.to(device)
        batch_x = batch_x.view(-1, 1, args.patch_size, args.patch_size)
        
        Phi = Phi_matrix[ratio_dict[cs_ratio]]
        PhiWeight = Phi.contiguous().view(n_input, 1, 33, 33)
        Phix = F.conv2d(batch_x, PhiWeight, padding=0,stride=33, bias=None)

        x_output = model(Phix, Phi)

        # Compute and print loss
        loss_all = nn.L1Loss()(x_output, batch_x)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        output_data = "[%02d/%02d] Total Loss: %.4f\n" % (epoch_i, end_epoch, loss_all.item())
        print(output_data)

    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % 10 == 0 and epoch_i <= 400:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
    elif epoch_i > 400:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
