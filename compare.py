import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import scipy.io as sio 
from scipy.io import savemat
import numpy as np
import math
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from collections import OrderedDict

img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
# network params
encoded_dim = 512 #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32
img_size = 32
in_chans = 2
num_heads = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
depth = 1
p = 0.
attn_p = 0.
qkv_bias=True
window = 8


class GroupAttention(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, num_heads=4, qkv_bias=False, attn_p=0., proj_p=0.):
        super(GroupAttention, self).__init__()

        self.num_heads = num_heads
        head_dim = img_size // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(img_size, img_size * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(img_size, img_size)
        # self.proj_drop = nn.Dropout(proj_p)
        self.ws = window

    def forward(self, x):
        # print(x.shape)
        B, C, H, W = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, C, h_group, self.ws, W)
        # print(x.shape)
        qkv = self.qkv(x).reshape(B, C, total_groups, -1, 3, self.num_heads, self.ws // self.num_heads).permute(4, 0, 1, 2, 5, 3, 6)
        # B, 2, hw, ws*ws, 3, n_head, head_dim -> 3, B, 2, hw, n_head, ws*ws, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, 2, hw, n_head, ws*ws, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, 2, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(
            # attn)  # attn @ v-> B, 2, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, 2, hw, ws*ws, n_head,  head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, C, H, W)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x

class GlobalAttention(nn.Module):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """
    def __init__(self, num_heads=4, qkv_bias=False, attn_p=0., proj_p=0.):
        super().__init__()

        self.dim = img_size
        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        # self.kv = nn.Linear(self.dim//window, self.dim//window * 2, bias=qkv_bias)
        self.kv = nn.Linear(self.dim, self.dim*2, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(self.dim, self.dim)
        # self.proj_drop = nn.Dropout(proj_p)
        self.sr = nn.Conv2d(2, 2, kernel_size=window, stride=window)
        self.norm = nn.LayerNorm(self.dim//window)

    # def forward(self, x):
    #     B, C, H, W = x.shape
    #     q = self.q(x).reshape(B, C, -1, self.dim//window, self.dim//window).permute(0,1,3,2,4)
    #     x_ = self.sr(x).reshape(B, C, -1, self.dim//window, self.dim//window)
    #     x_ = self.norm(x_)
    #     kv = self.kv(x_).reshape(B, C, -1, 2, self.dim//window, self.dim//window).permute(3,0,1,4,2,5)
    #     k, v = kv[0], kv[1]

    #     attn = (q @ k.transpose(-2, -1)) * self.scale
    #     attn = attn.softmax(dim=-1)
    #     # attn = self.attn_drop(attn)

    #     x = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
    #     x = self.proj(x)
    #     # x = self.proj_drop(x)

    #     return x

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, C, H, self.num_heads, W // self.num_heads).permute(0, 1, 3, 2, 4)

        kv = self.kv(x).reshape(B, C, -1, 2, self.num_heads, W // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)
        # x = self.proj_drop(x)

        return x

class MLP(nn.Module):

    def __init__(self, p=0.):
        super().__init__()
        self.cc1 = nn.Linear(img_size, img_size)
        self.cc2 = nn.Linear(img_size, img_size)
        # self.cc1 = nn.Conv2d(2,2, kernel_size=5, stride=1, padding=2)
        # self.cc1 = nn.Conv2d(2,2, kernel_size=window, stride=window)
        # self.cc2 = nn.ConvTranspose2d(2,2, kernel_size=window, stride=window)
        self.act = nn.GELU()
        # self.cc2 = nn.Conv2d(2,2, kernel_size=5, stride=1, padding=2)
        # self.drop = nn.Dropout(p)

    def forward(self, x):

        x = self.cc1(
                x
        ) # (n_samples, n_patches, hidden_features)
        # print(x.size())
        x = self.act(x)  # (n_samples, n_patches, hidden_features)
        # x = self.drop(x)  # (n_samples, n_patches, hidden_features)
        x = self.cc2(x) 
         # (n_samples, n_patches, out_features)
        # x = self.drop(x)  # (n_samples, n_patches, out_features)

        return x


class WTL(nn.Module):
    def __init__(self, num_heads, qkv_bias, p, attn_p):
        super().__init__()
        self.norm1 = nn.LayerNorm(img_size, eps=1e-6)
        self.attn1 = GroupAttention(
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.attn2 = GlobalAttention(
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.norm2 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm4 = nn.LayerNorm(img_size, eps=1e-6)
        self.mlp1 = MLP()
        self.mlp2 = MLP()

    def forward(self, x):

        # x = x + self.attn(x.clone())
        x = x + self.attn1(self.norm1(x))
        x = x + self.mlp1(self.norm2(x))
        x = x + self.attn2(self.norm3(x))
        x = x + self.mlp2(self.norm4(x))

        return x

class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(2, 7, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(7, 7, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(7, 7, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(2, 7, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(7, 7, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(7 * 2, 2, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            img_size=img_size,
            depth=depth,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            p=p,
            attn_p=attn_p,
    ):
        super().__init__()

        # self.pos_embed = nn.Parameter(
        #         torch.zeros(1, 2, img_size, img_size)
        # )
        # self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                WTL(
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        # self.norm1 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(img_size, eps=1e-6)
        self.conv1 = nn.Conv2d(2,16, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(16,2, kernel_size=5, stride=1, padding=2)
        # self.conv5 = nn.Conv2d(2,2, kernel_size=1, stride=1, padding=0)
        # self.conv3 = nn.Conv2d(2,2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(2,2, kernel_size=4, stride=2, padding=1)
        self.convT = nn.ConvTranspose2d(2,2, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(2*img_size*img_size, encoded_dim)

    def forward(self, x):

        n_samples = x.shape[0]
        # print(x.size()) 
        # y = x
        x = self.conv1(x)
        x = self.conv5(x)
        # print(x.size()) 
        X = x #+ self.pos_embed  # (n_samples, n_patches, embed_dim)
        # print(X.size()) 
        # x = self.pos_drop(X)

        for block in self.blocks:
            x = block(x)
        x = self.norm3(x)
        # for block in self.blocks:
        #     x = block(x)
        # x = self.norm3(x)
        # x = self.convT(x)
        # # # print(x.size()) 
        # x = self.conv4(x) 
        # for block in self.blocks:
        #     x = block(x)
        # x = self.norm1(x)
        x = self.convT(x) 
        x = X + self.conv4(x)
        x = self.norm2(x)
        # print(x.size()) 
        # x = self.convT(x)
        # print(x.size()) 
        # print(y.size())
        # x = self.conv3(x)
        # x = self.norm3(x)
        ## x = self.convT(x) 
        # x = self.conv3(x)
        # print(x.size()) 
        x = x.reshape(n_samples,2*img_size*img_size)
        x = self.fc(x)
        # print(x.size()) 
        return x


class Decoder(nn.Module):   
    def __init__(self):
        super(Decoder, self).__init__()

        # self.pos_embed = nn.Parameter(
        #         torch.zeros(1, 2, img_size, img_size)
        # )

        # self.pos_drop = nn.Dropout(p=p)

        self.fc = nn.Linear(encoded_dim, in_chans*img_size*img_size)
        self.act = nn.Sigmoid()
        self.conv5 = nn.Conv2d(2,2, kernel_size=5, stride=1, padding=2)
        # self.conv5 = nn.Conv2d(2,2, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(2,2, kernel_size=4, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(2,2, kernel_size=3, stride=1, padding=1)
        # self.conv8 = nn.Conv2d(2,2, kernel_size=3, stride=8, padding=1)
        self.convT = nn.ConvTranspose2d(2,2, kernel_size=4, stride=2, padding=1)
        self.blocks = nn.ModuleList(
            [
                WTL(
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )
        # self.norm1 = nn.BatchNorm2d(2, eps=1e-6)
        self.norm2 = nn.LayerNorm(img_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(img_size, eps=1e-6)
        # self.norm4 = nn.BatchNorm2d(2, eps=1e-6)

        self.dense_layers = nn.Sequential(
            nn.Linear(encoded_dim, img_total)
        )

        # self.cnn_layers = nn.Sequential(
        #     # Defining a 2D convolution layer
        #     nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(16, 2, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(2),
        #     nn.LeakyReLU(),
        # )

        # self.final_layers = nn.Sequential(
        #     nn.LeakyReLU(),
        #     )

        # self.output = nn.Sequential(               
        #     nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1),
        #     nn.Sigmoid(),
        #     )

        decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(2, 2, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock())
            # ("CRBlock2", CRBlock())
        ])
        self.decoder_feature = nn.Sequential(decoder)
        # self.hsig= nn.Sigmoid()

    def forward(self, x):
        img = x
        img = self.dense_layers(img)
        X = img.view(-1, img_channels, img_height, img_width)
        img = X
        # x = self.cnn_layers(X)
        # x = torch.add(x,X)
        # # X = self.final_layers(x)
        # X = self.norm1(x)
        # out1 = X

        out = self.decoder_feature(img)
        # out1 = self.hsig(out)
        # n_samples = img.shape[0]
        # x = self.fc(img)
        # print(x.size()) 
        # x = x.reshape(n_samples,in_chans,img_size,img_size)
        # print(x.size()) 
        y = self.conv5(img)
        # print(x.size()) 
        x = y #+ self.pos_embed  # (n_samples, n_patches, embed_dim)
        # print(X.size()) 
        # x = self.pos_drop(X)

        for block in self.blocks:
            x = block((x+out))

        x = self.norm2(x)

        # x = self.cnn_layers(X)
        # x = torch.add(x,out)
        # # X = self.final_layers(x)
        # X = self.norm3(x)

        x = self.convT(x)
        # # print(x.size()) 
        x = self.conv4(x) 

        for block in self.blocks:
            x = block((x+out))
        x = self.norm3(x)
        # x = self.convT(x)
        # # print(x.size()) 
        # x = self.conv3(x)

        # x = self.norm2(x)
        # # print(x.size()) 
        # out2 = self.conv4(x)
        out2 = x

        x = self.act(out2) 

        return x

cr = 128
mode = 5
hori = 5
kmph = 120

x_test = torch.load(f"512/{cr}/target_{mode}_{cr}_{kmph}_{hori}.pt").float()
x_hat  = torch.load(f"512/{cr}/predict_{mode}_{cr}_{kmph}_{hori}.pt").float()

print(x_test.size())

decoder = torch.load(f"decoder_{cr}")
decoder.eval()
decoder.to('cpu')

nmse = np.zeros((10,1))
corr = np.zeros((10,1))
se   = np.zeros((10,1))



with torch.no_grad():
    h_test = decoder.forward(x_test)
    h_hat = decoder.forward(x_hat)


# print(h_test.size())
h_test = h_test.to('cpu')
h_hat = h_hat.to('cpu')


# print(h_test.size())
h_test = h_test.numpy()
h_hat = h_hat.numpy()



h_test_real = np.reshape(h_test[:, 0, :, :], (len(h_test), -1))
h_test_imag = np.reshape(h_test[:, 1, :, :], (len(h_test), -1))
# h_test_C = h_test_real + 1j*(h_test_imag)
h_test_C = h_test_real-0.5 + 1j*(h_test_imag-0.5)
h_hat_real = np.reshape(h_hat[:, 0, :, :], (len(h_hat), -1))
h_hat_imag = np.reshape(h_hat[:, 1, :, :], (len(h_hat), -1))
# h_hat_C = h_hat_real + 1j*(h_hat_imag)
h_hat_C = h_hat_real-0.5 + 1j*(h_hat_imag-0.5)
power = np.sum(abs(h_test_C)**2, axis=1)
mse = np.sum(abs(h_test_C-h_hat_C)**2, axis=1)


H_test = np.reshape(h_test_C, (-1, 32, 32))
print(np.shape(H_test))
H_hat = np.reshape(h_hat_C, (-1, 32, 32))

torch.save(torch.tensor(h_test[:,0,:,:]+1j*h_test[:,1,:,:]), f"512/{cr}_{kmph}/H_test_{cr}_{kmph}_{mode}_{hori}")
torch.save(torch.tensor(h_hat[:,0,:,:]+1j*h_hat[:,1,:,:]), f"512/{cr}_{kmph}/H_hat_{cr}_{kmph}_{mode}_{hori}")


import matplotlib.pyplot as plt
'''abs'''
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display origoutal
    ax = plt.subplot(2, n, i + 1 )
    # print(i)
    # h_testplo = abs(h_test[i,0]+1j*h_test[i,1])
    h_testplo = abs(h_test[10*i,0]+1j*h_test[10*i,1])
    order = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27 ,28 ,29, 30, 31]
    H = h_testplo[order,:]
    plt.imshow(H)
    # plt.imshow(np.max(np.max(h_testplo))-h_testplo.T)
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # ax.invert_yaxis()
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    # decoded_imgsplo = abs(h_hat[i,0]+1j*h_hat[i,1])
    decoded_imgsplo = abs(h_hat[10*i,0]+1j*h_hat[10*i,1])
    Ht = decoded_imgsplo[order,:]
    plt.imshow(Ht)
    # plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # ax.invert_yaxis()
plt.show()




# print(np.shape(x_test))

n1 = abs(np.sqrt(np.sum(np.conj(H_test)*H_test, axis=1)))
# print(np.shape(n1))
n2 = abs(np.sqrt(np.sum(np.conj(H_hat)*H_hat, axis=1)))
aa = abs(np.sum(np.conj(H_hat)*H_test, axis=1))
rho2 = np.mean(aa/(n1*n2), axis=0)
# print("NMSE is ", 10*math.log10(np.mean(mse/power)))
# print("Correlation is ", np.max(rho2))
# print("SE is ", np.mean(capacity))


# print("NMSE is ", 10*math.log10(np.mean(mse1/power)))
# print("Correlation is ", np.mean(rho3))

nmse[0] = 10*math.log10(np.mean(mse/power))
corr[0] = np.max(rho2)

print("Total NMSE is ", np.mean(nmse))
print("Total Corr is ", np.mean(corr))