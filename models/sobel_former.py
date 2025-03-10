
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = torch.tensor(keep_prob)
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor()
        output = inputs.divide(keep_prob) * random_tensor
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_channels):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim,
                                   out_channels)

        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):  #
        h, w = self.input_resolution
        b, _, c = x.shape
        x = x.reshape([b, h, w, c])  #batch  height width channels

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.reshape([b, -1, 4 * c])

        x = self.norm(x)
        x = self.reduction(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, dropout):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features,
                             hidden_features)

        self.fc2 = nn.Linear(hidden_features,
                             in_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def windows_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.reshape([B, H // window_size, window_size, W // window_size, window_size,C])
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape([-1, window_size, window_size, C])

    return x


def windows_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape([B, H // window_size, W // window_size, window_size, window_size, -1])
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape([B, H, W, -1])
    return x


class WindowAttention(nn.Module):
    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attention_dropout=0.,
                 dropout=0.):
        super(WindowAttention, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.dim = dim
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head ** -0.5

        self.relative_position_bias_table = torch.nn.parameter.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))#生成一个参数矩阵

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1)

        relative_coords = relative_coords.permute(1, 2, 0)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim,
                             dim * 3)

        self.attn_dropout = nn.Dropout(attention_dropout)

        self.proj = nn.Linear(dim,
                              dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_multihead(self, x):
        new_shape = list(x.shape[:-1]) + [self.num_heads, self.dim_head]
        x = x.reshape(new_shape)
        x = x.permute(0, 2, 1, 3)
        return x

    def get_relative_pos_bias_from_pos_index(self):
        table = self.relative_position_bias_table

        index = self.relative_position_index.reshape([-1])

        relative_position_bias = torch.index_select(table, dim=0, index=index)
        return relative_position_bias

    def forward(self, x, mask=None):
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead, qkv)

        q = q * self.scale

        attn = torch.matmul(q, k.permute(0, 1, 3, 2))

        relative_position_bias = self.get_relative_pos_bias_from_pos_index()

        relative_position_bias = relative_position_bias.reshape(
            [self.window_size[0] * self.window_size[1],
             self.window_size[0] * self.window_size[1],
             -1])

        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(
                [x.shape[0] // nW, nW, self.num_heads, x.shape[1], x.shape[1]])
            attn += mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, x.shape[1], x.shape[1]])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_dropout(attn)

        z = torch.matmul(attn, v)
        z = z.permute(0, 2, 1, 3)
        new_shape = list(z.shape[:-2]) + [self.dim]
        z = z.reshape(new_shape)
        z = self.proj(z)
        z = self.proj_dropout(z)

        return z


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0.,
                 attention_dropout=0., droppath=0.):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = nn.LayerNorm(dim)

        self.attn = WindowAttention(dim,
                                    window_size=(self.window_size, self.window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attention_dropout=attention_dropout,
                                    dropout=dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else None

        self.norm2 = nn.LayerNorm(dim)

        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       dropout=dropout)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros(1, H, W, 1)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = windows_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape((-1, self.window_size * self.window_size))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = torch.where(attn_mask != 0,
                                    torch.ones_like(attn_mask) * float(-100.0),
                                    attn_mask)

            attn_mask = torch.where(attn_mask == 0,
                                    torch.zeros_like(attn_mask),
                                    attn_mask)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x)
        H, W = self.input_resolution
        B, L, C = x.shape
        h = x
        new_shape = [B, H, W, C]
        x = x.reshape(new_shape)

        if self.shift_size > 0:
            shifted_x = torch.roll(x,
                                   shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x

        x_windows = windows_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])

        shifted_x = windows_reverse(attn_windows, self.window_size, H, W)


        if self.shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
               
        else:
            x = shifted_x

        x = x.reshape([B, H * W, C])
        x = self.norm1(x)
        #Layer Norm

        if self.drop_path is not None:
            x = h + self.drop_path(x)
        else:
            x = h + x
        h = x

        x = self.mlp(x)
        x = self.norm2(x)
     
        if self.drop_path is not None:
            x = h + self.drop_path(x)
        else:
            x = h + x
        return x


class SwinT(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0.,
                 attention_dropout=0., droppath=0., downsample=False):
        super().__init__()
        self.dim = in_channels
        self.out_channels = out_channels
        self.input_resolution = input_resolution

        self.blocks = nn.ModuleList()
        for i in range(2):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=in_channels, input_resolution=input_resolution,
                    num_heads=num_heads, window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    dropout=dropout, attention_dropout=attention_dropout,
                    droppath=droppath[i] if isinstance(droppath, list) else droppath))

        self.cnn = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=(1, 1),
                             stride=(1, 1),
                             )

        if downsample:
            self.downsample = PatchMerging(input_resolution, dim=in_channels, out_channels=out_channels)
        else:
            self.downsample = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape([B, C, H * W])
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
            x = x.permute(0, 2, 1)
            x = x.reshape([B, self.out_channels, H // 2, W // 2])
        else:
            x = x.permute(0, 2, 1)
            x = x.reshape([B, C, H, W])
            x = self.cnn(x)
        return x

##

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, act_layer=nn.LeakyReLU(inplace=True), norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.convs = nn.Sequential(
            conv3x3(input_dim, output_dim), norm_layer(output_dim), act_layer(),
            conv3x3(output_dim, output_dim), norm_layer(output_dim), act_layer()
        )
        self.shortcut = nn.Conv2d(input_dim, output_dim, kernel_size=(1, 1)) if input_dim != output_dim \
            else nn.Identity()

    def forward(self, x):
        return self.convs(x) + self.shortcut(x)


class SwinTransformerResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, input_resolution):
        super().__init__()
        self.swinTs = nn.Sequential(
            SwinT(in_channels=input_dim, out_channels=output_dim, input_resolution=input_resolution, num_heads=8,
                  window_size=7, downsample=False),
            SwinT(in_channels=output_dim, out_channels=output_dim, input_resolution=input_resolution, num_heads=8,
                  window_size=7, downsample=False)
        )
        self.shortcut = nn.Conv2d(input_dim, output_dim, kernel_size=(1, 1)) if input_dim != output_dim \
            else nn.Identity()

    def forward(self, x):
        try:
            x1 = self.swinTs(x)
        except:

            bs, c, h, w = x.shape

            m = max(h, w)

            result = []
            for i in range(14, m+1):

                if i % 2 == 0 and i % 7 == 0:
  
                    result.append(i)
            h1 = result[abs(np.array(result) - h).argmin()]
            w1 = result[abs(np.array(result) - w).argmin()]

            x1 = F.interpolate(x, size=(h1, w1))

            x1 = self.swinTs(x1)

            x1 = F.interpolate(x1, size=(h, w)) 

        return x1 +  self.shortcut(x)



class Down(nn.Module):
    def __init__(self, input_dim, output_dim, input_resolution=None, ratio=2, layer='conv'):
        super().__init__()
        if input_resolution:
            self.res_block = SwinTransformerResidualBlock(input_dim, output_dim, input_resolution)
        else:
            self.res_block = ResidualBlock(input_dim, output_dim)

        if layer == 'conv':
            self.down = nn.Sequential(
                nn.Conv2d(output_dim, output_dim, kernel_size=(ratio + 1, ratio + 1), stride=(ratio, ratio), padding=(1, 1)), 
                nn.LeakyReLU(inplace=True)
            )
        else:
            self.down = nn.AvgPool2d(kernel_size=ratio, stride=ratio) 
    def forward(self, x):
        x = self.res_block(x)
        x = self.down(x)
        return x


class ConvResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvResidualBlock, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_dim)

        # Shortcut connection
        # If input_dim != output_dim, we need to match the dimensions using a 1x1 convolution
        self.shortcut = nn.Conv2d(input_dim, output_dim, kernel_size=1) if input_dim != output_dim else None

    def forward(self, x):
        identity = x
        x = self.conv1(x)  # Apply the first convolution
        x = self.bn1(x)  # Batch normalization
        x = self.relu(x)  # ReLU activation
        x = self.conv2(x)  # Apply the second convolution
        x = self.bn2(x)  # Batch normalization

        # Adjust the identity shortcut if necessary
        if self.shortcut is not None:
            identity = self.shortcut(identity)

        # Skip connection
        x += identity  # Add the shortcut to the output
        x = self.relu(x)  # Final ReLU activation
        return x



class Up(nn.Module):
    def __init__(self, input_dim, output_dim, input_resolution=None, ratio=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=ratio, mode='nearest')
        self.reduce = nn.Conv2d(input_dim, input_dim // 2, kernel_size=(1,1))
        self.ConvResidualBlock=ConvResidualBlock(input_dim, output_dim)
        if input_resolution:
            self.res_block = SwinTransformerResidualBlock(input_dim, output_dim, input_resolution)
        else:
            self.res_block = ResidualBlock(input_dim, output_dim)

    def forward(self, x, r):
        x = self.up(x)
        x = self.reduce(x)

        try:
            x = torch.cat((x, r), dim=1)
        except:
            # print(r.shape)
            x = torch.narrow(x, 2, 0, r.shape[2])
            x = torch.narrow(x, 3, 0, r.shape[3])
            x = torch.cat((x, r), dim=1)
        x = self.ConvResidualBlock(x)
        return x


class IdentityConv(nn.Module):
    def __init__(self, in_channels):
        super(IdentityConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

 
        with torch.no_grad():
            self.conv.weight.fill_(0)
            self.conv.weight[:, :, 0, 0] = 1 

    def forward(self, x):
        return self.conv(x)



class SpatialAttention(torch.nn.Module):

  def __init__(self, input_dim):
    super().__init__()
    self.id= IdentityConv(input_dim)
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self,f ,fe ):
    cap = torch.mean(fe, dim=1)
    w_avg = self.sigmoid(cap)
    w_avg = w_avg.unsqueeze(1)
    cmp = torch.max(fe, dim=1)[0]
    w_max = self.sigmoid(cmp)
    w_max = w_max.unsqueeze(1)
    x_cat_w = f * w_avg * w_max
    x_cat_w = self.id(x_cat_w)
    return x_cat_w





class my_Unet(nn.Module):
    def __init__(self, input_dim, output_dim, num_features, input_resolutions=None, sobel =True): 
        super().__init__()
        self.sobel = sobel
        self.num_class = output_dim

        # self.id1 = Identity()
        # self.id2 = Identity()
        # self.id3 = Identity()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

        self.downsample_2 = nn.MaxPool2d(2, 2)
        self.downsample_4 = nn.MaxPool2d(4, 4)
        self.sa1 = SpatialAttention(32)
        self.sa2 = SpatialAttention(64)
        self.sa3 = SpatialAttention(128)
        self.conv_1 = nn.Sequential(
            conv3x3(input_dim, 50), nn.BatchNorm2d(50), nn.LeakyReLU(0.05),
            conv3x3(50, num_features[0]), nn.BatchNorm2d(num_features[0]), nn.LeakyReLU(0.05)
        )
        if input_resolutions:
            self.encoder_x = nn.ModuleList([  # x个encoder, x=len(num_features)-1
                Down(in_dim, out_dim, input_resolution=resolution) for in_dim, out_dim, resolution in
                zip(num_features, num_features[1:], input_resolutions)
            ])

            # num_features = num_features + [0, 0, 1, 0, 0]

            self.decoder_x = nn.ModuleList([  # x个decoder, x=len(num_features)-1
                Up(in_dim, out_dim, input_resolution=resolution) for in_dim, out_dim, resolution in
                zip(reversed(num_features), reversed(num_features[:-1]), reversed(input_resolutions))
            ])
        else:
            self.encoder_x = nn.ModuleList([  # x个encoder, x=len(num_features)-1
                Down(in_dim, out_dim) for in_dim, out_dim in zip(num_features, num_features[1:]) # 用for循环来创建, 放到ModuleList里面
            ])

            self.decoder_x = nn.ModuleList([  # x个decoder, x=len(num_features)-1
                Up(in_dim, out_dim) for in_dim, out_dim in zip(reversed(num_features), reversed(num_features[:-1]))
            ])

        self.conv_out = conv3x3(num_features[0], output_dim)

        if self.sobel:
            print("----------use sobel-------------")
            self.sobel_x1, self.sobel_y1 = get_sobel(32, 1)
            self.sobel_x2, self.sobel_y2 = get_sobel(64, 1)
            self.sobel_x3, self.sobel_y3 = get_sobel(128, 1)
            self.sobel_x4, self.sobel_y4 = get_sobel(512, 1)

        self.erb_db_1 = ERB(64, 1)
        self.erb_db_2 = ERB(128, 1)
        self.erb_db_3 = ERB(256, 1)
        self.erb_db_4 = ERB(512, 1)

        self.erb_trans_1 = ERB(1, 1)
        self.erb_trans_2 = ERB(1, 1)
        self.erb_trans_3 = ERB(1, 1)

    def forward(self, x):
        x = self.conv_1(x)
        e_outputs = []    
        sobel_input = []


        for down in self.encoder_x:
            e_outputs.append(x)
            x = down(x)
            sobel_input.append(x)

        c1, c2, c3, c4 = sobel_input
        f1, f2, f3, f4 = e_outputs

        if self.sobel:
            f1e = run_sobel(self.sobel_x1, self.sobel_y1, f1)#64
            f2e = run_sobel(self.sobel_x2, self.sobel_y2, f2)#128
            f3e = run_sobel(self.sobel_x3, self.sobel_y3, f3)#256
        else:
            res1 = self.erb_db_1(self.downsample_2(c1))
            res1 = self.erb_trans_1(res1 + self.erb_db_2(c2))
            res1 = self.erb_trans_2(res1 + self.upsample(self.erb_db_3(c3)))
            res1 = self.erb_trans_3(res1 + self.upsample_4(self.erb_db_4(c4)), relu=False)

        # e_outputs[0] = self.id1(e_outputs[0])
        # e_outputs[1] = self.id2(e_outputs[1])
        # e_outputs[2] = self.id3(e_outputs[2])

        f1e = self.sa1(f1, f1e)
        f2e = self.sa2(f2, f2e)
        f3e = self.sa3(f3, f3e)
        e_outputs[0] = f1e
        e_outputs[1] = f2e
        e_outputs[2] = f3e
        for up, r in zip(self.decoder_x, reversed(e_outputs)):  
            x = up(x, r) 

        x = self.conv_out(x)

        if x.shape[1] == 1:
            x = nn.LeakyReLU(inplace=True)(x)
        return x


def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input


def get_sobel(in_chan, out_chan):
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)

    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y

class ERB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ERB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, relu=True):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        if relu:
            return self.relu(x + res)
        else:
            return x+res


if __name__ == '__main__':

    resolutions = [(252, 252), (126, 126), (56, 56), (28, 28)]
    model = my_Unet(3, 6, [32, 64, 128, 256, 512], input_resolutions=resolutions).cpu()
    x = torch.randn(1,3,256,256)
    y = model(x)
    print(y.shape)



