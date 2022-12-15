import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block_my(nn.Module):
    def __init__(self, in_channels, out_channels,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1):
        """

        Arguments:
        ----------
        in_channels: int
            The number of input channels.

        out_channels: int
            The number of output channels.

        """
        super(conv_block_my, self).__init__()

        nn_Conv2d = lambda in_channels, out_channels: nn.Conv2d(
            in_channels = in_channels, out_channels = out_channels,
            kernel_size = 3, stride=1, padding = 1, dilation=1)

        self.conv_block_my = nn.Sequential(
            nn_Conv2d(in_channels,  out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn_Conv2d(out_channels, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, inputs):
        """
        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, in_channels,  height, width]
            Input to the convolutional block.

        Returns:
        --------
        outputs: another 4-th order tensor of size
            [batch_size, out_channels, height, width]
            Output of the convolutional block.

        """
        outputs = self.conv_block_my(inputs)
        return outputs

# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps = 1e-8, affine = True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)                                  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)                          # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
#-----------------------------------------------
#                Gated ARMAConvBlock
#-----------------------------------------------
class GatedARMAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding = 0, w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none'):
        super(GatedARMAConv2d, self).__init__()

        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
     
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace = True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace = True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace = True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

       
        self.conv2d = conv_block_my(in_channels, out_channels, w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)
        self.mask_conv2d = conv_block_my(in_channels, out_channels, w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1,activation = 'lrelu', norm = 'none',scale_factor = 2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedARMAConv2d(in_channels, out_channels,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1, activation = activation, norm = norm)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.gated_conv2d(x)
        return x




class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super(SpatialGatingUnit, self).__init__()

        self.norm = nn.LayerNorm(d_ffn // 2)
        self.proj = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = v.permute(0, 2, 1)
        v = self.proj(v)
        v = v.permute(0, 2, 1)
        return u * v


class GatingMlpBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, survival_prob):
        super(GatingMlpBlock, self).__init__()

        self.norm = nn.LayerNorm(d_model)
        self.proj_1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SpatialGatingUnit(d_ffn, seq_len)
        self.proj_2 = nn.Linear(d_ffn // 2, d_model)
        self.prob = survival_prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))

    def forward(self, x):
        if self.training and torch.equal(self.m.sample(), torch.zeros(1)):
            return x
        shorcut = x.clone()
        x = self.norm(x)
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shorcut

class gMLP_attention(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len, survival_prob):
        super(gMLP_attention, self).__init__()
        self.gmlp_block = GatingMlpBlock(d_model, d_ffn, seq_len, survival_prob)

    def forward(self, x):
        channel_pool = x.mean(1)
        gMLP_out = self.gmlp_block(channel_pool)
        attention_out = gMLP_out * x
        return attention_out

class My_net(nn.Module):
    def __init__(self, in_channels = 6, out_channels = 1, factor = 1):
        """
        Arguments:
        ----------
        in_channels: int
            The number of input channels.

        out_channels: int
            The number of output channels.


        """
        super(My_net,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size = 2)
        self.Conv1 = GatedARMAConv2d(in_channels, 64//factor,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)

        self.Conv2 = GatedARMAConv2d(64//factor,  128//factor,   w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)
        self.Conv3 = GatedARMAConv2d(128//factor, 256//factor,   w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)
        self.Conv4 = GatedARMAConv2d(256//factor, 512//factor,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)
        self.Conv5 = GatedARMAConv2d(512//factor, 1024//factor,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)

        self.dil_conv1 = GatedARMAConv2d(1024//factor, 1024//factor,  w_stride = 1, w_dilation = 2, a_stride = 1, a_dilation = 2)
        self.dil_conv2 = GatedARMAConv2d(1024//factor, 1024//factor,  w_stride = 1, w_dilation = 4, a_stride = 1, a_dilation = 4)
        self.dil_conv3 = GatedARMAConv2d(1024//factor, 1024//factor,  w_stride = 1, w_dilation = 8, a_stride = 1, a_dilation = 8)
        self.dil_conv4 = GatedARMAConv2d(1024//factor, 1024//factor,  w_stride = 1, w_dilation = 16, a_stride = 1, a_dilation = 16)

        self.Up5 = TransposeGatedConv2d(1024//factor, 512//factor)
        self.Up_conv5 = GatedARMAConv2d(1024//factor, 512//factor,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)

        self.Up4 = TransposeGatedConv2d(512//factor, 256//factor)
        self.Up_conv4 = GatedARMAConv2d(512//factor, 256//factor,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)
        
        self.Up3 = TransposeGatedConv2d(256//factor, 128//factor)
        self.Up_conv3 = GatedARMAConv2d(256//factor, 128//factor,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)
        
        self.Up2 = TransposeGatedConv2d(128//factor, 64//factor)
        self.Up_conv2 = GatedARMAConv2d(64//factor, 64//factor,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)

        self.Conv_1x1 = nn.Conv2d(64//factor, out_channels, 1)


        self.p2_Conv1 = GatedARMAConv2d(in_channels, 64//factor,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)

        self.p2_Conv2 = GatedARMAConv2d(64//factor,  128//factor,   w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)
        self.p2_Conv3 = GatedARMAConv2d(128//factor, 256//factor,   w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)
        self.p2_Conv4 = GatedARMAConv2d(256//factor, 512//factor,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)
        self.p2_Conv5 = GatedARMAConv2d(512//factor, 1024//factor,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)

        self.p2_dil_conv1 = GatedARMAConv2d(1024//factor, 1024//factor,  w_stride = 1, w_dilation = 2, a_stride = 1, a_dilation = 2)
        self.p2_dil_conv2 = GatedARMAConv2d(1024//factor, 1024//factor,  w_stride = 1, w_dilation = 4, a_stride = 1, a_dilation = 4)
        self.p2_dil_conv3 = GatedARMAConv2d(1024//factor, 1024//factor,  w_stride = 1, w_dilation = 8, a_stride = 1, a_dilation = 8)
        self.p2_dil_conv4 = GatedARMAConv2d(1024//factor, 1024//factor,  w_stride = 1, w_dilation = 16, a_stride = 1, a_dilation = 16)

        self.p2_Up5 = TransposeGatedConv2d(1024//factor, 512//factor)
        self.gmlp_attn1 = gMLP_attention(d_model = 32, d_ffn = 64, seq_len = 32, survival_prob = [1, 0])
        self.p2_Up_conv5 = GatedARMAConv2d(1024//factor, 512//factor,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)

        self.p2_Up4 = TransposeGatedConv2d(512//factor, 256//factor)
        self.gmlp_attn2 = gMLP_attention(d_model = 64, d_ffn = 128, seq_len = 64, survival_prob = [1, 0])
        self.p2_Up_conv4 = GatedARMAConv2d(512//factor, 256//factor,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)
        
        self.p2_Up3 = TransposeGatedConv2d(256//factor, 128//factor)
        self.gmlp_attn3 = gMLP_attention(d_model = 128, d_ffn = 256, seq_len = 128, survival_prob = [1, 0])
        self.p2_Up_conv3 = GatedARMAConv2d(256//factor, 128//factor,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)
        
        self.p2_Up2 = TransposeGatedConv2d(128//factor, 64//factor)
        self.gmlp_attn4 = gMLP_attention(d_model = 256, d_ffn = 512, seq_len = 256, survival_prob = [1, 0])
        self.p2_Up_conv2 = GatedARMAConv2d(128//factor, 64//factor,  w_stride = 1, w_dilation = 1, a_stride = 1, a_dilation = 1)

        self.p2_Conv_1x1 = nn.Conv2d(64//factor, out_channels, 1)


        

    def forward(self, in1, in2):
        """
                
        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, in_channels, height, width]
           

        Returns:
        --------
        outputs: a 4-th order tensor of size
            [batch_size, out_channels, height, width]
            

        """

        # encoding path
        x = torch.cat((in1,in2),dim=1)
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        dil1 = self.dil_conv1(x5)
        dil2 = self.dil_conv2(dil1)
        dil3 = self.dil_conv3(dil2)
        dil4 = self.dil_conv4(dil3)

        # decoding + concat path
        d5 = self.Up5(dil4)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        second_masked_img = in1 * (1 - in2) + d1 * in2
        p2_x = torch.cat((second_masked_img,in2),dim=1)


        p2_x1 = self.p2_Conv1(p2_x)

        p2_x2 = self.Maxpool(p2_x1)
        p2_x2 = self.p2_Conv2(p2_x2)
        
        p2_x3 = self.Maxpool(p2_x2)
        p2_x3 = self.p2_Conv3(p2_x3)

        p2_x4 = self.Maxpool(p2_x3)
        p2_x4 = self.p2_Conv4(p2_x4)

        p2_x5 = self.Maxpool(p2_x4)
        p2_x5 = self.p2_Conv5(p2_x5)

        p2_dil1 = self.p2_dil_conv1(p2_x5)
        p2_dil2 = self.p2_dil_conv2(p2_dil1)
        p2_dil3 = self.p2_dil_conv3(p2_dil2)
        p2_dil4 = self.p2_dil_conv4(p2_dil3)

        # decoding + concat through SPL path
        p2_d5 = self.p2_Up5(p2_dil4)
        p2_x4_skip = self.gmlp_attn1(p2_x4)
        p2_d5 = torch.cat((p2_x4_skip,p2_d5),dim=1)
        
        p2_d5 = self.p2_Up_conv5(p2_d5)
        
        p2_d4 = self.p2_Up4(p2_d5)
        p2_x3_skip = self.gmlp_attn2(p2_x3)
        p2_d4 = torch.cat((p2_x3_skip,p2_d4),dim=1)
        p2_d4 = self.p2_Up_conv4(p2_d4)

        p2_d3 = self.p2_Up3(p2_d4)
        p2_x2_skip = self.gmlp_attn3(p2_x2)
        p2_d3 = torch.cat((p2_x2_skip,p2_d3),dim=1)
        p2_d3 = self.p2_Up_conv3(p2_d3)

        p2_d2 = self.p2_Up2(p2_d3)
        p2_x1_skip = self.gmlp_attn4(p2_x1)
        p2_d2 = torch.cat((p2_x1_skip,p2_d2),dim=1)
        p2_d2 = self.p2_Up_conv2(p2_d2)

        p2_d1 = self.p2_Conv_1x1(p2_d2)


        return d1, p2_d1