import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models.vgg as models
from models.networks.base_network import BaseNetwork
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d

class ADGANPPGenerator(BaseNetwork):
    def __init__(self, opt, activ='relu', pad_type='reflect'):
        super(ADGANPPGenerator, self).__init__()
        self.opt = opt
        dim = 64
        style_dim = 512
        if self.opt.stage == '1':
            n_downsample = 2
            input_dim = 3
            SP_input_nc = 1   
            self.enc_style = VggStyleEncoder(3, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type, stage='1')
            n_res = 6
            self.enc_content = ContentEncoder(n_downsample, n_res, SP_input_nc, dim, 'in', activ, pad_type=pad_type)
            self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim + 1, 3, res_norm='adain', activ=activ, pad_type=pad_type)
            self.fc = LinearBlock(style_dim, style_dim, norm='none', activation=activ)
            mlp_dim = 256
            self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)
        if self.opt.stage == '2':
            input_dim = 3
            n_downsample = 3   
            n_res = 4
            SP_input_nc = 3
            nf =  32
            self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
            self.SegINBlock_0 = SegINBlock(16 * nf, 16 * nf, opt)
            self.SegINBlock_1 = SegINBlock(16 * nf, 16 * nf, opt)
            self.SegINBlock_2 = SegINBlock(16 * nf, 16 * nf, opt)
            self.SegINBlock_3 = SegINBlock(16 * nf, 8 * nf, opt)
            self.SegINBlock_4 = SegINBlock(8 * nf, 4 * nf, opt)
            self.SegINBlock_5 = SegINBlock(4 * nf, 1 * nf, opt)
            self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)
            self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, sem, img):
        if len(sem) == 3 and sem[2] == 'transfer':
            content = self.enc_content(sem[0])
            sem_guidance = torch.ones(content.shape[0], 1, content.shape[2], content.shape[3], dtype = torch.float).mul(self.opt.part_idx).cuda()
            content = torch.cat((content,sem_guidance), 1)
            interpolation_ratio = 0.0
            style_orig = self.enc_style(img[0], sem[0])
            style = self.enc_style(img[1], sem[1])
            style = style * (1-interpolation_ratio) + style_orig * interpolation_ratio
            style = self.fc(style.view(style.size(0), -1))
            style = torch.unsqueeze(style, 2)
            style = torch.unsqueeze(style, 3)
            images_recon = self.decode(content, style)
            return images_recon  
        else:
            if self.opt.stage == '1':
                content = self.enc_content(sem)
                sem_guidance = torch.ones(content.shape[0], 1, content.shape[2], content.shape[3], dtype = torch.float).mul(self.opt.part_idx).cuda()
                content = torch.cat((content,sem_guidance), 1)
                style = self.enc_style(img, sem)
                style = self.fc(style.view(style.size(0), -1))
                style = torch.unsqueeze(style, 2)
                style = torch.unsqueeze(style, 3)
                images_recon = self.decode(content, style)
                return images_recon
            elif self.opt.stage == '2':
                content = self.enc_content(img)
                content = self.SegINBlock_0(content, sem)
                content = self.SegINBlock_1(content, sem)
                content = self.SegINBlock_2(content, sem)
                content = self.up(content)
                content = self.SegINBlock_3(content, sem)
                content = self.up(content)
                content = self.SegINBlock_4(content, sem)
                content = self.up(content)
                content = self.SegINBlock_5(content, sem)
                images_recon = self.conv_img(F.leaky_relu(content, 2e-1)) + img
                images_recon = F.tanh(images_recon)
                return images_recon 


    def decode(self, content, style):
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params



class VggStyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type, stage):
        super(VggStyleEncoder, self).__init__()
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load('/checkpoints/vgg19-dcbb9e9d.pth'))
        self.vgg = vgg19.features
        self.stage = stage
        for param in self.vgg.parameters():
            param.requires_grad_(False)

        self.conv1 = Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type) 
        dim = dim*2
        self.conv2 = Conv2dBlock(dim , dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)
        dim = dim*2
        self.conv3 = Conv2dBlock(dim , dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type) 
        dim = dim * 2
        self.conv4 = Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)  
        dim = dim * 2

        self.model = []
        self.model += [nn.AdaptiveAvgPool2d(1)] 
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)

    def get_features(self,image, model, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1'}
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def texture_enc(self, x):
        if x.shape[1] == 4:
            sty_fea = self.get_features(x[:,:-1,:,:], self.vgg)
        else:
            sty_fea = self.get_features(x, self.vgg)
        x = self.conv1(x)
        x = torch.cat([x, sty_fea['conv1_1']], dim=1)
        x = self.conv2(x)
        x = torch.cat([x, sty_fea['conv2_1']], dim=1)
        x = self.conv3(x)
        x = torch.cat([x, sty_fea['conv3_1']], dim=1)
        x = self.conv4(x)
        x = torch.cat([x, sty_fea['conv4_1']], dim=1)
        x = self.model(x)
        return x

    def forward(self, x, sem):    
        out = self.texture_enc(x) 
        return out        


class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] 
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] 
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

class SegINBlock(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        self.norm_0 = SegIN(fin, opt.semantic_nc)
        self.norm_1 = SegIN(fmiddle, opt.semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SegIN(fin, opt.semantic_nc)


    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
   


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)



class SegIN(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.BatchNorm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_alpha = nn.Conv2d(256, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(256, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, seg):
        seg = F.interpolate(seg, size=x.size()[2:], mode='nearest')
        shared_feat = self.mlp_shared(seg)
        alpha = self.mlp_alpha(shared_feat)
        beta = self.mlp_beta(shared_feat) 

        normalized = self.BatchNorm(x)
        out = normalized * (1 + alpha) + beta

        return out