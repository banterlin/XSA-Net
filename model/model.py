import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# from focal_loss.focal_loss import FocalLoss as FL

from run.utils import Show_Samples
from run.Args import args
# from ResNet3D import generate_model as resnet3D
from model.ResNet3D import generate_model

def ini_weights(module_list:list):
    for m in module_list:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Res_3D_Encoder(nn.Module):
    def __init__(self, kargs = args, **kwargs) -> None:
        super().__init__()
        layer = kargs.model_depth
        if layer == 50:
            self.model = generate_model(kargs)
            self.feature_channel = 2048
        elif layer == 18:
            self.model = generate_model(kargs)
            self.feature_channel = 512

            # self.model = models.video.r3d_18(pretrained=False)
            # self.feature_channel = 256

    def forward(self, x, squeeze_to_vector = False, pool="max"):
        # input: b, c, d, h, w -> b, (d), c
        assert x.dim() == 5, 'Wrong input dimension'
        if pool == "avg":
            pool_func = F.adaptive_avg_pool3d
        else:
            pool_func = F.adaptive_max_pool3d
        x = self.model(x)
        if squeeze_to_vector:
            x = pool_func(x, 1)
            x = torch.flatten(x, start_dim=1)
        else:
            x = x.transpose(1,2) # b, d, c, h, w
            x = pool_func(x, (self.feature_channel,1,1))
            x = torch.flatten(x, start_dim=2)
        return x

class Res_2D_Encoder(nn.Module):
    def __init__(self, kargs, *args,**kwargs) -> None:
        raise Exception("This module is deprecated")
        super().__init__()
        self.kargs = kargs
        if kargs.model_depth == 50:
            self.model = models.resnet50(weights="IMAGENET1K_V2")
            self.feature_channel = 2048
        elif kargs.model_depth == 18:
            self.model = models.resnet18(weights="IMAGENET1K_V1")
            self.feature_channel = 512
        
    def forward(self, input, squeeze_to_vector = False, pool="avg"):
        # x.shape = (1, 1, slice, h, w)
        assert input.dim() == 5, 'Wrong input dimension'
        assert input.shape[0]==1 and input.shape[1] == 1, "only support batchsize=1, but got shape"+str(input.shape) 
        x = torch.cat((input, input, input), dim=1).transpose(1, 2).squeeze(0) # slice, c, h, w
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.unsqueeze(0) # 1, slice, c, h, w
        if squeeze_to_vector:
            x = x.transpose(1,2) #1, c, d, h, w
            x = F.adaptive_max_pool3d(x, 1)
            x = torch.flatten(x, start_dim=1)
        else:
            #-> 1, slice, c, 1
            x = F.adaptive_max_pool3d(x, (self.feature_channel,1,1))
            x = torch.flatten(x, start_dim=2)
        return x

class Eff_2D_Encoder(nn.Module):
    def __init__(self, kargs, *args, **kwargs) -> None:
        raise Exception("This module is deprecated")
        super().__init__(*args, **kwargs)
        self.model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        self.feature_channel = 1280

    def forward(self, input, squeeze_to_vector = False, pool="avg"):
        # x.shape = (1, 1, slice, h, w)
        assert input.dim() == 5, 'Wrong input dimension'
        assert input.shape[0]==1 and input.shape[1] == 1, "only support batchsize=1, but got shape"+str(input.shape) 
        x = torch.cat((input, input, input), dim=1).transpose(1, 2).squeeze(0) # slice, c, h, w
        x = self.model.features(x)
        s, c, h, w = x.shape
        x = x.unsqueeze(0)
        if squeeze_to_vector:
            #->b, c, 1
            x = x.transpose(1,2)
            x = F.adaptive_max_pool3d(x, 1)
            x = torch.flatten(x, start_dim=1)
        else:
            #-> b, slice, c
            x = F.adaptive_max_pool3d(x, (self.feature_channel,1,1))
            x = torch.flatten(x, start_dim=2)
        return x
# One block of SA
# class Self_Plane_Att(nn.Module):
#     def __init__(self, embed_dim, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.emb_dim = embed_dim
#         self.mq = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.mk = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.mv = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.norm = nn.LayerNorm(embed_dim)
#         ini_weights(self.modules())
#
#     def forward(self, x):
#         # (batch, channel, d)
#         res = x
#         q = self.mq(x)
#         k = self.mk(x).permute(0, 2, 1)
#         v = self.mv(x)
#         att = torch.matmul(q, k) / np.sqrt(self.emb_dim)
#         att = torch.softmax(att, dim=-1)
#         out = torch.matmul(att, v)
#         self.attmap = att.detach().cpu()
#         f = self.norm(out + res)
#         f = f.transpose(1, 2)
#         f = F.adaptive_max_pool1d(f, 1)
#         f = torch.flatten(f, start_dim=1)
#         return f

class Self_Att_Block(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.emb_dim = embed_dim
        self.mq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # Accept [B, D] or [B, N, D]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # -> [B, 1, D]
        elif x.dim() == 3:
            pass
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

        res = x
        q = self.mq(x)
        k = self.mk(x).transpose(1, 2)
        v = self.mv(x)

        att = torch.matmul(q, k) / np.sqrt(self.emb_dim)
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v)
        f = self.norm(out + res)  # [B, N, D]
        return f


class Self_Plane_Att(nn.Module):
    """Stacked self-attention layers for 1D or sequence input."""
    def __init__(self, embed_dim, num_layers=1):
        super().__init__()
        self.emb_dim = embed_dim
        self.layers = nn.ModuleList([Self_Att_Block(embed_dim) for _ in range(num_layers)])
    def forward(self, x):
        # x: [B, D] or [B, N, D]
        for layer in self.layers:
            x = layer(x)  # always returns [B, N, D]

        # Global pooling across sequence length
        x = x.transpose(1, 2)         # [B, D, N]
        x = F.adaptive_max_pool1d(x, 1)  # [B, D, 1]
        x = torch.flatten(x, start_dim=1)  # [B, D]
        return x

class Self_Att_Block_CLI(nn.Module):
    # def __init__(self, embed_dim=1536):
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.emb_dim = embed_dim
        self.mq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # Accept [B, D] or [B, N, D]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # -> [B, 1, D]
        elif x.dim() == 3:
            pass
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

        res = x
        q = self.mq(x)
        k = self.mk(x).transpose(1, 2)
        v = self.mv(x)

        att = torch.matmul(q, k) / np.sqrt(self.emb_dim)
        att = torch.softmax(att, dim=-1)
        out = torch.matmul(att, v)
        f = self.norm(out + res)  # [B, N, D]
        return f


class Self_Plane_Att_CLI(nn.Module):
    """Stacked self-attention layers for 1D or sequence input."""
    # def __init__(self, embed_dim=1536, num_layers=10):
    def __init__(self, embed_dim=1024, num_layers=10):
        super().__init__()
        self.emb_dim = embed_dim
        self.layers = nn.ModuleList([Self_Att_Block_CLI() for _ in range(num_layers)])
    def forward(self, x):
        # x: [B, D] or [B, N, D]
        for layer in self.layers:
            x = layer(x)  # always returns [B, N, D]

        # Global pooling across sequence length
        x = x.transpose(1, 2)         # [B, D, N]
        x = F.adaptive_max_pool1d(x, 1)  # [B, D, 1]
        x = torch.flatten(x, start_dim=1)  # [B, D]
        return x


class ClinicalNormalizer(nn.Module):
    """
    Normalizes and projects clinical features to match the embedding statistics of imaging tokens.
    """
    def __init__(self, in_dim=10, out_dim=512, norm_type="batch"):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        if norm_type == "layer":
            self.norm = nn.LayerNorm(out_dim)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_dim)
        else:
            self.norm = nn.Identity()
    def forward(self, cli):
        # Handle both [B, D] and [B, 1, D]
        if cli.dim() == 3:
            cli = cli.squeeze(1)
        cli = cli.float()
        # Optional: standardize per batch to avoid scale drift
        # mean = cli.mean(dim=0, keepdim=True)
        # std = cli.std(dim=0, keeping=True) + 1e-6
        # cli = (cli - mean) / std
        # Project to embedding dimension and normalize
        cli = self.proj(cli)
        cli = self.norm(cli)
        return cli

def ini_weights(modules):
    for m in modules:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

class Co_Plane_Att_Block(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.emb_dim = embed_dim
        self.mq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mk1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mk2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mv1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mv2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        ini_weights(self.modules())

    def forward(self, main_f, co_f1, co_f2):
        res = main_f
        q = self.mq(main_f)
        k1 = self.mk1(co_f1).permute(0, 2, 1)
        k2 = self.mk2(co_f2).permute(0, 2, 1)
        v1 = self.mv1(co_f1)
        v2 = self.mv2(co_f2)

        att1 = torch.matmul(q, k1) / np.sqrt(self.emb_dim)
        att2 = torch.matmul(q, k2) / np.sqrt(self.emb_dim)
        att1 = torch.softmax(att1, -1)
        att2 = torch.softmax(att2, -1)

        out1 = torch.matmul(att1, v1)
        out2 = torch.matmul(att2, v2)

        f = self.norm(0.5 * (out1 + out2) + res)
        return f  # keep the sequence form for stacking

class Co_Plane_Att(nn.Module):
    def __init__(self, embed_dim, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([Co_Plane_Att_Block(embed_dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, main_f, co_f1, co_f2):
        f = main_f
        for layer in self.layers:
            f = layer(f, co_f1, co_f2)
        # collapse spatial dimension only once at the end
        f = f.transpose(1, 2)
        f = F.adaptive_max_pool1d(f, 1)
        f = torch.flatten(f, start_dim=1)
        f = self.final_norm(f)
        return f

# One layer design
# class Co_Plane_Att(nn.Module):
#     def __init__(self, embed_dim, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.emb_dim = embed_dim
#         self.mq = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.mk1 = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.mk2 = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.mv1 = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.mv2 = nn.Linear(embed_dim, embed_dim, bias=False)
#         self.norm = nn.LayerNorm(embed_dim)
#         ini_weights(self.modules())
#
#     def forward(self, main_f, co_f1, co_f2):
#         # (batch, channel, d)
#         res = main_f
#         q = self.mq(main_f)
#         k1 = self.mk1(co_f1).permute(0, 2, 1)
#         k2 = self.mk2(co_f2).permute(0, 2, 1)
#         v1 = self.mv1(co_f1)
#         v2 = self.mv2(co_f2)
#         att1 = torch.matmul(q, k1)/np.sqrt(self.emb_dim)
#         att1 = torch.softmax(att1, -1)
#         att2 = torch.matmul(q, k2)/np.sqrt(self.emb_dim)
#         att2 = torch.softmax(att2, -1)
#         out1 = torch.matmul(att1, v1)
#         out2 = torch.matmul(att2, v2)
#         self.attmap1 = att1.detach().cpu()
#         self.attmap2 = att2.detach().cpu()
#         f = self.norm(0.5*(out1+out2)+res)
#         f = f.transpose(1, 2)
#         f = F.adaptive_max_pool1d(f, 1)
#         f = torch.flatten(f, start_dim=1)
#         return f


class Cross_Modal_Att(nn.Module):
    def __init__(self, feature_channel, kargs = args, **kwargs) -> None:
        super().__init__(**kwargs)
        self.transform_matrix = nn.Linear(2*feature_channel, feature_channel)
        self.norm = nn.BatchNorm1d(num_features=feature_channel)
        ini_weights([self.transform_matrix, self.norm])

    def forward(self, pdw_f, aux_f): #pdw feature and auxiliary modal
        assert aux_f.dim() == pdw_f.dim()
        add_f = pdw_f+aux_f
        sub_f = torch.cat((pdw_f, aux_f), dim=1)
        att_f = self.transform_matrix(sub_f)
        att_f = torch.relu(att_f)
        att_f = torch.softmax(att_f, -1)
        f = add_f*att_f
        return f

class Branch_Classifier(nn.Module):
    def __init__(self, classnum, feature_channel, dropout_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.classifiers = nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(feature_channel, classnum))
        ini_weights(self.classifiers)

    def forward(self, f):
        if f.dim() == 3:
            f = f.squeeze(2)
        return self.classifiers(f)

class Post_Classifier(nn.Module):
    def __init__(self, classnum, feature_channel, dropout_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.classifiers = nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(feature_channel*2, classnum))
        # self.classifiers = nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(1536, classnum))
        self.classifiers = nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(1024, classnum))
        ini_weights(self.classifiers)

    def forward(self, f):
        if f.dim() == 3:
            f = f.squeeze(2)
        return self.classifiers(f)

class Postnoca_Classifier(nn.Module):
    def __init__(self, classnum, feature_channel, dropout_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.classifiers = nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(feature_channel*4, classnum))
        self.classifiers = nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(feature_channel, classnum))
        ini_weights(self.classifiers)

    def forward(self, f):
        if f.dim() == 3:
            f = f.squeeze(2)
        return self.classifiers(f)

class ResNet4_Classifier(nn.Module):
    def __init__(self, classnum, feature_channel, dropout_rate, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.classifiers = nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(feature_channel*4, classnum))
        ini_weights(self.classifiers)

    def forward(self, f):
        if f.dim() == 3:
            f = f.squeeze(2)
        return self.classifiers(f)

class ResNet6_Classifier(nn.Module):
    def __init__(self, classnum, feature_channel, dropout_rate, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.classifiers = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(feature_channel * 6, classnum))
            ini_weights(self.classifiers)

    def forward(self, f):
            if f.dim() == 3:
                f = f.squeeze(2)
            return self.classifiers(f)
## Main function for xsanet
class XSA_Net(nn.Module):
    def __init__(self, backbone = 'ResNet', encoder_layer = 18, pretrain = True, parallel_device = True, kargs = args) -> None:
        super().__init__()
        self.kargs = kargs
        self.class_num = kargs.ClassNum
        self.para_device = parallel_device
        self.branch = kargs.active_branch
        # set parallel devices
        if torch.cuda.device_count() == 3 and self.para_device:
            self.device_list = ["cuda:%d"%x for x in range(torch.cuda.device_count())]
        else:
            self.device_list = ["cuda:0"]*3
        self.dropout_rate = 0.05
        # self.dropout_rate = 0.3
        self.backbone = kargs.backbone
        self.module_list = []
        self.__make_encoder__(self.backbone)
        self.__make_co_plane_att__()
        self.__make_cross_modal_att__()
        self.__make_classifier__()
        self.mining_conv = nn.Conv3d(1, 12, (12,12,12))
        ini_weights([self.mining_conv, self.multi_view_classifier])
        model_param = sum(np.prod(v.size()) for name, v in self.named_parameters()) / 1e6
        print('model param = %f MB'%model_param)


    def __make_encoder__(self, backbone_name):
        if backbone_name == 'ResNet':
            encoder_func = Res_2D_Encoder
        elif backbone_name == 'ResNet3D':
            encoder_func = Res_3D_Encoder
        elif backbone_name == "EffNet":
            encoder_func = Eff_2D_Encoder
        else:
            raise Exception('Wrong Backbone Name!')
        self.encoder_func = encoder_func
        if self.branch[0]:
            self.sag_enc = encoder_func(self.kargs).to(self.device_list[0])
            self.t2w_enc = encoder_func(self.kargs).to(self.device_list[0])
            if not self.kargs.no_cross_modal:
                self.t2w_enc = encoder_func(self.kargs).to(self.device_list[0])
        if self.branch[1]:
            self.cor_enc = encoder_func(self.kargs).to(self.device_list[1])
            self.t1w_enc = encoder_func(self.kargs).to(self.device_list[0])
            if not self.kargs.no_cross_modal:
                self.t1w_enc = encoder_func(self.kargs).to(self.device_list[0])
        if self.branch[2]:
            self.axi_enc = encoder_func(self.kargs).to(self.device_list[2])
            self.t1w_enc = encoder_func(self.kargs).to(self.device_list[0])
        return

    def __make_co_plane_att__(self):
        if self.kargs.no_co_att:
            return
        emb_dim = self.kargs.emb_dim
        if self.branch[0]:
            self.sag_att = Co_Plane_Att(emb_dim).to(self.device_list[0])
        if self.branch[1]:
            self.cor_att = Co_Plane_Att(emb_dim).to(self.device_list[1])
        if self.branch[2]:
            self.axi_att = Co_Plane_Att(emb_dim,num_layers=2).to(self.device_list[2])
            self.self_att = Self_Plane_Att(emb_dim,num_layers=2).to(self.device_list[2])
            self.cli_norm = ClinicalNormalizer().to(self.device_list[2])
            # self.final_self_att = Self_Plane_Att(10,num_layers=6).to(self.device_list[2])
            self.final_self_att = Self_Plane_Att_CLI(num_layers=8).to(self.device_list[2])
            # self.stack_att = SelfAttStack(emb_dim).to(self.device_list[2])
        return
    
    def __make_cross_modal_att__(self):
        if self.kargs.no_cross_modal:
            return
        if self.branch[0]:
            self.sag_cross_att = Cross_Modal_Att(self.sag_enc.feature_channel, self.kargs).to(self.device_list[0])
        if self.branch[1]:
            self.cor_cross_att = Cross_Modal_Att(self.cor_enc.feature_channel, self.kargs).to(self.device_list[1])
        if self.branch[2]:
            self.axi_cross_att = Cross_Modal_Att(self.axi_enc.feature_channel, self.kargs).to(self.device_list[1])
        return
    
    def __make_classifier__(self):
        if self.branch[0]:
            self.sag_classifier = Branch_Classifier(self.class_num, self.sag_enc.feature_channel, self.dropout_rate).to(self.device_list[0])
            self.res4_classifier = ResNet4_Classifier(self.class_num, self.sag_enc.feature_channel, self.dropout_rate).to(self.device_list[0])
            self.res6_classifier = ResNet6_Classifier(self.class_num, self.sag_enc.feature_channel, self.dropout_rate).to(
                self.device_list[0])
        if self.branch[1]:
            self.cor_classifier = Branch_Classifier(self.class_num, self.cor_enc.feature_channel, self.dropout_rate).to(self.device_list[1])
            self.noxa_classifier = Post_Classifier(self.class_num, self.cor_enc.feature_channel, self.dropout_rate).to(
                self.device_list[2])
        if self.branch[2]:
            self.axi_classifier = Branch_Classifier(self.class_num, self.axi_enc.feature_channel, self.dropout_rate).to(self.device_list[2])
            self.post_classifier = Post_Classifier(self.class_num, self.axi_enc.feature_channel, self.dropout_rate).to(self.device_list[2])
            self.noxa_classifier = Postnoca_Classifier(self.class_num, self.axi_enc.feature_channel, self.dropout_rate).to(
                self.device_list[2])
        self.multi_view_classifier = nn.Sequential(nn.Linear(12, 12)).to(self.device_list[2])
        return

    def __post_branch__(self, input, device = None): ## For preoperative ResNet Baseline
        cor_sag, cor_img, cor_axi, premask,t1post,postmask = input.unbind(dim=1)

        if device != None:
            cor_img = cor_img.to(device, non_blocking=True)
            cor_sag = cor_sag.to(device, non_blocking=True)
            cor_axi = cor_axi.to(device, non_blocking=True)
            premask = premask.to(device, non_blocking=True)
            t1post = t1post.to(device, non_blocking=True)
            postmask = postmask.to(device, non_blocking=True)

        main_f = torch.squeeze(self.sag_enc(cor_img))
            # with torch.no_grad():
        co_f1 = torch.squeeze(self.sag_enc(cor_sag))
        co_f2 = torch.squeeze(self.sag_enc(cor_axi))
        co_f3 = torch.squeeze(self.sag_enc(t1post))
        aux_f1 = self.t2w_enc(premask, squeeze_to_vector = True)
        aux_f2 = self.t2w_enc(postmask, squeeze_to_vector = True)
        concat_f = torch.cat((main_f, co_f1,co_f2,co_f3,aux_f1,aux_f2), 1)
        pred = self.res6_classifier(concat_f)

        return pred
    
    def __pre_branch__(self, input, device = None):
        # sag_img, cor_img, axi_img, _, t1_img = input
        cor_sag, cor_img, cor_axi, premask,_,_ = input.unbind(dim=1)
        # ["dwipre", "t2pre", "t1pre", "t1post"]

        if device != None:
            cor_img = cor_img.to(device, non_blocking=True)
            cor_sag = cor_sag.to(device, non_blocking=True)
            cor_axi = cor_axi.to(device, non_blocking=True)
            premask = premask.to(device, non_blocking=True)

        main_f = self.cor_enc(cor_img)
        # with torch.no_grad():
        co_f1 = self.cor_enc(cor_sag)
        co_f2 = self.cor_enc(cor_axi)
        pdw_f = self.cor_att(main_f, co_f1, co_f2)
# Else to here

        if self.kargs.no_cross_modal:
            aux_f = self.t1w_enc(premask, squeeze_to_vector = True)
            pred = self.noxa_classifier(torch.concat((pdw_f,aux_f),1))
        else:
            aux_f = self.t1w_enc(premask, squeeze_to_vector = True)
            cross_m_f = self.cor_cross_att(pdw_f, aux_f)
            pred = self.cor_classifier(cross_m_f)
        return pred
        
    def __final_branch__(self, input,cli, device = None): ## Postoperative data input
        cor_sag, cor_img, cor_axi, premask,t1post,postmask = input.unbind(dim=1)
        # ["dwipre", "t2pre", "t1pre", "preoperative masks","t1post","postopeartive masks"]

        if device != None:
            cor_img = cor_img.to(device, non_blocking=True)
            cor_sag = cor_sag.to(device, non_blocking=True)
            cor_axi = cor_axi.to(device, non_blocking=True)
            premask = premask.to(device, non_blocking=True)
            t1post = t1post.to(device, non_blocking=True)
            postmask = postmask.to(device, non_blocking=True)
            cli = cli.unsqueeze(0).to(device, non_blocking=True)
        main_f = self.axi_enc(cor_img)
        t1post_f = self.axi_enc(t1post)
        # with torch.no_grad():
        co_f1 = self.axi_enc(cor_sag)
        co_f2 = self.axi_enc(cor_axi)

        pre_aux_f = self.t1w_enc(premask, squeeze_to_vector=True)
        post_aux_f = self.t1w_enc(postmask, squeeze_to_vector=True)

        if self.kargs.no_cross_modal:
            pdw_f = self.axi_att(main_f, co_f1, co_f2)
            post_f = self.self_att(t1post_f)  # self-attention
            pred = self.noxa_classifier(torch.concat((pdw_f,post_f,pre_aux_f,post_aux_f),1))
        elif self.kargs.encoder_only:
            pred = self.post_classifier(torch.concat((torch.squeeze(main_f),
                                                      torch.squeeze(t1post_f), torch.squeeze(co_f1),
                                                     torch.squeeze(co_f2),pre_aux_f,post_aux_f), 1))
        elif self.kargs.no_post_img:
            pdw_f = self.axi_att(main_f, co_f1, co_f2)
            cross_pre_f = self.axi_cross_att(pdw_f, pre_aux_f)

            cli_tensors = torch.nan_to_num(cli, nan=0.0, posinf=1.0, neginf=-1.0)
            cli_tensors = torch.nn.functional.layer_norm(cli_tensors, cli_tensors.shape[-1:])
            cli_tokens = self.cli_norm(cli_tensors)   #Cli_var embedding

            concat_f = torch.cat((cross_pre_f,cli_tokens), dim=-1)
            final_tokens = self.final_self_att(concat_f.squeeze(0))
            pred = self.post_classifier(final_tokens)
        elif self.kargs.no_clic:
            pdw_f = self.axi_att(main_f, co_f1, co_f2)
            post_f = self.self_att(t1post_f)  # self-attention
            cross_pre_f = self.axi_cross_att(pdw_f, pre_aux_f)
            cross_post_f = self.axi_cross_att(post_f, post_aux_f)
            concat_f = torch.cat((cross_pre_f, cross_post_f), dim=-1)
            final_tokens = self.final_self_att(concat_f.squeeze(0))
            pred = self.post_classifier(final_tokens)
        elif self.kargs.post_img_only:
            post_f = self.self_att(t1post_f)
            # cross_post_f = self.axi_cross_att(post_f, post_aux_f)
            # pred = self.noxa_classifier(cross_post_f)
            pred = self.noxa_classifier(post_f)
        else:
            ################### Main Stream ####################
            pdw_f = self.axi_att(main_f, co_f1, co_f2)
            post_f = self.self_att(t1post_f)  # self-attention
            cross_pre_f = self.axi_cross_att(pdw_f, pre_aux_f)
            cross_post_f = self.axi_cross_att(post_f, post_aux_f)

            cli_tensors = torch.nan_to_num(cli, nan=0.0, posinf=1.0, neginf=-1.0)
            cli_tensors = torch.nn.functional.layer_norm(cli_tensors, cli_tensors.shape[-1:])
            cli_tokens = self.cli_norm(cli_tensors)   #Cli_var embedding

            concat_f = torch.cat((cross_pre_f, cross_post_f,cli_tokens), dim=-1)
            # concat_f = torch.cat((cross_pre_f, cross_post_f), dim=-1)
            final_tokens = self.final_self_att(concat_f.squeeze(0))
            pred = self.post_classifier(final_tokens)

        return pred

    def forward(self, input,cli):
        # input: [[bz, channel, slice, h, w], []..]
        if self.branch[0]:
            sag_pred = self.__post_branch__(input)
            final_pred = sag_pred
        if self.branch[1]:
            cor_pred = self.__pre_branch__(input,cli)
            final_pred = cor_pred
        if self.branch[2]:
            axi_pred = self.__final_branch__(input,cli)
            final_pred = axi_pred
        if sum(self.branch) == 1:
            return final_pred
