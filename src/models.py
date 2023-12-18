import torch
import torch.nn.functional as F
from torch import nn

from src.model.resnet import resnet50, resnet101

from src.model.pspnet import PSPNet,get_resnet,get_pspnet

from .optimizer import get_optimizer, get_scheduler
from .classifier import SemiClassifier, StandardClassifier
from src.seghead.upernet import UPerNet, FF
from src.model.listencoder import ListEncoder
from src.mmpretrain.mmpretrain.apis.model import get_model
from src.dino_vision_transformer import vit_base
import pdb

from .util import to_one_hot, compute_wce

class SegmentModule(nn.Module):
    def __init__(self,backbone,segmenthead,classifier,backbone_arch,segment_arch,train_seg):
        super(SegmentModule,self).__init__()
        self.backbone = backbone
        self.segmenthead = segmenthead
        self.classifier = classifier
        self.backbone_arch = backbone_arch
        self.segment_arch = segment_arch
        self.train_seg = train_seg


    def forward(self,x):
        x = self.extract_feat(x)
        x = self.classifier(x, gt_dim = 473)
        return x

    def extract_feat(self,x):
        x = self.backbone.extract_feat(x)
        if self.backbone_arch == "pspnet":
            pass
        elif self.segment_arch == "UPerNet":
            if self.backbone_arch[-3:] == "net":
                pass
            else:
                #pick [2,5,8,11] from 12 layers
                x = x[2:12:3]
        elif self.segment_arch == "None":
            x = x[-1]
        if not self.train_seg:
            x = self.segmenthead(x)
        return x


    def init_weight(self, base_weight, base_bias,num_novel_classes, channel_num):
        novel_weight = torch.zeros((num_novel_classes, channel_num, 1, 1), device=torch.device('cuda'))
        novel_bias = torch.zeros((num_novel_classes,), device=torch.device('cuda'))
        assert base_weight.shape[1] == novel_weight.shape[1]
        init_weights = torch.cat((base_weight, novel_weight), dim=0)
        init_bias = torch.cat((base_bias, novel_bias), dim=0)
        self.classifier.conv1.weight = torch.nn.Parameter(init_weights)
        self.classifier.conv1.bias = torch.nn.Parameter(init_bias)

        self.classifier.conv1.weight[:16,:,:].requires_grad_ = False
        self.classifier.conv1.bias[:16].requires_grad_ = False
        # pdb.set_trace()

    def _valid_mean(self, t, valid_pixels, dim):
        s = (valid_pixels * t).sum(dim=dim)
        return s / (valid_pixels.sum(dim=dim) + 1e-10)

    def train_classifier(self, s_feat, gt_s, num_classes_tr, num_classes_val, iterations=100, lr_semi=0.01):
        gt_dim = gt_s.shape[-2:]
        # criterion = nn.CrossEntropyLoss(ignore_index=255)
        # optimizer = optim.SGD(model.parameters(), lr=lr_semi)
        # params = [model.novel_weight, model.novel_bias]
        # pdb.set_trace()
        params = [p for p in self.classifier.parameters()]
        # params = [self.classifier.conv1.weight,self.classifier.conv1.bias]
        for m in params:
            m.requires_grad_()
        optimizer = torch.optim.SGD(params, lr=lr_semi)

        one_hot_gt_s = to_one_hot(gt_s, num_classes_val + num_classes_tr)
        valid_pixels_s = (gt_s != 255).float()
        for iteration in range(iterations):
            # pdb.set_trace()
            # print("parameters update: ", params[0].sum())
            
            # print(model.novel_weight.shape,model.novel_weight.sum())
            # print(model.conv1.weight.shape,model.conv1.weight.sum())
            s_logit = self.classifier.forward(s_feat, gt_dim)  # [N, num_classes,473,473])
            s_prob = F.softmax(s_logit, dim = 2)
            s_prob = s_prob.clone()
            s_prob = s_prob.unsqueeze(dim=1)
            s_prob[:, :, 0, ...] += s_prob[:, :, 1:num_classes_tr, ...].sum(dim=2)
            s_prob[:, :, 1:num_classes_tr, ...] = 0.
            loss = - (one_hot_gt_s * torch.log(s_prob + 1e-10))
            loss = (loss * compute_wce(one_hot_gt_s, num_classes_val)).sum(2)
            loss = self._valid_mean(loss, valid_pixels_s, (1, 2, 3))  # [batch_size_val,]
            loss = loss.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (iteration+1) % 50 == 0:
                print(f'train_semi_iter {iteration+1}, loss = {loss.item()}' )
        print('finish ===============')

        return loss
    def train_decoder(self, s_feat, gt_s, num_classes_tr, num_classes_val, iterations=100, lr_semi=0.01):
        gt_dim = gt_s.shape[-2:]
        # criterion = nn.CrossEntropyLoss(ignore_index=255)
        # optimizer = optim.SGD(model.parameters(), lr=lr_semi)
        # params = [model.novel_weight, model.novel_bias]
        # pdb.set_trace()
        params = [p for p in self.classifier.parameters()]+[p for p in self.segmenthead.parameters()]
        # params = [self.classifier.conv1.weight,self.classifier.conv1.bias]
        for m in params:
            m.requires_grad_()
        optimizer = torch.optim.SGD(params, lr=lr_semi)

        one_hot_gt_s = to_one_hot(gt_s, num_classes_val + num_classes_tr)
        valid_pixels_s = (gt_s != 255).float()
        for iteration in range(iterations):
            # pdb.set_trace()
            # print("parameters update: ", params[0].sum())
            
            # print(model.novel_weight.shape,model.novel_weight.sum())
            # print(model.conv1.weight.shape,model.conv1.weight.sum())
            s_logit = self.classifier.forward(self.segmenthead(s_feat), gt_dim)  # [N, num_classes,473,473])
            s_prob = F.softmax(s_logit, dim = 2)
            s_prob = s_prob.clone()
            s_prob = s_prob.unsqueeze(dim=1)
            s_prob[:, :, 0, ...] += s_prob[:, :, 1:num_classes_tr, ...].sum(dim=2)
            s_prob[:, :, 1:num_classes_tr, ...] = 0.
            loss = - (one_hot_gt_s * torch.log(s_prob + 1e-10))
            loss = (loss * compute_wce(one_hot_gt_s, num_classes_val)).sum(2)
            loss = self._valid_mean(loss, valid_pixels_s, (1, 2, 3))  # [batch_size_val,]
            loss = loss.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (iteration+1) % 50 == 0:
                print(f'train_semi_iter {iteration+1}, loss = {loss.item()}' )
        print('============finish ===============')

        return loss

    def train_val(self, s_feat, gt_s, num_classes_tr, num_classes_val, iterations=100, lr_semi=0.01):
        if self.train_seg:
            return self.train_decoder(self, s_feat, gt_s, num_classes_tr, num_classes_val, iterations, lr_semi)
        else:
            return self.train_classifier(self, s_feat, gt_s, num_classes_tr, num_classes_val, iterations, lr_semi)


def get_model_optimizer(args):

    # ====== Back_Bone ======
    backbone_name = args.arch
    inplanes = (768,768,768,768)
    indim = 768
    if args.arch == "resnet":
        backbone = get_resnet(args)
        backbone = ListEncoder(backbone)
        inplanes = (256,512,1024,2048)
        indim = 2048
        if args.layers <=34:
            indim = 512
            inplanes = (64,128,256,512)
    elif args.arch == "pspnet":
        backbone = get_pspnet(args)
        # backbone = ListEncoder(backbone)
        # inplanes = (256,512,1024,2048)
        indim = 512

# ViT based backbone's output: [[batchsize, 768, 30, 30] *12 ]
    elif backbone_name == "Dino":
        backbone = vit_base(16)
        dino_vitb16_param = torch.load('src/dino_vitbase16_pretrain.pth')
        backbone.load_state_dict(dino_vitb16_param)
    elif backbone_name == 'DeiT':
        backbone = get_model('deit-base_16xb64_in1k', pretrained=True) #torch.Size([1, 768, 28, 28])
    elif backbone_name == 'ViT':
        backbone = get_model('vit-base-p16_32xb128-mae_in1k', pretrained=True) #output size torch.Size([1, 768, 28, 28])
    elif backbone_name == 'Dino2':
        backbone = get_model('vit-base-p14_dinov2-pre_3rdparty', pretrained=True) #torch.Size([1, 768, 32, 32])
    backbone = backbone.cuda()
    # ====== Segmenthead ======
    feat_dim = 256
    if args.seghead == "UPerNet":
        segmenthead = UPerNet(fc_dim = indim, fpn_inplanes = inplanes).cuda()
        feat_dim = 256
    elif args.seghead == "None":
        segmenthead = FF().cuda()
        feat_dim = indim
    

    # ====== Classifier ======
    if args.episodic_val:
        classifier = SemiClassifier(feat_dim = feat_dim,classes = args.num_classes_tr).cuda()
    else:
        classifier = StandardClassifier(feat_dim = feat_dim, classes = args.num_classes_tr).cuda()

    # ====== Model and classifier ======

    model = SegmentModule(backbone,segmenthead,classifier,backbone_arch = args.arch,segment_arch = args.seghead, train_seg = args.train_seg)
    if args.arch[-3:] == "net":
        if args.layers<=34 or args.layers>50:
            modules_ori = [model.backbone.relu,model.backbone.conv1, model.backbone.bn1, model.backbone.layer1, model.backbone.layer2, model.backbone.layer3, model.backbone.layer4]
        else:
            modules_ori = [model.backbone.layer0, model.backbone.layer1, model.backbone.layer2, model.backbone.layer3, model.backbone.layer4]
        if args.seghead == "UPerNet":
            modules_new = [model.segmenthead.ppm, model.segmenthead.bottleneck, model.segmenthead.fpn_in,
                    model.segmenthead.fpn_out, model.segmenthead.conv_fusion, model.segmenthead.fn,model.classifier.conv1]
        elif args.seghead == "None":
             modules_new = [model.classifier.conv1]
        params_list = []
        for module in modules_ori:
            params_list.append(dict(params=module.parameters(), lr=args.lr))
        for module in modules_new:
            params_list.append(dict(params=module.parameters(), lr=args.lr * args.scale_lr))
        optimizer = get_optimizer(args, params_list)
    else:
        if args.seghead == "UPerNet":
            modules_new = [model.segmenthead.ppm, model.segmenthead.bottleneck, model.segmenthead.fpn_in,
                    model.segmenthead.fpn_out, model.segmenthead.conv_fusion, model.segmenthead.fn,model.classifier.conv1]
        elif args.seghead == "None":
             modules_new = [model.classifier.conv1]
        params_list = []
        for module in modules_new:
            params_list.append(dict(params=module.parameters(), lr=args.lr * args.scale_lr))
        optimizer = get_optimizer(args, params_list)
    
    return model,optimizer
    
