from model import common
import torch.nn as nn
import torch.nn.functional as F
import torch
url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

def make_model(args, parent=False):
    return EDSR(args)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class AuxSR(nn.Module):
    def __init__(self,args,inplanes,conv,  scale, n_feats, feature_dim=64):
        super(AuxSR, self).__init__()

        assert inplanes in [16, 32, 64]

        self.feature_dim = feature_dim
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        act = nn.ReLU(True)     
        self.criterion = nn.L1Loss()
        self.upsample=common.Upsampler(conv, scale, n_feats, act=False)
        self.endconv= conv(n_feats, args.n_colors, 3)


        # self.head = nn.Sequential(
        #             nn.Conv2d(inplanes,feature_dim,1,1,0,bias=False),
        #             nn.BatchNorm2d(feature_dim),
        #             nn.ReLU(),
        #             nn.Conv2d(feature_dim,feature_dim,1,1,0,bias=False),

        # )
        self.head=nn.Sequential(
                    conv(feature_dim, feature_dim, 3),
                    conv(feature_dim, feature_dim, 3),
                    conv(feature_dim, feature_dim, 3)
                    # nn.Conv2d(feature_dim,feature_dim,1,1,0,bias=False),
                    # nn.BatchNorm2d(feature_dim),
                    # nn.ReLU(),
                    # nn.Conv2d(feature_dim, feature_dim, 1, 1, 0,bias=False),
                    # nn.BatchNorm2d(feature_dim),
                    # nn.ReLU(),
            #   common.ResBlock(
            #     conv, n_feats, 3, act=act, res_scale=args.res_scale
            # ),
            #   common.ResBlock(
            #     conv, n_feats, 3, act=act, res_scale=args.res_scale
            # )
        )


    def forward(self, x, target,res):
 
        x = self.head(x)
        res=res+x
        x=self.upsample(x)
        x=self.endconv(x)
        features=self.add_mean(x)
      
        loss = self.criterion(features, target)
        return loss

class EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR, self).__init__()
        print("args=",args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        print("尺寸是",scale)
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        self.aux_classifiers = {} 
        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        # m_body = [
        #     common.ResBlock(
        #         conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
        #     ) for _ in range(n_resblocks)
        # ]
        m_body=[]
        for i in range(n_resblocks):
            resblock=common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            )
            m_body.append(resblock)
            aux_classifier_key = f"aux_{i}" 
            self.aux_classifiers[aux_classifier_key] = AuxSR(args,64,conv,  scale, n_feats).to(device)
            

        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x,target):
        # x = self.sub_mean(x)
        x = self.head(x)



        res = x.clone().detach()
        for i,block in enumerate(self.body):
            x=block(x)
            if self.training:

                if isinstance(block, common.ResBlock):
                # # 获取对应的AuxNet
                    aux_net = self.aux_classifiers[f'aux_{i}']
                # 通过AuxNet处理ResBlock的输出
                    loss = aux_net(x,target,res)
                    loss.backward()
                    x=x.detach()
                
        # res = self.body(x)
        # print("大小是",x.shape)
        res = res + x

        x = self.tail(res)
        # x = self.add_mean(x)
        # print("大小是",x.shape)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

