import torch
import torch.nn as nn
import torch.nn.functional as F

class CoattentionModel(nn.Module):
    '''Co-attention Module
    '''
    def  __init__(self, all_channel=512):
        '''
        This model builds co-attention between two feature maps.

        Args:
            all_channel: channel of dispnet feature map

        eg: dispnet output shape (n, c, h, w), all_channel = c
        '''
        super(CoattentionModel, self).__init__()
        self.channel = all_channel

        self.linear = nn.Linear(all_channel, all_channel, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(all_channel, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.exemplar_att_classifier = self._build_classifier()
        self.query_att_classifier = self._build_classifier()
        
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()    
    

    def _build_classifier(self):
        return nn.Sequential(
            nn.Conv2d(self.channel * 2, self.channel, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(self.channel),
            nn.ReLU(inplace=True)
        )
		

    def forward(self, exemplar: torch.tensor, query: torch.tensor): #注意input2 可以是多帧图像
        input_size, fea_size = exemplar.size()[2:], query.size()[2:]

        exemplar_att, query_att = self._build_attention(exemplar, query)
        exemplar_att, query_att = self._after_attention(exemplar_att, exemplar, fea_size, self.exemplar_att_classifier),\
                                  self._after_attention(query_att, query, fea_size, self.query_att_classifier)
        exemplar_out, query_out = self._before_output(exemplar_att, input_size),\
                                  self._before_output(query_att, input_size)
        return exemplar_out, query_out  #shape: N, C, 1


    def _build_attention(self, exemplar: torch.tensor, query: torch.tensor):
        # reshape to (N, C, H * W)
        _, c, h, w = exemplar.size()
        view_params = [-1, c, h * w]
        exemplar_view, query_view = exemplar.view(view_params), query.view(view_params)
        exemplar_corr = self.linear(torch.transpose(exemplar_view, 1, 2).contiguous()) # transpose

        S = torch.bmm(exemplar_corr, query_view) # build relation
        S_c = F.softmax(S, dim=1)
        S_r = F.softmax(S.transpose(1, 2), dim=1)

        query_attention = torch.bmm(exemplar_view, S_c).contiguous()
        exemplar_attention = torch.bmm(query_view, S_r).contiguous()

        return exemplar_attention, query_attention


    def _after_attention(self, in_feature: torch.tensor, src_feature: torch.tensor,
            fea_size: tuple, classfier: nn.Sequential):
        in_feature = in_feature.view(-1, self.channel, fea_size[0], fea_size[1])
        mask = self.gate(in_feature)
        in_feature = in_feature * mask
        feature_cat = torch.cat([in_feature, src_feature], 1)
        return classfier(feature_cat)


    def _before_output(self, in_feature: torch.tensor, output_size: tuple):
        in_feature = F.interpolate(in_feature, output_size, mode='bilinear', align_corners=True)
        return torch.relu(in_feature)