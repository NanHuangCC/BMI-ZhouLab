"""This file provide some classical model for training"""
import torch
import torch.nn as nn


# ICNET
class ICNET(nn.Module):
    def __init__(self, img_size=128, pool_size=2, out_channel=8):
        self.img_size = img_size
        self.pool_size = pool_size
        self.out_channel = out_channel
        self.new_size = int(((img_size/pool_size) ** 2) * out_channel)
        super(ICNET, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.out_channel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.pool_size, stride=2,padding=0)
        )

        self.LinNet = nn.Sequential(
            nn.Linear(self.new_size, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.ConvNet(x)
        x = x.view(x.size(0), self.new_size)
        out = self.LinNet(x)
        return out


# RNN_net
class Rnn(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, outsize=1):
        self.input_size = input_size    # input size
        self.hidden_size = hidden_size
        self.num_layers = num_layers    # layer size
        self.outsize = outsize
        super(Rnn, self).__init__()
        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)
        self.LinNet = nn.Linear(self.hidden_size, self.outsize)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)

        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.LinNet(r_out[:, time, :]))
        out = torch.stack(outs, dim=1)
        return out, h_state


# SeqNet
class SeqNET(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, outsize=1):
        self.input_size = input_size    # input size
        self.hidden_size = hidden_size
        self.outsize = outsize
        super(SeqNET, self).__init__()
        self.LinNet = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.outsize),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        out = self.LinNet(x)
        return out



# CNN_LSTM
class CNN_LSTM(nn.Module):
    def __init__(self, conv_input, conv_out, kernel_size, stride, hidden_size, num_layers, output_size):
        super(CNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv = nn.Conv1d(conv_input, conv_input, kernel_size=kernel_size, stride=stride)
        self.lstm = nn.LSTM(conv_out, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out






class CNN_LSTM2D(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size, stride, conv_out, hidden_size, num_layers, output_size):
        super(CNN_LSTM2D, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channel, kernel_size=kernel_size, stride=stride, bias=False)
        self.lstm = nn.LSTM(conv_out, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        print(x.shape)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out


class NeuActRegNET(nn.Module):
    def __init__(self, in_channels2DS=1, out_channel2DS = 1, kernel_size2DS = (3,3), stride2DS = (3,3),
                       in_channels1DS=5, out_channels1DS = 1, kernel_size1DS = 1, stride1DS = 1,
                       in_channels2DM=1, out_channel2DM = 1, kernel_size2DM = (3,3), stride2DM = (3,3),
                       in_channels1DM=5, out_channels1DM = 1, kernel_size1DM = 1, stride1DM = 1,
                       DeepFeatures=1, hidden_size=512, outsize=1):
        super(NeuActRegNET, self).__init__()
        self.conv2DS = nn.Conv2d(in_channels=in_channels2DS, out_channels=out_channel2DS,
                                kernel_size=kernel_size2DS, stride=stride2DS, bias=False)
        self.conv1DS = nn.Conv1d(in_channels=in_channels1DS, out_channels=out_channels1DS,
                                kernel_size=kernel_size1DS, stride=stride1DS)
        self.conv2DM = nn.Conv2d(in_channels=in_channels2DM, out_channels=out_channel2DM,
                                 kernel_size=kernel_size2DM, stride=stride2DM, bias=False)
        self.conv1DM = nn.Conv1d(in_channels=in_channels1DM, out_channels=out_channels1DM,
                                 kernel_size=kernel_size1DM, stride=stride1DM)
        self.LinNet = nn.Sequential(
            nn.Linear(DeepFeatures,hidden_size),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, outsize)
        )

    def forward(self, Ust, Umt, Xt):
        # get sensory feature
        SFeature = self.conv2DS(Ust)
        SFeature = SFeature.squeeze(dim=1)
        SFeature = self.conv1DS(SFeature)
        SFeature = SFeature.squeeze(dim=1)
        # get movement feature
        MFeature = self.conv2DM(Umt)
        MFeature = MFeature.squeeze(dim=1)
        MFeature = self.conv1DM(MFeature)
        MFeature = MFeature.squeeze(dim=1)
        # Linear process
        BehFeature = torch.cat((SFeature, MFeature), 1)
        DeepFeatures = torch.cat((BehFeature, Xt), 1)
        out = self.LinNet(DeepFeatures)
        return out, DeepFeatures