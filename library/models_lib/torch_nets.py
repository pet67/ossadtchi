class envelope_detector(nn.Module):
    def __init__(self, in_channels, channels_per_channel):
        super(self.__class__,self).__init__()
        self.FILTERING_SIZE = 50
        self.ENVELOPE_SIZE = self.FILTERING_SIZE * 2
        self.CHANNELS_PER_CHANNEL = channels_per_channel
        self.OUTPUT_CHANNELS = self.CHANNELS_PER_CHANNEL * in_channels
        self.conv_filtering = nn.Conv1d(in_channels, self.OUTPUT_CHANNELS, bias=False, kernel_size=self.FILTERING_SIZE, groups=in_channels)
        self.conv_envelope = nn.Conv1d(self.OUTPUT_CHANNELS, self.OUTPUT_CHANNELS, kernel_size=self.ENVELOPE_SIZE, groups=self.OUTPUT_CHANNELS)
        
    def forward(self, x):
        x = self.conv_filtering(x)
        x = F.leaky_relu(x, negative_slope=-1)
        x = self.conv_envelope(x)
        return x


class simple_filtering(nn.Module):
    def __init__(self, in_channels):
        super(self.__class__,self).__init__()
        self.SIMPLE_FILTER_SIZE = 149
        self.simple_filter = nn.Conv1d(in_channels, in_channels, bias=False, kernel_size=self.SIMPLE_FILTER_SIZE, groups=in_channels)

    def forward(self, x):
        x = self.simple_filter(x)
        return x

    
class Dropout1D(nn.Module):
    def __init__(self, p):
        super(self.__class__,self).__init__()
        self.dropout_2d = nn.Dropout2d(p)

    def forward(self, x):
        x = x.unsqueeze(2)           # (N, C, 1, T)
        x = self.dropout_2d(x)       # (N, K, 1, T)
        x = x.squeeze(2)             # (N, C, T)
        return x


class simple_net(nn.Module):
    def __init__(self, in_channels, output_channels, lag_backward, lag_forward):
        super(self.__class__,self).__init__()
        self.ICA_CHANNELS = 0
        self.CHANNELS_PER_CHANNEL = 15
        self.OUTPUT_FILTERING = False
        self.EXTRA_FEATURES = False
        
        self.total_input_channels = self.ICA_CHANNELS + in_channels
        self.lag_backward = lag_backward
        self.lag_forward = lag_forward
        self.middle_point_share = lag_backward * 1.0 / (lag_backward + lag_forward)

        self.final_out_features = (self.CHANNELS_PER_CHANNEL + 2) * self.total_input_channels - 2 * in_channels * int(not self.EXTRA_FEATURES) 

        if self.ICA_CHANNELS > 0:
            self.ica = nn.Conv1d(in_channels, self.ICA_CHANNELS, 1)

        self.detector = envelope_detector(self.total_input_channels, self.CHANNELS_PER_CHANNEL)
        self.simple_filter = simple_filtering(self.total_input_channels)
        self.features_batchnorm = torch.nn.BatchNorm1d(self.final_out_features, affine=False)
        self.unmixed_batchnorm = torch.nn.BatchNorm1d(self.total_input_channels, affine=False)

        if self.OUTPUT_FILTERING:
            self.final_dropout = Dropout1D(p=0.8)
            self.wights_second = nn.Conv1d(self.final_out_features, output_channels, 1)
            self.predictions_filtering = nn.Conv1d(output_channels, output_channels, bias=True, kernel_size=52)
        else:
            self.final_dropout = torch.nn.Dropout(p=0.8)
            self.wights_second = nn.Linear(self.final_out_features, output_channels)



    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2) # for data_generator compareability
        if self.ICA_CHANNELS > 0:
            inputs_unmixed = self.ica(inputs)
            all_inputs = torch.cat((inputs, inputs_unmixed), 1)
            all_inputs = self.unmixed_batchnorm(all_inputs)
        else:
            all_inputs = inputs

        detected_envelopes = self.detector(all_inputs)
        if self.EXTRA_FEATURES:
            simple_filtered_signals = self.simple_filter(all_inputs)
            features = torch.cat((detected_envelopes, simple_filtered_signals, all_inputs[:, :, 148:]), 1)
        else:
            features = detected_envelopes
        
        if self.OUTPUT_FILTERING:
            features = self.features_batchnorm(features)
            features = self.final_dropout(features)
            output = self.wights_second(features)
            output = self.predictions_filtering(output)
            output = output.squeeze(2)
        else:
            middle_point = int((features.shape[-1]- 1) * self.middle_point_share) 
            features  = features[:, :, middle_point]
            features = features.contiguous()

            features = features.view(features.size(0), -1)
            features = self.features_batchnorm(features)
            features = self.final_dropout(features)
            output = self.wights_second(features)

        return output
