import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from sentence_transformers import SentenceTransformer
from ray.rllib.utils.annotations import override


class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0),  
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=2, padding=0), 
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0), 
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0),
            nn.ELU(), 
            nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0),
            nn.ELU(),
            nn.Conv2d(256, 512, kernel_size=2, stride=2, padding=0),
            nn.ELU(),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.cnn(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model=6):
        super().__init__()
        self.d_model= d_model
        if self.d_model % 6 != 0:
            raise ValueError("d_models must be divedable on 6!")

        pe = np.zeros((9, 11, 11, d_model))

        for pos_x in range(9):
            pe[pos_x,:,:,0:d_model//3:2] = np.sin(0.33 * pos_x / 10_000 ** (6*np.arange(d_model//6)/d_model))
            pe[pos_x,:,:,1:d_model//3:2] = np.cos(0.33 * pos_x / 10_000 ** (6*np.arange(d_model//6)/d_model))

        for pos_y in range(11):
            pe[:,pos_y,:,d_model//3:2*d_model//3:2] = np.sin(0.33 * pos_y / 10_000 ** (6*np.arange(d_model//6)/d_model))
            pe[:,pos_y,:,1+d_model//3:2*d_model//3:2] = np.cos(0.33 * pos_y / 10_000 ** (6*np.arange(d_model//6)/d_model))

        for pos_z in range(11):
            pe[:,:,pos_z,2*d_model//3::2] = np.sin(0.33 * pos_z / 10_000 ** (6*np.arange(d_model//6)/d_model))
            pe[:,:,pos_z,1+2*d_model//3::2] = np.cos(0.33 * pos_z / 10_000 ** (6*np.arange(d_model//6)/d_model))
            
        pe = pe.reshape(9 * 11 * 11, d_model)
        self.pe = torch.tensor(pe).float()
        
    def forward(self):
        #x = x * math.sqrt(d_model // 3) # is it needed?
        #x = x + self.pe
        return self.pe


class FusionNet(nn.Module):
    def __init__(self, d_model=6, num_heads=1):
        super().__init__()
        self.d_model = d_model
        self.pe = nn.Parameter(PositionalEncoder(d_model)())
        
        self.img_preproc = nn.Sequential(
            nn.Linear(512, 60),
            nn.ELU(),
        )
        
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        self.conv_net = nn.Sequential(
            nn.Conv3d(6, 8, kernel_size=3, padding=1),            # perceptive field = 3
            nn.ELU(),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),           # perceptive field = 5
            nn.ELU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),          # perceptive field = 7
            nn.ELU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),          # perceptive field = 9
            nn.ELU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),         # perceptive field = 11
            nn.ELU(),
            nn.MaxPool3d(kernel_size=(9, 11, 11))
        )
        
        self.img_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )
        
    def forward(self, target_features, img_features):
        batch_size = target_features.shape[0]
        
        img_features2 = self.img_preproc(img_features)
        target_features = target_features.permute(0, 2, 3, 4, 1).reshape(batch_size, 9 * 11 * 11, self.d_model)
        img_features2 = img_features2.reshape(batch_size, -1, self.d_model)
        target_features += self.cross_attn(key=img_features2, value=img_features2, query=target_features)[0]
        k = q = target_features + self.pe
        target_features += self.self_attn(key=k, value=target_features, query=q)[0]
        
        target_features = target_features.reshape(batch_size, 9, 11, 11, self.d_model).permute(0, 4, 1, 2, 3)
        target_features = self.conv_net(target_features).reshape(batch_size, -1)
        
        img_features = self.img_mlp(img_features)
        
        features = torch.cat([target_features, img_features], dim=1)
        features = self.mlp(features)
        
        return features


class MyModelClass(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        visual_features_dim = 512
        target_features_dim = 9 * 11 * 11 
        self.visual_encoder = VisualEncoder()
        #self.visual_encoder.load_state_dict(
        #    torch.load("/IGLU-Minecraft/models/AngelaCNN/encoder_weigths.pth", map_location=torch.device('cpu'))
        #)
        self.target_encoder = nn.Sequential(
            nn.Conv3d(7, 6, kernel_size=1, stride=1, padding=0),
            nn.ELU(),
        )
        policy_hidden_dim = 256 
        self.policy_network = FusionNet()
        
        self.action_head = nn.Linear(policy_hidden_dim, action_space.n)
        self.value_head = nn.Linear(policy_hidden_dim, 1)
        self.last_value = None
        
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.visual_encoder.cuda()
            self.target_encoder.cuda()
            self.policy_network.cuda()
            self.action_head.cuda()
            self.value_head.cuda()
        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        pov = obs['pov'].permute(0, 3, 1, 2).float() / 255.0
        target = one_hot(obs['target_grid'].long(), num_classes=7).permute(0, 4, 1, 2, 3).float()
        if self.use_cuda:
            pov.cuda()
            target.cuda()
            
        with torch.no_grad():
            visual_features = self.visual_encoder(pov)
            
        target_features = self.target_encoder(target)
        
        features = self.policy_network(target_features, visual_features)
        
        action = self.action_head(features)
        self.last_value = self.value_head(features).squeeze(1)
        return action, state
    
    @override(TorchModelV2)
    def value_function(self):
        assert self.last_value is not None, "must call forward() first"
        return self.last_value


class TargetDecoder(nn.Module):
    def __init__(self, features_dim=768):
        super(TargetDecoder, self).__init__()

        self.linear = nn.Sequential(nn.Linear(features_dim, 15680))

        self.cnn = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.ConvTranspose3d(32, 7, kernel_size=3),
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], 64, 5, 7, 7)
        x = self.cnn(x)
        return x


class TargetPredictor(nn.Module):
    def __init__(self):
        super(TargetPredictor, self).__init__()
        self.bert = SentenceTransformer('all-distilroberta-v1', 
                                        cache_folder='robert_cache')
        self.target_decoder = TargetDecoder()
        self.target_decoder.load_state_dict(
            torch.load("target_decoder.pth", map_location=torch.device('cpu'))
        )

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.bert.cuda()
            self.target_decoder.cuda()

    def forward(self, chat):
        text_features = torch.tensor(self.bert.encode(chat)[None,])
        #print(text_features, text_features.shape)
        prediction = self.target_decoder(text_features)
        target = torch.argmax(nn.Softmax(dim=1)(prediction).permute(0, 2, 3, 4, 1), dim=4)[0]
        return target