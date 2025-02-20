from libs import *
from mamba_ssm import Mamba

class mamba(nn.Module):
    def __init__(self, model_dim=34, d_state=16, d_conv=4, expand=2, dropout=0.15):
        super(mamba, self).__init__()
        self.model_dim = model_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.mamba = Mamba(d_model=self.model_dim, d_state=self.d_state, d_conv=self.d_conv, expand=self.expand)
        self.mlp = nn.Linear(self.model_dim, self.model_dim)
        self.ln = nn.LayerNorm(self.model_dim)
        self.ln1 = nn.LayerNorm(self.model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x):
        x = x.unsqueeze(1)
        residual = x
        
        x = self.ln(x)
        out = self.mamba(x)
        out = self.dropout1(out)
        z = out + residual
        out = self.ln1(z)

        out = self.mlp(out)
        out = self.dropout2(out) + z
        return out


class multi_feature_model(nn.Module):

    def __init__(self, base_model, input_features):
        super(multi_feature_model, self).__init__()

        self.base_model = base_model

        self.mlp_classification = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.Tanh(),
            nn.Linear(256, 2)
        )

        self.mlp_regression = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        self.mamba = mamba()
        self.fc = nn.Linear(34, 768)
        

        self.gate1 = nn.Sigmoid()
        self.gate2 = nn.Sigmoid()
        self.d_in1 = nn.ReLU(inplace=True)
        self.d_in2 = nn.ReLU(inplace=True)

    def forward(self, x, clinic, task_type='classification'):
        aggregated_features_list = []
        for patient_images in x:
            patient_features = self.base_model(patient_images)
            aggregated_feature = patient_features.mean(dim=0)
            aggregated_features_list.append(aggregated_feature)

        aggregated_features = torch.stack(aggregated_features_list, dim=0)
        
        ag = aggregated_features
        
        cli = self.mamba(clinic)
        cli = self.fc(cli)
        cli = cli.squeeze(1)
        cl = cli
        
        g_m1 = self.gate1(aggregated_features)
        g_m2 = self.gate2(cli)
        aggregated_features = aggregated_features * g_m1
        cli = cli * g_m2
        full_features = aggregated_features + cli
        full_features = full_features + cl + ag
        if task_type == 'classification':
            x = self.mlp_classification(full_features)

        elif task_type == 'regression':
            x = self.mlp_regression(full_features)

        return x




