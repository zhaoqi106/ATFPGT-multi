from torch import nn
from mtl_model import Fox


class Multitask(nn.Module):
    def __init__(self, task='reg', tasks=1, attn_head=4, output_dim=128, d_k=64, d_v=64, attn_layers=4, D=16,
                 dropout=0.1, disw=1.5, device='cuda:0', fp_type=None, num_tasks=4, input_dim=256, hidden_dim=256):
        super(Multitask, self).__init__()
        self.feature = Fox(task, tasks, attn_head, output_dim, d_k, d_k, attn_layers, D, dropout, 1.5, device, fp_type)

        self.fc = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(output_dim, num_tasks),
            nn.ReLU(),

        )
        self.shared_feature_extractor = SharedFeatureExtractor(input_dim, hidden_dim)
        self.task1_output = TaskSpecificOutput(hidden_dim, 1)
        self.task2_output = TaskSpecificOutput(hidden_dim, 1)
        self.task3_output = TaskSpecificOutput(hidden_dim, 1)
        self.task4_output = TaskSpecificOutput(hidden_dim, 1)
        # self.sigmoid = Sigmoid()

    def forward(self, data_task1, data_task2, data_task3, data_task4):
        feature_1 = self.feature(data_task1)  # 共享特征提取层 ，cat（graph，fp）
        feature_2 = self.feature(data_task2)
        feature_3 = self.feature(data_task3)
        feature_4 = self.feature(data_task4)

        features_task1 = self.shared_feature_extractor(feature_1)  # 共享全连接层
        features_task2 = self.shared_feature_extractor(feature_2)
        features_task3 = self.shared_feature_extractor(feature_3)
        features_task4 = self.shared_feature_extractor(feature_4)

        out1 = self.task1_output(features_task1).sigmoid()  # 为每个任务单独的创建输出
        out2 = self.task2_output(features_task2).sigmoid()
        out3 = self.task3_output(features_task3).sigmoid()
        out4 = self.task4_output(features_task4).sigmoid()

        return out1, out2, out3, out4


class SharedFeatureExtractor(nn.Module):  # 两层共享全链接
    def __init__(self, input_dim, hidden_dim):
        super(SharedFeatureExtractor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        features = self.fc(x)
        return features


class TaskSpecificOutput(nn.Module):  # 每个任务的单独输出
    def __init__(self, input_dim, output_dim):
        super(TaskSpecificOutput, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, features):
        output = self.fc(features)
        return output





