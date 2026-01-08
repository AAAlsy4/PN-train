import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from data_classification_loader import create_classification_dataset


# 分类模型定义
class TimeSeriesClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.2):
        """
        时间序列分类模型
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            num_classes: 类别数量 (2: 节假日/常规日, 3: 节假日/工作日/周末)
            dropout: dropout率
        """
        super(TimeSeriesClassifier, self).__init__()

        # 输入层归一化
        self.input_norm = nn.LayerNorm(input_dim)

        # 时间特征提取
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)

        # 输入归一化
        x = self.input_norm(x)

        # LSTM提取时间特征
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_dim*2)

        # 注意力机制
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # 加权求和
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_dim*2)

        # 分类
        logits = self.classifier(context)

        return logits, attention_weights


# 数据集类
class ClassificationDataset(Dataset):
    def __init__(self, data, labels, seq_length=12):
        """
        分类数据集
        Args:
            data: 时间序列数据 (num_samples, seq_length, num_features)
            labels: 标签 (num_samples,)
            seq_length: 序列长度
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 数据增强方法
class DataAugmentation:
    @staticmethod
    def add_noise(data, noise_level=0.01):
        """添加高斯噪声"""
        noise = torch.randn_like(data) * noise_level
        return data + noise

    @staticmethod
    def time_warp(data, warp_factor=0.1):
        """时间扭曲"""
        batch_size, seq_len, features = data.shape
        warp_points = int(seq_len * warp_factor)

        if warp_points < 2:
            return data

        # 选择扭曲点
        warp_indices = sorted(np.random.choice(range(1, seq_len - 1), warp_points, replace=False))
        warped_data = data.clone()

        for i in range(len(warp_indices) - 1):
            start = warp_indices[i]
            end = warp_indices[i + 1]

            # 随机拉伸或压缩
            scale = np.random.uniform(0.9, 1.1)
            new_length = int((end - start) * scale)

            if new_length > 0:
                # 插值
                original_segment = data[:, start:end, :]
                warped_segment = F.interpolate(
                    original_segment.transpose(1, 2),
                    size=new_length,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

                # 调整大小以匹配原始长度
                if new_length < (end - start):
                    padding = torch.zeros(batch_size, (end - start) - new_length, features)
                    warped_segment = torch.cat([warped_segment, padding], dim=1)
                elif new_length > (end - start):
                    warped_segment = warped_segment[:, :(end - start), :]

                warped_data[:, start:end, :] = warped_segment

        return warped_data

    @staticmethod
    def scaling(data, scale_range=(0.9, 1.1)):
        """缩放"""
        scale = np.random.uniform(*scale_range)
        return data * scale

    @staticmethod
    def apply_augmentation(batch_data, methods=['noise', 'scale'], prob=0.5):
        """应用数据增强"""
        augmented_data = batch_data.clone()

        for i in range(len(batch_data)):
            if np.random.random() < prob:
                if 'noise' in methods:
                    augmented_data[i] = DataAugmentation.add_noise(batch_data[i:i + 1])[0]
                if 'scale' in methods:
                    augmented_data[i] = DataAugmentation.scaling(batch_data[i:i + 1])[0]

        return augmented_data


# 分类模型训练器
class ClassifierTrainer:
    def __init__(self, model, device='cuda', num_classes=2):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5,
            verbose=True
        )

        # 数据增强
        self.augmentation = DataAugmentation()

    def train_epoch(self, args, train_loader, use_augmentation=True):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(self.device), labels.to(self.device)

            # 数据增强
            if use_augmentation and np.random.random() < 0.5:
                data = self.augmentation.apply_augmentation(data)

            # 前向传播
            self.optimizer.zero_grad()
            logits, _ = self.model(data)
            loss = self.criterion(logits, labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # mae = torch.mean(torch.abs(predicted.float() - labels.float())).item()
            # rmse = torch.sqrt(torch.mean((predicted.float() - labels.float()) ** 2)).item()
            # wmape = torch.mean(torch.abs((predicted.float() - labels.float()) / (labels.float() + 1.0))).item()

            import sklearn.metrics
            mae = sklearn.metrics.mean_absolute_error(labels.cpu().numpy(), predicted.cpu().numpy())
            rmse = sklearn.metrics.mean_squared_error(labels.cpu().numpy(), predicted.cpu().numpy(), squared=False)
            wmape = sklearn.metrics.mean_absolute_error(labels.cpu().numpy(), predicted.cpu().numpy()) / (
                    labels.cpu().numpy())

            # 计算MAE
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')
                print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}, WMAPE: {wmape:.4f}')

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy, mae, rmse, wmape

    def validate(self, args, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)

                logits, _ = self.model(data)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)

    def train(self, args, train_loader, val_loader, epochs=50, patience=20):
        best_val_loss = float('inf')
        patience_counter = 0
        trlosses = []
        traccuracys = []
        vallosses = []
        valaccuracys = []
        mae_list = []
        rmse_list = []
        wmape_list = []

        for epoch in range(epochs):
            # 训练
            train_loss, train_acc, mae, rmse, wmape = self.train_epoch(args, train_loader)
            trlosses.append(train_loss)
            traccuracys.append(train_acc)
            mae_list.append(mae)
            rmse_list.append(rmse)
            wmape_list.append(wmape)

            # 验证
            val_loss, val_acc, _, _ = self.validate(args, val_loader)
            vallosses.append(val_loss)
            valaccuracys.append(val_acc)

            # 学习率调度
            self.scheduler.step(val_loss)

            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  MAE: {mae:.4f}, RMSE: {rmse:.4f}, WMAPE: {wmape:.4f}')

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(),
                           f'/home/liushuyu/PN-Train/results/{args.data}/classifier/best_classifier.pth')
                print('  Saved best model')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        # losses和accuracys保存为txt文件
        with open(f'/home/liushuyu/PN-Train/results/{args.data}/classifier/classifiertrain_losses.txt', 'w') as f:
            f.write("train Losses:\n")
            for l in trlosses:
                f.write(f"{l:.4f} ")
            f.write("\n")
            f.write("Val Losses:\n")
            for l in vallosses:
                f.write(f"{l:.4f} ")
            f.write("\n")
        with open(f'/home/liushuyu/PN-Train/results/{args.data}/classifier/classifiertrain_accuracys.txt', 'w') as f:
            f.write("train Accuracys:\n")
            for a in traccuracys:
                f.write(f"{a:.4f} ")
            f.write("\n")
            f.write("Val Accuracys:\n")
            for a in valaccuracys:
                f.write(f"{a:.4f} ")
            f.write("\n")
        with open(f'/home/liushuyu/PN-Train/results/{args.data}/classifier/classifiertrain_ex.txt', 'w') as f:
            f.write("train MAE:\n")
            for a in mae_list:
                f.write(f"{a:.4f} ")
            f.write("\n")
            f.write("Val RMSE:\n")
            for a in rmse_list:
                f.write(f"{a:.4f} ")
            f.write("\n")
            f.write("train WMAPE:\n")
            for a in wmape_list:
                f.write(f"{a:.4f} ")
            f.write("\n")

        # 加载最佳模型
        self.model.load_state_dict(
            torch.load(f'/home/liushuyu/PN-Train/results/{args.data}/classifier/best_classifier.pth'))

        return self.model

    def predict(self, data_loader):
        self.model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                logits, _ = self.model(data)
                probs = F.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)


# 特征工程工具
class FeatureExtractor:
    @staticmethod
    def extract_statistical_features(data):
        """提取统计特征"""
        # data shape: (batch_size, seq_len, features)

        # 基本统计量
        mean_features = data.mean(dim=1)  # (batch_size, features)
        std_features = data.std(dim=1)  # (batch_size, features)
        max_features = data.max(dim=1)[0]  # (batch_size, features)
        min_features = data.min(dim=1)[0]  # (batch_size, features)

        # 变化特征
        diff_features = torch.diff(data, dim=1).mean(dim=1)  # 差分均值

        # 组合特征
        statistical_features = torch.cat([
            mean_features, std_features, max_features, min_features, diff_features
        ], dim=1)

        return statistical_features

    @staticmethod
    def extract_temporal_features(data, time_features):
        """提取时间特征"""
        # time_features shape: (batch_size, seq_len, time_dim)

        # 合并时间特征和数据
        temporal_features = torch.cat([data, time_features], dim=-1)

        return temporal_features


# 集成分类器
class EnsembleClassifier:
    def __init__(self, models, weights=None):
        """
        集成多个分类模型
        Args:
            models: 模型列表
            weights: 权重列表
        """
        self.models = models
        self.weights = weights if weights else [1 / len(models)] * len(models)

    def predict(self, data_loader, device='cuda'):
        all_probs = []

        for model, weight in zip(self.models, self.weights):
            model.eval()
            model = model.to(device)

            probs_list = []
            with torch.no_grad():
                for data, _ in data_loader:
                    data = data.to(device)
                    logits, _ = model(data)
                    probs = F.softmax(logits, dim=1)
                    probs_list.append(probs.cpu().numpy())

            model_probs = np.concatenate(probs_list, axis=0)
            weighted_probs = model_probs * weight
            all_probs.append(weighted_probs)

        # 集成预测
        ensemble_probs = np.sum(all_probs, axis=0)
        predictions = np.argmax(ensemble_probs, axis=1)

        return predictions, ensemble_probs


# 主训练函数
def train_classification_model(data_config, num_classes=2):
    """
    训练分类模型的主函数
    Args:
        data_config: 数据配置
        num_classes: 类别数量
    """
    # 1. 加载数据
    print("Loading data...")

    class Args:
        def __init__(self):
            self.data = 'pedestrian'
            self.root_path = '/home/liushuyu/PN-Train/datasets/'
            self.data_path = 'metro-traffic/traffic.npz'
            #   self.data_path = 'pedestrian/pedestrian.npz'
            self.seq_len = 12
            self.train_ratio = 0.7
            self.batch_size = 32
            self.num_workers = 0
            self.holiday_threshold = 12  # 节假日时间步阈值
            self.epochs = 50  # 训练轮数
            self.patience = 10  # 早停耐心值

    args = Args()

    # 测试二分类数据加载
    print("Testing binary classification dataset...")
    binary_dataset = create_classification_dataset(args, classification_type='binary')

    # 查看一个批次的数据
    for batch_x, batch_y in binary_dataset.train_loader:
        if len(batch_x) > 0:
            print(f"Batch x shape: {batch_x.shape}")  # (batch_size, seq_len, features)
            print(f"Batch y shape: {batch_y.shape}")  # (batch_size,)
            print(f"Batch y distribution: {torch.bincount(batch_y)}")
            break

    # 2. 初始化模型
    print("Initializing model...")
    input_dim = batch_x.shape[2]  # 根据实际特征维度调整
    model = TimeSeriesClassifier(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.2
    )

    # 3. 训练
    print("Training...")
    trainer = ClassifierTrainer(model, device='cuda', num_classes=num_classes)
    trained_model = trainer.train(args, binary_dataset.train_loader, binary_dataset.val_loader, epochs=args.epochs,
                                  patience=args.patience)

    # 4. 测试
    print("Testing...")
    test_loss, test_acc, test_preds, test_labels = trainer.validate(args, binary_dataset.test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")

    return trained_model, test_acc


# 三分类的标签生成函数
def generate_three_class_labels(time_features, holiday_threshold=6):
    """
    生成三分类标签 (节假日/周末/工作日)
    Args:
        time_features: 时间特征，包含星期几和节假日标记
        holiday_threshold: 节假日判定阈值 (12小时中有多少小时是节假日)
    Returns:
        labels: 三分类标签
    """
    # time_features shape: (num_samples, seq_len, features)
    # 假设features中: [分钟, 小时, 星期几, 节假日标记]

    weekday = time_features[..., 2]  # 星期几
    holiday_flag = time_features[..., 3]  # 节假日标记

    labels = []

    for i in range(len(time_features)):
        # 计算节假日小时数
        holiday_hours = np.sum(holiday_flag[i])

        # 判断类别
        if holiday_hours >= holiday_threshold:  # 至少12小时是节假日
            labels.append(0)  # 节假日
        elif weekday[i, 0] >= 5:  # 星期六或星期日
            labels.append(1)  # 周末
        else:
            labels.append(2)  # 工作日

    return np.array(labels)


# 使用示例
if __name__ == "__main__":
    # 1. 二分类示例
    print("Training binary classifier...")
    model_binary, acc_binary = train_classification_model(
        data_config={},
        num_classes=2  # 节假日/常规日
    )

    print(f"\nResults:")
    print(f"Binary Classification Accuracy: {acc_binary:.2f}%")