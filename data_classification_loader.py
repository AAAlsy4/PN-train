import torch
import torch.utils.data as Data
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

class ClassificationDataset:
    def __init__(self, args, root_path, size=None, data_path='', scale=True, timeenc=2, classification_type='binary'):
        """
        分类数据加载器
        Args:
            args: 参数
            root_path: 根路径
            size: [seq_len, pred_len]
            data_path: 数据文件路径
            scale: 是否标准化
            timeenc: 时间特征编码
            classification_type: 'binary'或'ternary' (二分类或三分类)
        """
        if size is None:
            self.seq_len = 12  # 输入序列长度
        else:
            self.seq_len = size[0]

        self.args = args
        self.dataset = args.data
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.timeenc = timeenc
        self.classification_type = classification_type
        self.holiday_threshold = args.holiday_threshold if hasattr(args, 'holiday_threshold') else 12
        
        # 初始化scaler
        self.scaler = StandardScaler() if scale else None
        
        # 加载和预处理数据
        self.__read_data__()
        
        # 创建数据加载器
        self.train_loader = self.get_dataset(self.border1s[0], self.border2s[0], 'train')
        self.val_loader = self.get_dataset(self.border1s[1], self.border2s[1], 'val')
        self.test_loader = self.get_dataset(self.border1s[2], self.border2s[2], 'test')
        
        print(f"Dataset created: {len(self.train_loader.dataset)} train, "
              f"{len(self.val_loader.dataset)} val, {len(self.test_loader.dataset)} test samples")

    def __read_data__(self):
        """读取和预处理数据"""
        data_path = os.path.join(self.root_path, self.data_path)
        print(f"Loading data from: {data_path}")

        if '.h5' in data_path:
            df_raw = pd.read_hdf(os.path.join(self.root_path, self.data_path))
        elif '.npz' in data_path:
            df_raw = np.load(os.path.join(self.root_path, self.data_path), allow_pickle=True)
            df_raw = df_raw['data']
        elif '.txt' in data_path:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), delimiter=';')
            df_raw = df_raw.values
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        print(f"Raw data shape: {df_raw.shape}")
        
        # 划分数据集
        num_train = int(len(df_raw) * self.args.train_ratio)
        num_test = int(len(df_raw) * (1 - self.args.train_ratio) / 2)
        num_vali = len(df_raw) - num_train - num_test
        self.border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        self.border2s = [num_train, num_train + num_vali, len(df_raw)]
        
        print(f"Dataset split: train={num_train}, val={num_vali}, test={num_test}")

        # 提取数据
        if '.h5' in data_path:
            df_data = df_raw.values.astype(float)
        elif '.npz' in data_path:
            # 处理多维数据
            if len(df_raw.shape) == 3:
                # (时间步, 节点数, 特征数)
                df_data = df_raw[..., -1].astype(float)  # 取最后一个特征
            else:
                df_data = df_raw.astype(float)
        elif '.txt' in data_path:
            df_data = df_raw[:, 1:].astype(float)

        print(f"Data shape after extraction: {df_data.shape}")
        
        # 标准化
        if self.scale and self.scaler is not None:
            train_data = df_data[self.border1s[0]:self.border2s[0]]
            if len(train_data) > 0:
                self.scaler.fit(train_data)
                data = self.scaler.transform(df_data)
            else:
                data = df_data
                print("Warning: Train data is empty, skipping scaling")
        else:
            data = df_data

        # 提取时间特征
        if '.h5' in data_path:
            df_stamp = df_raw.index.tolist()
            df_stamp = pd.DataFrame(df_stamp, columns=['date'])
        elif '.npz' in data_path:
            if len(df_raw.shape) == 3:
                df_stamp = df_raw[:, 0, 0]  # 取第一个节点的第一个特征作为时间戳
            else:
                df_stamp = df_raw[:, 0]  # 取第一列作为时间戳
            df_stamp = pd.DataFrame(df_stamp, columns=['date'])
        elif '.txt' in data_path:
            df_stamp = pd.DataFrame(df_raw[:, 0], columns=['date'])  # 假设第一列是日期
        
        print(f"Time stamp shape: {df_stamp.shape}")
        
        data_stamp = time_features(df_stamp, timeenc=2, dataset=self.dataset)

        self.data = data
        self.data_stamp = data_stamp
        
        print(f"Final data shape: {self.data.shape}")
        print(f"Final data_stamp shape: {self.data_stamp.shape}")

    def generate_labels(self, time_features, holiday_index=-2):
        """
        生成分类标签
        Args:
            time_features: 时间特征数组 (seq_len, features)
            holiday_index: 节假日标记在时间特征中的索引位置
        Returns:
            label: 0=节假日, 1=常规日 (二分类) 或 0=节假日, 1=周末, 2=工作日 (三分类)
        """
        # 检查time_features的形状
        if len(time_features.shape) == 1:
            time_features = time_features.reshape(1, -1)

        # 提取节假日标记和星期几
        if time_features.shape[1] > abs(holiday_index):
            holiday_flags = time_features[:, holiday_index]  # 节假日标记
        else:
            # 如果没有节假日标记，默认都为常规日
            holiday_flags = np.zeros(time_features.shape[0])

        # 统计节假日时间步数量
        holiday_count = np.sum(holiday_flags > 0)

        if self.classification_type == 'binary':
            # 二分类: 节假日 vs 常规日

            # 方案1: 原始PN-Train方法 - 只要有节假日标记就认为是节假日
            # label = 0 if holiday_count > 0 else 1

            # 方案2: 图片建议方法 - 24h内有12h及以上为节假日的样本定义为节假日样本
            if hasattr(self, 'strict_holiday_threshold') and self.strict_holiday_threshold > 0:
                # 使用严格阈值 (如12小时)
                label = 0 if holiday_count >= self.strict_holiday_threshold else 1
            else:
                # 使用默认方法
                if self.holiday_threshold > 0:
                    label = 0 if holiday_count >= self.holiday_threshold else 1
                else:
                    label = 0 if holiday_count > 0 else 1

            return label

        elif self.classification_type == 'ternary':
            # 三分类: 节假日 > 周末 > 工作日

            # 首先判断是否为节假日 (使用严格阈值)
            if hasattr(self, 'strict_holiday_threshold') and self.strict_holiday_threshold > 0:
                is_holiday = holiday_count >= self.strict_holiday_threshold
            else:
                is_holiday = holiday_count >= self.holiday_threshold if self.holiday_threshold > 0 else holiday_count > 0

            if is_holiday:
                return 0  # 节假日

            # 尝试提取星期几信息
            if time_features.shape[1] > 2:
                # 假设第3列是星期几（0-6）
                day_of_week = np.mean(time_features[:, 2])
                weekday = int(np.round(day_of_week))

                # 检查星期几的范围
                if 0 <= weekday <= 6:
                    if weekday >= 5:  # 星期六(5)或星期日(6)
                        return 1  # 周末
                    else:
                        return 2  # 工作日

            # 如果没有星期几信息或范围不对，回退到二分类
            return 0 if is_holiday else 1

    def create_samples(self, start, end, split_type='train'):
        """
        创建分类样本
        Args:
            start: 起始索引
            end: 结束索引
            split_type: 数据集类型 ('train', 'val', 'test')
        Returns:
            samples: 样本列表
            labels: 标签列表
        """
        samples = []
        labels = []
        
        # 确保有足够的数据创建样本
        if end - start < self.seq_len:
            print(f"Warning: Not enough data to create samples from {start} to {end} with seq_len={self.seq_len}")
            return np.array(samples), np.array(labels)
        
        # 计算最大可能的起始索引
        max_start = end - self.seq_len
        
        for curr in range(start, max_start + 1):
            s_end = curr + self.seq_len
            
            # 提取输入序列
            s_x = self.data[curr:s_end]
            
            # 确保数据形状正确
            if len(s_x.shape) == 1:
                # 如果是一维数据，转换为二维 (seq_len, 1)
                s_x = s_x.reshape(-1, 1)
            
            # 检查序列长度
            if s_x.shape[0] != self.seq_len:
                print(f"Warning: Skipping sample at {curr}, expected length {self.seq_len}, got {s_x.shape[0]}")
                continue
            
            # 提取时间特征
            s_x_mark = self.data_stamp[curr:s_end]
            
            # 生成标签
            label = self.generate_labels(s_x_mark)
            
            # 将数据和标签添加到列表中
            samples.append(s_x)
            labels.append(label)
        
        # 转换为numpy数组
        try:
            samples_array = np.array(samples)
            labels_array = np.array(labels)
        except Exception as e:
            print(f"Error converting to numpy array: {e}")
            print(f"Sample shapes: {[s.shape for s in samples[:5]]}")
            # 如果形状不一致，尝试调整
            if samples:
                # 找到最大形状
                max_shape = max([s.shape for s in samples], key=lambda x: x[0])
                # 填充或截断所有样本到相同形状
                padded_samples = []
                for s in samples:
                    if s.shape != max_shape:
                        # 创建全零数组
                        padded = np.zeros(max_shape)
                        # 复制现有数据
                        min_shape = [min(s.shape[i], max_shape[i]) for i in range(len(max_shape))]
                        padded[:min_shape[0], :min_shape[1]] = s[:min_shape[0], :min_shape[1]]
                        padded_samples.append(padded)
                    else:
                        padded_samples.append(s)
                samples_array = np.array(padded_samples)
                labels_array = np.array(labels)
            else:
                samples_array = np.array([])
                labels_array = np.array([])
        
        print(f"Created {len(samples)} samples for {split_type} set")
        if len(samples) > 0:
            print(f"Sample shape: {samples_array.shape}")
            print(f"Labels distribution: {np.bincount(labels_array)}")
        
        return samples_array, labels_array
    
    def get_dataset(self, start, end, split_type='train'):
        """
        获取数据集加载器
        Args:
            start: 起始索引
            end: 结束索引
            split_type: 数据集类型
        Returns:
            data_loader: 数据加载器
        """
        # 创建样本和标签
        samples, labels = self.create_samples(start, end, split_type)
        
        # 如果没有样本，创建空数据集
        if len(samples) == 0:
            print(f"Warning: No samples created for {split_type} set")
            # 创建空tensor
            samples_tensor = torch.FloatTensor([])
            labels_tensor = torch.LongTensor([])
        else:
            # 转换为Tensor
            samples_tensor = torch.FloatTensor(samples)
            labels_tensor = torch.LongTensor(labels)
        
        # 创建数据集
        dataset = Data.TensorDataset(samples_tensor, labels_tensor)
        
        # 创建数据加载器
        shuffle_flag = True if split_type == 'train' else False
        drop_last = True if split_type == 'train' else False
        
        data_loader = Data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_flag,
            num_workers=self.num_workers,
            drop_last=drop_last
        )
        
        return data_loader
    
    def get_class_weights(self):
        """计算类别权重，用于处理类别不平衡"""
        # 获取训练集所有标签
        train_labels = []
        for _, labels in self.train_loader:
            train_labels.extend(labels.numpy())
        
        if not train_labels:
            print("Warning: No training labels found")
            return None
        
        train_labels = np.array(train_labels)
        
        # 计算每个类别的样本数
        if self.classification_type == 'binary':
            class_counts = np.bincount(train_labels, minlength=2)
        else:  # ternary
            class_counts = np.bincount(train_labels, minlength=3)
        
        # 计算权重
        total_samples = len(train_labels)
        class_weights = total_samples / (len(class_counts) * class_counts)
        
        # 归一化
        class_weights = class_weights / np.sum(class_weights)
        
        return torch.FloatTensor(class_weights)

# 数据工厂函数
def create_classification_dataset(args, classification_type='binary'):
    """
    创建分类数据集的工厂函数
    Args:
        args: 参数
        classification_type: 分类类型 ('binary' 或 'ternary')
    Returns:
        dataset: 分类数据集对象
    """
    # 设置默认参数
    if not hasattr(args, 'seq_len'):
        args.seq_len = 12  # 默认输入序列长度
    
    if not hasattr(args, 'train_ratio'):
        args.train_ratio = 0.7  # 默认训练集比例
    
    if not hasattr(args, 'batch_size'):
        args.batch_size = 32  # 默认批量大小
    
    if not hasattr(args, 'num_workers'):
        args.num_workers = 0  # 默认工作线程数，设置为0避免多线程问题
    
    if not hasattr(args, 'root_path'):
        args.root_path = './datasets/'
    
    if not hasattr(args, 'data_path'):
        # 尝试不同的文件扩展名
        possible_extensions = ['.h5', '.npz', '.txt']
        data_path = None
        for ext in possible_extensions:
            test_path = os.path.join(args.root_path, args.data + ext)
            if os.path.exists(test_path):
                data_path = args.data + ext
                break
        
        if data_path is None:
            raise FileNotFoundError(f"No data file found for {args.data} in {args.root_path}")
        args.data_path = data_path
    
    print(f"Creating classification dataset with args:")
    print(f"  Data: {args.data}")
    print(f"  Data path: {args.data_path}")
    print(f"  Seq len: {args.seq_len}")
    print(f"  Classification type: {classification_type}")
    
    # 创建数据集
    dataset = ClassificationDataset(
        args,
        root_path=args.root_path,
        size=[args.seq_len, 0],  # 分类任务不需要预测长度
        data_path=args.data_path,
        scale=True,
        timeenc=2,
        classification_type=classification_type
    )
    
    return dataset

# 测试函数
if __name__ == "__main__":
    # 模拟参数
    class Args:
        def __init__(self):
            self.data = 'metro-traffic'
            self.root_path = '/home/liushuyu/PN-Train/datasets/'
            self.data_path = 'metro-traffic/traffic.npz'  
            self.seq_len = 12
            self.train_ratio = 0.7
            self.batch_size = 32
            self.num_workers = 0
            self.holiday_threshold = 12  # 节假日时间步阈值
    
    args = Args()
    
    try:
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
        
        # 计算类别权重
        print("\nCalculating class weights...")
        class_weights = binary_dataset.get_class_weights()
        if class_weights is not None:
            print(f"Class weights: {class_weights}")
        
    except Exception as e:
        print(f"Error creating binary dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # try:
    #     # 测试三分类数据加载
    #     print("Testing ternary classification dataset...")
    #     ternary_dataset = create_classification_dataset(args, classification_type='ternary')
        
    #     # 查看一个批次的数据
    #     for batch_x, batch_y in ternary_dataset.train_loader:
    #         if len(batch_x) > 0:
    #             print(f"Batch x shape: {batch_x.shape}")
    #             print(f"Batch y shape: {batch_y.shape}")
    #             print(f"Batch y distribution: {torch.bincount(batch_y)}")
    #             break
        
    # except Exception as e:
    #     print(f"Error creating ternary dataset: {e}")
    #     import traceback
    #     traceback.print_exc()