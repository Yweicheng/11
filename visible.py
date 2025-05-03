import os
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns # Added for heatmap
from tqdm import tqdm
import warnings
import traceback

# 忽略 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
# 忽略 PIL 的 DecompressionBombWarning
warnings.filterwarnings("ignore", message="Possibly corrupt EXIF data.*")


# ===== 1. 数据处理器 (无变化) =====
class DataProcessor:
    """
    用于处理数值序列（如高度或阀门开度）的标准化和填充。
    """
    def __init__(self, method='standardize'):
        """
        初始化处理器。

        Args:
            method (str): 标准化方法 ('standardize' 或 'normalize')。
        """
        if method not in ['standardize', 'normalize']:
            raise ValueError("Method must be 'standardize' or 'normalize'")
        self.method = method
        self.scaler = None # 将在 fit 方法中初始化

    def fit(self, data_list):
        """
        根据提供的数据列表拟合Scaler。

        Args:
            data_list (list): 包含多个数值列表的列表，例如 [[h1, h2], [h3, h4, h5], ...]
                              或 [[v1], [v2], ...]
        """
        # 展平列表以拟合所有值
        all_values = [item for sublist in data_list for item in sublist if isinstance(item, (int, float))]
        if not all_values:
             print("Warning: No numeric data found to fit the scaler. Scaler will not be fitted.")
             self.scaler = None # Ensure scaler is None if no data
             return

        all_values = np.array(all_values).reshape(-1, 1)

        if self.method == 'standardize':
            self.scaler = StandardScaler()
        else: # normalize
            self.scaler = MinMaxScaler()

        try:
            self.scaler.fit(all_values)
            print(f"Scaler ({self.method}) fitted successfully.")
            if self.method == 'standardize':
                print(f"  Mean: {self.scaler.mean_[0]:.4f}, Scale (Std Dev): {self.scaler.scale_[0]:.4f}")
            else: # normalize
                print(f"  Min: {self.scaler.min_[0]:.4f}, Max: {self.scaler.data_max_[0]:.4f}")
        except Exception as e:
            print(f"Error fitting scaler: {e}. Scaler might not be usable.")
            self.scaler = None


    def transform(self, sequence, target_len):
        """
        对单个序列进行变换（标准化/归一化）并填充/截断到目标长度。

        Args:
            sequence (list): 要处理的数值列表，例如 [h1, h2, h3]。
            target_len (int): 目标序列长度。

        Returns:
            torch.Tensor: 处理后的张量，形状为 [target_len, 1]。
        """
        if self.scaler is None or not hasattr(self.scaler, 'mean_') and not hasattr(self.scaler, 'min_'): # Check if scaler is fitted
            # print(f"Warning: Scaler ({self.method}) not fitted. Returning zeros.") # Reduce noise
            return torch.zeros((target_len, 1), dtype=torch.float32)

        # 1. 转换序列为 NumPy 数组
        numeric_sequence = [item for item in sequence if isinstance(item, (int, float))]
        if not numeric_sequence:
            processed_sequence = np.zeros((0, 1)) # Start with empty array if no numbers
        else:
            sequence_np = np.array(numeric_sequence).reshape(-1, 1)
            # 2. 应用缩放器
            try:
                processed_sequence = self.scaler.transform(sequence_np)
            except Exception as e:
                 print(f"Error transforming sequence: {e}. Using zeros.")
                 processed_sequence = np.zeros_like(sequence_np, dtype=float) # Use zeros if transform fails

        # 3. 填充/截断
        current_len = processed_sequence.shape[0]
        if current_len >= target_len:
            # 截断 (取最后 target_len 个)
            padded_sequence = processed_sequence[current_len - target_len:]
        else:
            # 填充 (使用第一个有效值或0) - Pad at the beginning
            pad_value = processed_sequence[0, 0] if current_len > 0 else 0
            padding = np.full((target_len - current_len, 1), pad_value)
            padded_sequence = np.vstack((padding, processed_sequence)) # Pad at the beginning

        return torch.tensor(padded_sequence, dtype=torch.float32)

    def inverse_transform(self, data_tensor):
        """
        将处理后的数据（张量）逆转换为原始尺度。

        Args:
            data_tensor (torch.Tensor or np.ndarray): 已处理的数据，通常形状为 [N, 1]。

        Returns:
            np.ndarray: 逆转换后的数据，形状与输入类似。
        """
        if self.scaler is None or not (hasattr(self.scaler, 'mean_') or hasattr(self.scaler, 'min_')):
            # print(f"Warning: Scaler ({self.method}) not fitted. Returning original data.") # Reduce noise
            # Attempt to convert to numpy if it's a tensor, otherwise return as is
            if isinstance(data_tensor, torch.Tensor):
                 data_tensor = data_tensor.detach().cpu().numpy()
            return data_tensor

        if isinstance(data_tensor, torch.Tensor):
            data_np = data_tensor.detach().cpu().numpy()
        else:
            data_np = data_tensor

        if data_np.ndim == 1:
            data_np = data_np.reshape(-1, 1)
        elif data_np.ndim == 0: # Handle scalar tensor/numpy array
             data_np = data_np.reshape(1, 1)
        elif data_np.ndim > 2:
            # print(f"Warning: Inverse transform input has unexpected ndim={data_np.ndim}. Attempting reshape.")
            data_np = data_np.reshape(-1, 1)


        try:
             # Ensure input is float, handle potential NaNs
             data_np = np.nan_to_num(data_np.astype(float))
             # print(f"Inverse transform input shape: {data_np.shape}, scaler features: {self.scaler.n_features_in_}")
             if data_np.shape[1] != self.scaler.n_features_in_:
                  print(f"Warning: Inverse transform input shape {data_np.shape} incompatible with scaler features {self.scaler.n_features_in_}. Returning input.")
                  original_scale_data = data_np
             else:
                  original_scale_data = self.scaler.inverse_transform(data_np)
        except ValueError as ve:
             print(f"ValueError during inverse transform: {ve}. Scaler state: Fitted={hasattr(self.scaler.scaler, 'mean_') or hasattr(self.scaler, 'min_')}, Method={self.method}. Input shape: {data_np.shape}. Returning input array.")
             original_scale_data = data_np # Return input if inverse fails
        except Exception as e:
             print(f"Unexpected error during inverse transform: {e}. Returning input array.")
             original_scale_data = data_np # Return input if inverse fails

        return original_scale_data


# ===== 2. 数据集类 (无变化) =====
class ValveDataset(Dataset):
    """
    自定义数据集类，用于加载图像序列、高度序列和对应的阀门开度数据。
    (使用字典格式的 JSON 输入和绝对图像路径)
    """
    def __init__(self, data, image_dir, transform=None, seq_len=10, is_train=False, height_proc_method='normalize', valve_proc_method='normalize'):
        """
        初始化数据集。

        Args:
            data (list): 包含样本字典的列表 (由 main 函数转换得到)。
            image_dir (str): 图像文件的根目录 (如果 JSON 中是绝对路径, 此参数作用减小)。
            transform (callable, optional): 应用于每个图像的转换。
            seq_len (int): 每个样本的序列长度。
            is_train (bool): 如果为 True，则在此数据集上拟合数据处理器。
            height_proc_method (str): 用于高度数据的处理方法 ('standardize' 或 'normalize')。
            valve_proc_method (str): 用于阀门开度数据的处理方法 ('standardize' 或 'normalize')。
        """
        self.samples = data # 接收由 main 函数处理后的列表
        self.image_dir = image_dir # 存储以备用，但可能不使用
        self.transform = transform
        self.seq_len = seq_len
        self.height_processor = DataProcessor(method=height_proc_method)
        self.valve_processor = DataProcessor(method=valve_proc_method)

        if is_train:
            print("Fitting processors on training data...")
            # 提取所有序列用于拟合
            all_heights = [sample.get('height', []) for sample in self.samples if isinstance(sample.get('height'), list)]
            all_valves = [sample.get('valve_opening', []) for sample in self.samples if isinstance(sample.get('valve_opening'), list)]
            print(f"Found {len(all_heights)} height sequences and {len(all_valves)} valve sequences for fitting.")

            print(f"Fitting height processor ({height_proc_method})...")
            self.height_processor.fit(all_heights)

            print(f"Fitting valve processor ({valve_proc_method})...")
            # 仅使用每个序列的最后一个阀门值来拟合目标处理器
            last_valves = [[seq[-1]] for seq in all_valves if seq and isinstance(seq[-1], (int, float))]
            self.valve_processor.fit(last_valves)
            print("Processors fitted.")
        else:
            # 对于非训练集, 确保处理器存在但未拟合 (将由 set_processors 设置)
            self.height_processor.scaler = None
            self.valve_processor.scaler = None

        # 预计算每个样本的图像路径 (使用 'image_paths' 键和绝对路径)
        self.image_paths_per_sample = self._precompute_image_paths()

    def _precompute_image_paths(self):
        """ 预计算并缓存每个样本所需的图像路径列表 (处理绝对路径) """
        all_paths = []
        for sample in tqdm(self.samples, desc="Precomputing image paths"):
            # --- 使用 'image_paths' 键 ---
            image_files = sample.get('image_paths', []) # 使用 'image_paths'
            if not isinstance(image_files, list):
                image_files = []

            # --- 直接使用绝对路径, 并标准化路径分隔符 ---
            # 确保是字符串并替换反斜杠为正斜杠以增加跨平台兼容性
            paths = [p.replace('\\', '/') for p in image_files if p and isinstance(p, str)]

            # --- 处理序列长度: 截断或填充 ---
            current_len = len(paths)
            if current_len >= self.seq_len:
                # 取最后 seq_len 个路径
                final_paths = paths[current_len - self.seq_len:]
            else:
                # Pad with the *first* available image path if sequence is too short
                pad_path = paths[0] if paths else None
                # Pad at the beginning to reach seq_len
                final_paths = [pad_path] * (self.seq_len - current_len) + paths

            all_paths.append(final_paths)
        return all_paths


    def set_processors(self, height_processor, valve_processor):
        """
        为测试/验证数据集设置预先拟合好的处理器。
        """
        if height_processor is None or valve_processor is None:
            raise ValueError("Height and Valve processors must be provided")
        # 检查提供的处理器是否真的被拟合过
        if height_processor.scaler is None or not (hasattr(height_processor.scaler, 'mean_') or hasattr(height_processor.scaler, 'min_')):
             print("Warning: Provided height processor might not be fitted. Transformation might fail.")
        if valve_processor.scaler is None or not (hasattr(valve_processor.scaler, 'mean_') or hasattr(valve_processor.scaler, 'min_')):
             print("Warning: Provided valve processor might not be fitted. Inverse transform might fail.")

        self.height_processor = height_processor
        self.valve_processor = valve_processor


    def get_original_data(self, idx):
        """
        获取指定索引样本的原始高度序列和最后一个阀门开度。
        用于可视化时提供非标准化数据上下文。
        """
        sample = self.samples[idx]
        original_height_sequence = [item for item in sample.get('height', []) if isinstance(item, (int, float))]
        original_valve_sequence = [item for item in sample.get('valve_opening', []) if isinstance(item, (int, float))]
        original_valve_target = original_valve_sequence[-1] if original_valve_sequence else 0.0
        return original_height_sequence, original_valve_target


    def __len__(self):
        """ 返回数据集中的样本数量 """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本。

        Args:
            idx (int): 样本的索引。

        Returns:
            tuple: 包含三个元素的元组 (images, height_seq, valve_target)
        """
        # 在获取样本前检查处理器是否已设置或拟合
        # 这些检查是为了避免在训练/验证循环中因为处理器未设置而崩溃
        # 实际的转换失败将在transform方法内部处理并返回零
        if self.height_processor is None or self.height_processor.scaler is None:
             pass # Handled within DataProcessor.transform
        if self.valve_processor is None or self.valve_processor.scaler is None:
             pass # Handled within DataProcessor.transform

        sample = self.samples[idx]
        # --- 使用预计算的绝对路径 ---
        image_paths = self.image_paths_per_sample[idx]

        images = []
        default_img = None # Lazy initialization of default image

        for i, path in enumerate(image_paths):
            img = None
            # --- 直接检查绝对路径是否存在 ---
            # path 已经是处理好的绝对路径 (或 None 如果需要填充且无可用图像)
            if path and os.path.exists(path):
                try:
                    img = Image.open(path).convert('RGB')
                except Image.DecompressionBombError:
                    # print(f"Warning: DecompressionBombError for image {path}. Using default grey image.") # Reduce noise
                    pass
                except Exception as e:
                    # print(f"Error opening image {path}: {e}. Using default grey image.") # Reduce noise
                    pass # Silently use default if opening fails
            # else: # Handle cases where path is None or file doesn't exist (implicitly handled by img is None)
                # if path is not None: # Only warn if the path was supposed to exist but didn't
                #     print(f"Warning: Image file not found at {path}. Using default grey image.") # Reduce noise

            # 如果加载失败或路径无效/不存在，则使用默认图像
            if img is None:
                if default_img is None: # Create default only once if needed
                    default_img = Image.new('RGB', (224, 224), color=(128, 128, 128)) # Grey image
                img = default_img

            # 应用图像转换
            if self.transform:
                try:
                    img = self.transform(img)
                except Exception as e:
                    print(f"Error applying transform to image (path: {path}, index: {i}): {e}. Using zero tensor.")
                    # Use a zero tensor of the expected shape if transform fails
                    img = torch.zeros((3, 224, 224), dtype=torch.float32)
            else:
                 # Fallback default transform if none provided
                 temp_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
                 img = temp_transform(img)

            images.append(img)

        # 堆叠图像张量
        try:
            images_tensor = torch.stack(images) # Shape: [seq_len, 3, 224, 224]
        except RuntimeError as e:
            # print(f"Error stacking images for sample index {idx}: {e}. Check image dimensions. Returning zeros.") # Reduce noise
            images_tensor = torch.zeros((self.seq_len, 3, 224, 224), dtype=torch.float32)
        except Exception as e:
            print(f"Unexpected error stacking images for sample index {idx}: {e}. Returning zeros.")
            images_tensor = torch.zeros((self.seq_len, 3, 224, 224), dtype=torch.float32)

        # --- 处理高度序列 ---
        # 只有当处理器有效时才进行转换，否则 transform 返回零张量
        height_sequence = sample.get('height', [])
        if not isinstance(height_sequence, list):
            height_sequence = [] # Ensure it's a list
        height_data = self.height_processor.transform(height_sequence, self.seq_len) # Shape: [seq_len, 1]

        # --- 处理阀门目标值 ---
        # 只有当处理器有效时才进行转换，否则 transform 返回零张量
        valve_opening_sequence = sample.get('valve_opening', [])
        if not isinstance(valve_opening_sequence, list):
             valve_opening_sequence = [] # Ensure it's a list

        # 取序列中最后一个有效的数值作为目标
        last_valve_value = None
        if valve_opening_sequence:
            # Iterate backwards to find the last numeric value
            for val in reversed(valve_opening_sequence):
                 if isinstance(val, (int, float)):
                     last_valve_value = val
                     break
        if last_valve_value is None:
            last_valve_value = 0.0 # Default to 0 if no valid value found

        # Transform the single target value
        # .transform expects a list of values, returns [1, 1] tensor
        # Squeeze twice to get a scalar tensor [].shape
        valve_target = self.valve_processor.transform([last_valve_value], 1).squeeze(0).squeeze(0) # Result is a scalar tensor


        return images_tensor, height_data, valve_target


# ===== 3. 模型架构 (已修改: forward 方法添加可选返回 attention 权重) =====
class ImageSequenceRegressionModel(nn.Module):
    """
    基于 ResNet, LSTM, 和 Multi-head Attention 的图像序列与高度序列回归模型。
    使用图像作为 Query, 高度作为 Key/Value。 (修改版)
    """
    def __init__(self, lstm_hidden_size=128, lstm_layers=1, dropout_rate=0.2, height_lstm_hidden_size=32, num_attention_heads=4): # 添加注意力头数参数
        super().__init__()
        # --- CNN for Images ---
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            print("Using ResNet18_Weights.IMAGENET1K_V1 pretrained weights.")
        except AttributeError:
            print("Using legacy pretrained=True for ResNet18.")
            weights = True # Use True for older torchvision versions
        self.cnn = models.resnet18(weights=weights)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity() # Remove final layer

        # --- LSTM for Image Features ---
        self.lstm = nn.LSTM(num_ftrs, lstm_hidden_size, num_layers=lstm_layers, batch_first=True, dropout=dropout_rate if lstm_layers > 1 else 0)

        # --- LSTM for Height Sequence ---
        self.height_lstm = nn.LSTM(1, height_lstm_hidden_size, num_layers=1, batch_first=True) # Typically 1 layer is enough here

        # --- Multi-head Attention (Q=Image, K/V=Height) ---
        # embed_dim 是 Query 的维度 (lstm_hidden_size)
        # kdim 和 vdim 是 Key 和 Value 的维度 (height_lstm_hidden_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size,        # Query dimension (Image LSTM output)
            kdim=height_lstm_hidden_size,      # Key dimension (Height LSTM output)
            vdim=height_lstm_hidden_size,      # Value dimension (Height LSTM output)
            num_heads=num_attention_heads,
            dropout=dropout_rate,              # Apply dropout within attention
            batch_first=True                   # Expect (N, L, E) input format
        )
        # Layer Normalization after attention (applied to the attention output dimension)
        self.attn_layer_norm = nn.LayerNorm(lstm_hidden_size)


        # --- Regression Layer ---
        # Input is the output dimension of the attention mechanism (after LayerNorm)
        self.regression_layer = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64), # Input from attention/norm output
            nn.ReLU(),
            nn.Dropout(dropout_rate),        # Dropout before final layer
            nn.Linear(64, 1)                 # Output a single value
        )

    # --- MODIFIED: Added return_attention_weights parameter and conditional return ---
    def forward(self, x_img, x_height, return_attention_weights=False):
        # x_img: [batch_size, seq_len, c, h, w]
        # x_height: [batch_size, seq_len, 1]
        batch_size, seq_len, c, h, w = x_img.shape

        # 1. CNN Feature Extraction
        # Reshape for CNN: [B*L, C, H, W]
        x_img = x_img.view(batch_size * seq_len, c, h, w)
        try:
            img_features = self.cnn(x_img) # Output: [B*L, num_ftrs]
        except Exception as e:
            print(f"Error during CNN forward pass: {e}")
            # Return zero prediction matching batch size if CNN fails
            output = torch.zeros((batch_size, 1), device=x_img.device)
            if return_attention_weights:
                 # Return dummy weights if requested but forward failed early
                 return output, torch.zeros((batch_size, seq_len, seq_len), device=x_img.device)
            else:
                 return output

        # Reshape back for LSTM: [B, L, num_ftrs]
        img_features = img_features.view(batch_size, seq_len, -1)

        # 2. Image LSTM
        try:
            # lstm_out contains hidden states for all time steps: [B, L, lstm_hidden_size]
            lstm_out, _ = self.lstm(img_features)
        except Exception as e:
            print(f"Error during Image LSTM forward pass: {e}")
            output = torch.zeros((batch_size, 1), device=x_img.device)
            if return_attention_weights:
                 return output, torch.zeros((batch_size, seq_len, seq_len), device=x_img.device)
            else:
                 return output


        # 3. Height LSTM
        try:
            # height_lstm_out: [B, L, height_lstm_hidden_size] - full sequence output
            height_lstm_out, _ = self.height_lstm(x_height)
        except Exception as e:
            print(f"Error during Height LSTM forward pass: {e}")
            output = torch.zeros((batch_size, 1), device=x_img.device)
            if return_attention_weights:
                 return output, torch.zeros((batch_size, seq_len, seq_len), device=x_img.device)
            else:
                 return output


        # 4. Multi-head Attention (Q=Image, K/V=Height)
        # Query (Q): Based on the entire image sequence representation from LSTM.
        query = lstm_out                         # Q: [B, L, lstm_hidden_size]
        # Key (K): Based on the entire height sequence representation from LSTM.
        key = height_lstm_out                    # K: [B, L, height_lstm_hidden_size]
        # Value (V): Based on the entire height sequence representation from LSTM.
        value = height_lstm_out                  # V: [B, L, height_lstm_hidden_size]

        try:
            # Pass need_weights=True if return_attention_weights is True
            attn_output, attn_output_weights = self.attention(
                query=query,
                key=key,
                value=value,
                need_weights=return_attention_weights # --- MODIFIED ---
            )
            # If need_weights was False, attn_output_weights will be None.
            # If need_weights was True, attn_output_weights is [B, num_heads, L_Q, L_KV].
            # For batch_first=True, it's [B, L_Q, L_KV].
            # The return from MultiheadAttention is already averaged over heads when batch_first=True.
            # The shape when need_weights=True and batch_first=True is [B, L_target, L_source]
            # L_target is query seq len (seq_len), L_source is key/value seq len (seq_len)
            # So, attn_output_weights is [B, seq_len, seq_len]
        except Exception as e:
            print(f"Error during Attention forward pass: {e}")
            output = torch.zeros((batch_size, 1), device=query.device)
            if return_attention_weights:
                 return output, torch.zeros((batch_size, seq_len, seq_len), device=query.device) # Dummy weights
            else:
                 return output


        # Use the output corresponding to the *last* time step of the query (image) sequence
        # attn_output is [B, L, lstm_hidden_size], take the last one -> [B, lstm_hidden_size]
        last_attn_output = attn_output[:, -1, :]

        # Apply Layer Normalization to the last time step's attention output
        attn_output_norm = self.attn_layer_norm(last_attn_output)

        # 5. Regression Layer
        # Feed the normalized attention output (last time step) into the regression head
        try:
            output = self.regression_layer(attn_output_norm) # Output: [B, 1]
        except Exception as e:
            print(f"Error during Regression Layer forward pass: {e}")
            output = torch.zeros((batch_size, 1), device=attn_output_norm.device)
            if return_attention_weights:
                 # Return the (potentially dummy) weights if requested
                 # attn_output_weights is None if need_weights=False, handle that
                 return output, attn_output_weights if return_attention_weights else torch.zeros((batch_size, seq_len, seq_len), device=attn_output_norm.device)
            else:
                 return output

        # --- MODIFIED: Conditional return based on flag ---
        if return_attention_weights:
             # Return prediction and weights
             return output, attn_output_weights # attn_output_weights is [B, seq_len, seq_len]
        else:
             # Return only prediction (default)
             return output


# ===== 4. 训练和评估工具 (修改 evaluate_model 捕获 attention 权重并调用绘图) =====

def setup_device(visible_devices="0"):
    """设置运行设备（GPU 或 CPU）。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        print(f"Using {gpu_count} GPU(s): {', '.join([torch.cuda.get_device_name(i) for i in range(gpu_count)])}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def setup_data_parallel(model, device):
    """设置数据并行。"""
    model = model.to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs.")
        # Wrap the model in DataParallel
        model = nn.DataParallel(model) # Automatically uses all visible GPUs
    return model

def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    batches_processed = 0
    for images, height, valve in tqdm(loader, desc="Training", leave=False):
        try:
            # Move data to the correct device
            images, height, valve = images.to(device), height.to(device), valve.to(device)
        except Exception as e:
            print(f"Error moving batch data to device {device}: {e}. Skipping batch.")
            continue

        optimizer.zero_grad()

        try:
            # Forward pass (default: return_attention_weights=False)
            pred = model(images, height) # pred shape should be [B, 1]
        except Exception as e:
            print(f"Error during model forward pass in training: {e}")
            # traceback.print_exc() # Optional: print full traceback for debugging
            continue # Skip this batch

        # Ensure prediction and target shapes are compatible for loss calculation
        # Target `valve` is expected to be [B] (scalar target per sample)
        # Prediction `pred` is expected to be [B, 1]
        if pred.shape[0] != valve.shape[0]:
             print(f"Shape mismatch: pred {pred.shape}, valve {valve.shape}. Skipping batch.")
             continue
        if pred.ndim != 2 or pred.shape[1] != 1:
             print(f"Unexpected prediction shape: {pred.shape}. Expected [B, 1]. Skipping batch.")
             continue

        try:
            # Squeeze prediction for MSELoss with scalar target: [B, 1] -> [B]
            loss = criterion(pred.squeeze(-1), valve.float())
        except Exception as e:
            print(f"Error calculating loss: {e}. Pred shape after squeeze: {pred.squeeze(-1).shape}, Valve shape: {valve.shape}. Skipping batch.")
            continue

        # Check for NaN/Inf loss
        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss detected ({loss.item()}). Skipping backward/step.")
            continue

        try:
            # Backward pass
            loss.backward()
        except Exception as e:
            print(f"Error during backward pass: {e}. Skipping optimizer step.")
            # Zero grad again just in case gradients are corrupted
            optimizer.zero_grad()
            continue

        try:
            # Gradient Clipping (optional but recommended)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()
        except Exception as e:
            print(f"Error during optimizer step: {e}.")
            # traceback.print_exc()
            continue # Continue to next batch even if step fails

        total_loss += loss.item()
        batches_processed += 1

    # Avoid division by zero if loader was empty or all batches failed
    return (total_loss / batches_processed) if batches_processed > 0 else 0.0

def validate(model, loader, criterion, device):
    """在验证集上评估模型。"""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    batches_processed = 0
    with torch.no_grad(): # Disable gradient calculations
        for images, height, valve in tqdm(loader, desc="Validating", leave=False):
            try:
                images, height, valve = images.to(device), height.to(device), valve.to(device)
            except Exception as e:
                 print(f"Error moving batch data to device {device} during validation: {e}. Skipping batch.")
                 continue

            try:
                 # Forward pass (default: return_attention_weights=False)
                 pred = model(images, height) # pred shape: [B, 1]
            except Exception as e:
                 print(f"Error during model forward pass in validation: {e}. Skipping batch.")
                 continue

            if pred.shape[0] != valve.shape[0]:
                 print(f"Shape mismatch in validation: pred {pred.shape}, valve {valve.shape}. Skipping.")
                 continue
            if pred.ndim != 2 or pred.shape[1] != 1:
                 print(f"Unexpected validation prediction shape: {pred.shape}. Expected [B, 1]. Skipping.")
                 continue

            try:
                # Calculate loss (MSE requires pred [B], target [B])
                loss = criterion(pred.squeeze(-1), valve.float())
                # Check for NaN/Inf loss
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    batches_processed += 1
                else:
                    print(f"Warning: Non-finite validation loss detected ({loss.item()}).")

            except Exception as e:
                print(f"Error calculating validation loss: {e}. Pred: {pred.squeeze(-1).shape}, Valve: {valve.shape}. Skipping batch.")
                continue

    # Return average loss, handle division by zero
    return (total_loss / batches_processed) if batches_processed > 0 else float('inf')

# --- MODIFIED: evaluate_model now captures attention weights and calls plot_attention_weights ---
def evaluate_model(model, loader, device, valve_processor, height_processor, attention_plot_filename=None):
    """
    在测试集上评估模型，计算多个指标。
    可选地捕获第一个 batch 的注意力权重并绘制。
    """
    model.eval()
    actuals_original = []
    predictions_original = []

    # Crucial check: Ensure the valve processor is fitted and ready for inverse transform
    if valve_processor is None or valve_processor.scaler is None or not (hasattr(valve_processor.scaler, 'mean_') or hasattr(valve_processor.scaler, 'min_')):
         print("Error: Evaluate requires a valid and fitted valve processor for inverse transformation.")
         # Cannot proceed with inverse transform or evaluation metrics calculation
         return {'std': np.nan, 'r2': np.nan, 'pearson_r': np.nan, 'mae': np.nan,
                 'actuals_original': [], 'predictions_original': []}

    # Ensure height processor is also valid if plotting attention
    if attention_plot_filename and (height_processor is None or height_processor.scaler is None or not (hasattr(height_processor.scaler, 'mean_') or hasattr(height_processor.scaler, 'min_'))):
         print("Warning: Height processor not valid. Cannot plot attention weights with original height values.")
         attention_plot_filename = None # Disable attention plotting if height processor is bad

    # Variables to capture data for the first batch for visualization
    first_batch_attn_weights = None
    first_batch_height_normalized = None
    first_batch_valve_normalized = None
    first_batch_indices = None # Need indices to get original data from dataset
    captured_first_batch = False


    with torch.no_grad():
        for batch_idx, (images, height_normalized, valve_normalized) in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
            try:
                # Only images and height need to go to device for model input
                images = images.to(device)
                height_normalized = height_normalized.to(device)
                # valve_normalized stays on CPU as it's the target (already loaded)
            except Exception as e:
                 print(f"Error moving batch data to device {device} during evaluation: {e}. Skipping batch.")
                 continue

            try:
                # Get model prediction (normalized scale)
                # --- MODIFIED: Conditionally request attention weights for the first batch ---
                if attention_plot_filename and not captured_first_batch:
                     pred_normalized, attn_weights = model(images, height_normalized, return_attention_weights=True)
                     # Capture data for the first batch
                     first_batch_attn_weights = attn_weights.detach().cpu().numpy() # [B, seq_len, seq_len]
                     first_batch_height_normalized = height_normalized.detach().cpu().numpy() # [B, seq_len, 1]
                     first_batch_valve_normalized = valve_normalized.detach().cpu().numpy() # [B]
                     # Need the actual indices processed by this batch to retrieve original data later if needed
                     # DataLoader doesn't easily provide original dataset indices directly
                     # For simplicity, we'll rely on the processors to inverse transform the batch data
                     # If strict original data access is needed, batch sampling needs modification (e.g., return indices)
                     captured_first_batch = True
                     pred_normalized = pred_normalized.cpu() # Move prediction to CPU as before
                else:
                    pred_normalized = model(images, height_normalized).cpu() # Shape: [B, 1]

            except Exception as e:
                 print(f"Error during model forward pass in evaluation: {e}. Skipping batch.")
                 # traceback.print_exc() # Uncomment for detailed error
                 continue

            # Inverse transform predictions and actuals
            try:
                # pred_normalized is tensor [B, 1], needs numpy [B, 1] for inverse_transform
                pred_original = valve_processor.inverse_transform(pred_normalized.numpy().reshape(-1, 1))
                # valve_normalized is tensor [B], needs numpy [B, 1] for inverse_transform
                actual_original = valve_processor.inverse_transform(valve_normalized.cpu().numpy().reshape(-1, 1))
            except Exception as e:
                 print(f"Error during inverse transformation in evaluation: {e}. Skipping this batch.")
                 # traceback.print_exc()
                 continue

            # Ensure shapes are reasonable after inverse transform (expecting [B, 1])
            if pred_original.ndim == 2 and actual_original.ndim == 2:
                actuals_original.extend(actual_original.flatten().tolist())
                predictions_original.extend(pred_original.flatten().tolist())
            else:
                print(f"Warning: Unexpected shapes after inverse transform. Actual: {actual_original.shape}, Pred: {pred_original.shape}. Skipping batch.")


    # --- MODIFIED: Call plot_attention_weights after the evaluation loop if data was captured ---
    if captured_first_batch and attention_plot_filename:
        print(f"\nPlotting attention weights for the first {first_batch_attn_weights.shape[0]} samples in the test set...")
        try:
            # Pass the captured data and processors to the plotting function
            plot_attention_weights(
                first_batch_attn_weights,
                first_batch_height_normalized,
                first_batch_valve_normalized,
                valve_processor,
                height_processor,
                attention_plot_filename,
                loader.dataset.seq_len # Pass seq_len from dataset
            )
        except Exception as e:
            print(f"Error generating attention plot: {e}")
            traceback.print_exc()


    if not actuals_original or not predictions_original:
        print("Warning: No valid data points collected for evaluation.")
        return {'std': np.nan, 'r2': np.nan, 'pearson_r': np.nan, 'mae': np.nan,
                'actuals_original': [], 'predictions_original': []}

    # Convert lists to numpy arrays for metric calculation
    actuals_original = np.array(actuals_original)
    predictions_original = np.array(predictions_original)

    # Filter out potential NaN/Inf values introduced during processing or inverse transform
    valid_indices = np.isfinite(actuals_original) & np.isfinite(predictions_original)
    if not np.all(valid_indices):
        num_invalid = np.sum(~valid_indices)
        print(f"Warning: Found {num_invalid} non-finite values in evaluation results. Removing them.")
        actuals_original = actuals_original[valid_indices]
        predictions_original = predictions_original[valid_indices]

    # Check if enough valid points remain for metrics
    if len(actuals_original) < 2:
        print("Warning: Not enough valid data points (< 2) to calculate metrics.")
        return {'std': np.nan, 'r2': np.nan, 'pearson_r': np.nan, 'mae': np.nan,
                'actuals_original': actuals_original.tolist(), 'predictions_original': predictions_original.tolist()}

    # Calculate metrics
    error = actuals_original - predictions_original
    std_dev = np.std(error) # Standard deviation of the prediction error

    try:
        r2 = r2_score(actuals_original, predictions_original)
    except ValueError:
        print("Warning: Could not calculate R2 score (possibly due to insufficient data).")
        r2 = np.nan

    try:
        mae = mean_absolute_error(actuals_original, predictions_original)
    except ValueError:
        print("Warning: Could not calculate MAE (possibly due to insufficient data).")
        mae = np.nan

    pearson_r = np.nan
    pearson_p = np.nan
    # Pearson R requires at least 2 points and variance in both arrays
    if len(np.unique(actuals_original)) > 1 and len(np.unique(predictions_original)) > 1:
         try:
            pearson_r, pearson_p = pearsonr(actuals_original.flatten(), predictions_original.flatten())
         except ValueError:
            print("Warning: Could not calculate Pearson correlation (ValueError).")
    else:
         print("Warning: Pearson correlation cannot be calculated (constant values detected or insufficient data).")

    return {'std': std_dev, 'r2': r2, 'pearson_r': pearson_r, 'mae': mae,
            'actuals_original': actuals_original.tolist(), 'predictions_original': predictions_original.tolist()}


# ===== 5. 绘图工具 (已修改: 添加注意力权重绘图函数) =====
def plot_predictions(actual, predicted, filename="prediction_comparison.png"):
    """绘制实际值 vs 预测值。 (散点图)"""
    if not actual or not predicted or len(actual) != len(predicted):
        print(f"Warning: Cannot plot predictions. Invalid data provided (Actual: {len(actual)}, Predicted: {len(predicted)}).")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, alpha=0.5, label=f'Predicted vs Actual (N={len(actual)})')

    # Determine plot limits safely
    try:
        # Filter finite values for limit calculation
        finite_actual = [x for x in actual if np.isfinite(x)]
        finite_predicted = [x for x in predicted if np.isfinite(x)]
        if not finite_actual or not finite_predicted:
            raise ValueError("No finite values to determine plot limits.")
        min_val = min(np.min(finite_actual), np.min(finite_predicted))
        max_val = max(np.max(finite_actual), np.max(finite_predicted))
        # Add a small buffer to limits
        buffer = (max_val - min_val) * 0.05
        min_val -= buffer
        max_val += buffer
        # Ensure min < max to avoid errors
        if min_val >= max_val:
             min_val = min(finite_actual + finite_predicted) - 1
             max_val = max(finite_actual + finite_predicted) + 1

        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
        plt.xlim(min_val, max_val) # Set limits based on data range
        plt.ylim(min_val, max_val)
    except (ValueError, TypeError) as e: # Handle case where actual or predicted might be empty or contain non-numeric after filtering
        print(f"Warning: Could not determine plot limits for y=x line. Error: {e}")
        min_val, max_val = None, None # Indicate limits couldn't be set

    plt.xlabel("Actual Valve Opening (Original Scale)")
    plt.ylabel("Predicted Valve Opening (Original Scale)")
    plt.title("Prediction vs Actual Values Comparison (Scatter Plot)")
    plt.legend()
    plt.grid(True)

    try:
        plt.savefig(filename)
        print(f"Prediction comparison plot saved to: {filename}")
    except Exception as e:
        print(f"Error saving prediction plot to {filename}: {e}")
    plt.close() # Close the plot to free memory

def plot_prediction_lines(actual, predicted, filename="prediction_lines.png"):
    """绘制实际值和预测值的折线图。"""
    if not actual or not predicted or len(actual) != len(predicted):
        print(f"Warning: Cannot plot prediction lines. Invalid data provided (Actual: {len(actual)}, Predicted: {len(predicted)}).")
        return

    plt.figure(figsize=(15, 7)) # Use a wider figure suitable for time series/sequences
    num_samples = len(actual)
    indices = range(num_samples) # X-axis will be the sample index

    # Filter out non-finite values for plotting
    actual_np = np.array(actual)
    predicted_np = np.array(predicted)
    valid_mask = np.isfinite(actual_np) & np.isfinite(predicted_np)

    if not np.any(valid_mask):
        print("Warning: No finite data points to plot for prediction lines.")
        plt.close()
        return

    indices_valid = np.array(indices)[valid_mask]
    actual_valid = actual_np[valid_mask]
    predicted_valid = predicted_np[valid_mask]

    plt.plot(indices_valid, actual_valid, 'b-', label=f'Actual Values (N={len(actual_valid)})', linewidth=1.5)
    plt.plot(indices_valid, predicted_valid, 'r--', label=f'Predicted Values (N={len(predicted_valid)})', linewidth=1.5, alpha=0.8) # Use dashed line for prediction

    plt.xlabel("Sample Index (in Test Set)")
    plt.ylabel("Valve Opening (Original Scale)")
    plt.title("Actual vs. Predicted Valve Opening Over Samples (Line Plot)")
    plt.legend()
    plt.grid(True)
    # Set x-axis limits based on the original number of samples
    plt.xlim(0, num_samples - 1 if num_samples > 1 else 1)

    try:
        plt.savefig(filename)
        print(f"Prediction line plot saved to: {filename}")
    except Exception as e:
        print(f"Error saving prediction line plot to {filename}: {e}")
    plt.close() # Close the plot to free memory

def plot_loss_curves(train_losses, val_losses, filename="loss_curves.png"):
    """绘制损失曲线。"""
    # Filter out potential non-numeric or infinite values
    valid_train_losses = [l for l in train_losses if isinstance(l, (int, float)) and np.isfinite(l)]
    valid_val_losses = [l for l in val_losses if isinstance(l, (int, float)) and np.isfinite(l)]

    if not valid_train_losses and not valid_val_losses:
        print("Warning: No valid loss data to plot.")
        return

    plt.figure(figsize=(10, 6))
    # Use original length for x-axis, even if some data points were invalid
    epochs = range(1, max(len(train_losses), len(val_losses)) + 1)

    # Plot only if valid data exists
    if valid_train_losses:
        plt.plot(epochs[:len(valid_train_losses)], valid_train_losses, 'bo-', label='Training Loss')
    if valid_val_losses:
        plt.plot(epochs[:len(valid_val_losses)], valid_val_losses, 'ro-', label='Validation Loss')

    plt.xlabel("Epoch")
    plt.ylabel("Average Loss (MSE)")
    plt.title("Training and Validation Loss Curves")
    # Only show legend if at least one curve was plotted
    if valid_train_losses or valid_val_losses:
        plt.legend()
    plt.grid(True)

    try:
        plt.savefig(filename)
        print(f"Loss curves plot saved to: {filename}")
    except Exception as e:
        print(f"Error saving loss curves plot to {filename}: {e}")
    plt.close() # Close the plot to free memory


# --- ADDED: Function to plot attention weights ---
def plot_attention_weights(attn_weights_batch, height_normalized_batch, valve_normalized_batch,
                           valve_processor, height_processor, filename, seq_len, num_samples_to_plot=4):
    """
    绘制一批样本的注意力权重热力图。

    Args:
        attn_weights_batch (np.ndarray): 第一个 batch 的注意力权重 [batch_size, seq_len, seq_len]。
        height_normalized_batch (np.ndarray): 第一个 batch 的标准化高度序列 [batch_size, seq_len, 1]。
        valve_normalized_batch (np.ndarray): 第一个 batch 的标准化阀门目标 [batch_size]。
        valve_processor (DataProcessor): 用于逆变换阀门值的处理器。
        height_processor (DataProcessor): 用于逆变换高度值的处理器。
        filename (str): 保存图的文件名。
        seq_len (int): 序列长度。
        num_samples_to_plot (int): 从 batch 中绘制的样本数量。
    """
    batch_size = attn_weights_batch.shape[0]
    num_samples_to_plot = min(num_samples_to_plot, batch_size) # Plot at most batch_size samples

    if num_samples_to_plot <= 0:
        print("Warning: No samples to plot attention weights for.")
        return

    print(f"Generating attention weight plots for the first {num_samples_to_plot} samples...")

    # Determine grid size for subplots
    cols = 2
    rows = (num_samples_to_plot + cols - 1) // cols

    plt.figure(figsize=(cols * 6, rows * 5)) # Adjust figure size based on number of subplots

    for i in range(num_samples_to_plot):
        if i >= batch_size: # Should not happen with min() but safety check
            break

        ax = plt.subplot(rows, cols, i + 1)

        # Get attention weights for this sample [seq_len, seq_len]
        sample_attn = attn_weights_batch[i, :, :]

        # Inverse transform height and valve for context
        # height_normalized_batch[i] is [seq_len, 1]
        original_height_seq = height_processor.inverse_transform(height_normalized_batch[i]).flatten()
        # valve_normalized_batch[i] is scalar
        original_valve_target = valve_processor.inverse_transform(np.array([valve_normalized_batch[i]]).reshape(-1, 1)).flatten()[0]


        # Create heatmap using seaborn
        sns.heatmap(sample_attn, annot=False, cmap='viridis', ax=ax, cbar=True) # Annot=True adds values (can be cluttered)

        ax.set_xlabel("Key/Value (Height Sequence Steps)")
        ax.set_ylabel("Query (Image Sequence Steps)")
        # Add title with sample index and original valve target
        ax.set_title(f"Sample {i+1} (Target Valve: {original_valve_target:.2f})")

        # Optional: Add tick labels showing corresponding original height values
        # This can make ticks labels too crowded if seq_len is large
        # tick_labels = [f'{h:.1f}' for h in original_height_seq]
        # ax.set_xticks(np.arange(seq_len) + 0.5) # Center ticks between cells
        # ax.set_yticks(np.arange(seq_len) + 0.5)
        # ax.set_xticklabels(tick_labels, rotation=90)
        # ax.set_yticklabels(tick_labels)

        # Set integer ticks for clarity
        ax.set_xticks(np.arange(seq_len) + 0.5)
        ax.set_yticks(np.arange(seq_len) + 0.5)
        ax.set_xticklabels(np.arange(seq_len))
        ax.set_yticklabels(np.arange(seq_len))


    plt.tight_layout() # Adjust layout to prevent overlap

    try:
        plt.savefig(filename)
        print(f"Attention weights plot saved to: {filename}")
    except Exception as e:
        print(f"Error saving attention weights plot to {filename}: {e}")
    plt.close() # Close the plot to free memory


# ===== 6. 主函数 (已修改: 添加注意力图文件名和调用) =====
def main():
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="Train an Image Sequence + Height Regression Model with Attention")
    # Data and Paths
    parser.add_argument('--train_data', type=str, default="/home/temp03/ywc/veiw_json/new_data/train_data.json", help='Path to training JSON data (list or dict format)')
    parser.add_argument('--test_data', type=str, default="/home/temp03/ywc/veiw_json/new_data/test_data.json", help='Path to testing JSON data (list or dict format)')
    parser.add_argument('--image_dir', type=str, default="/home/temp03/ywc/veiw_json/valve_img/", help='Base directory for images (only needed if paths in JSON are relative)')
    parser.add_argument('--output_dir', type=str, default="./results_attention420/", help='Output directory for models, logs, plots')
    # Model Hyperparameters
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length for images and height')
    parser.add_argument('--lstm_hidden', type=int, default=128, help='Hidden size for image feature LSTM')
    parser.add_argument('--lstm_layers', type=int, default=1, help='Number of layers for image feature LSTM')
    parser.add_argument('--height_lstm_hidden', type=int, default=32, help='Hidden size for height sequence LSTM')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads in MultiheadAttention')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for LSTM, Attention, and Regression layers')
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate for Adam optimizer')
    # Data Loading & Preprocessing
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--height_proc', type=str, default='normalize', choices=['normalize', 'standardize'], help='Processing method for height sequence (normalize: 0-1, standardize: mean 0 std 1)')
    parser.add_argument('--valve_proc', type=str, default='normalize', choices=['normalize', 'standardize'], help='Processing method for valve opening target (normalize: 0-1, standardize: mean 0 std 1)')
    # Execution Control
    parser.add_argument('--pretrained', type=str, default=None, help='Path to load pretrained model weights (.pth file)')
    parser.add_argument('--visible_devices', type=str, default="0,1,2,3", help='Comma-separated list of GPU IDs to use (e.g., "0,1")')
    parser.add_argument('--eval_only', action='store_true', help='If set, only perform evaluation on the test set using --pretrained model (requires --pretrained)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # --- Reproducibility ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # CUDNN settings for reproducibility (can slow down training)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # --- Setup Output Directory and Filenames ---
    os.makedirs(args.output_dir, exist_ok=True)
    # Include key hyperparameters in filenames
    proc_suffix = f"h_{args.height_proc}_v_{args.valve_proc}"
    model_config_suffix = f"seq{args.seq_len}_imLSTM{args.lstm_hidden}x{args.lstm_layers}_hLSTM{args.height_lstm_hidden}_attn{args.num_heads}_drop{args.dropout}"
    base_filename = f"model_{model_config_suffix}_{proc_suffix}"

    model_save_path = os.path.join(args.output_dir, f'{base_filename}_best.pth')
    pred_plot_path = os.path.join(args.output_dir, f'{base_filename}_predictions.png') # Scatter plot
    pred_line_plot_path = os.path.join(args.output_dir, f'{base_filename}_prediction_lines.png') # Line plot
    loss_plot_path = os.path.join(args.output_dir, f'{base_filename}_loss_curves.png')
    # --- ADDED: Attention plot filename ---
    attention_plot_path = os.path.join(args.output_dir, f'{base_filename}_attention_weights.png')

    print("--- Configuration ---")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("---------------------")
    print(f"Output directory: {args.output_dir}")
    print(f"Model save path: {model_save_path}")
    print(f"Prediction scatter plot path: {pred_plot_path}")
    print(f"Prediction line plot path: {pred_line_plot_path}")
    print(f"Loss plot path: {loss_plot_path}")
    print(f"Attention plot path: {attention_plot_path}") # --- ADDED ---
    # TODO: Add logging to file (log_file_path)

    # --- Setup Device ---
    device = setup_device(visible_devices=args.visible_devices)

    # --- Load and Prepare Data ---
    def load_json_data(filepath):
        print(f"Loading data from: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} entries (before type check).")

            # Handle dict or list format
            if isinstance(data, dict):
                print("Data is a dictionary. Converting values to a list.")
                processed_data = list(data.values())
            elif isinstance(data, list):
                print("Data is already a list.")
                processed_data = data
            else:
                print(f"Error: Unexpected data type ({type(data)}) loaded from JSON. Expected list or dict.")
                return None

            if not processed_data:
                print(f"Warning: Data list is empty after processing {filepath}.")
            else:
                 # Check if items are dictionaries as expected
                 if not all(isinstance(item, dict) for item in processed_data):
                      print("Warning: Not all items in the data list are dictionaries.")
                 # else:
                      # print(f"Data processed into a list of {len(processed_data)} dictionaries.") # Reduce noise

            return processed_data

        except FileNotFoundError:
            print(f"Error: Data file not found - {filepath}. Please check the path.")
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON file - {filepath}: {e}. Please check file format.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during data loading from {filepath}: {e}")
            return None

    train_data = load_json_data(args.train_data)
    test_data = load_json_data(args.test_data)

    if train_data is None or test_data is None:
        print("Exiting due to data loading errors.")
        return
    if not train_data:
        print(f"Error: Training data is empty after loading and processing {args.train_data}. Exiting.")
        return
    if not test_data:
        print(f"Error: Test data is empty after loading and processing {args.test_data}. Exiting.")
        return

    # Print example sample (optional, can remove for cleaner output)
    # print("Example training sample:", json.dumps(train_data[0], indent=2, ensure_ascii=False))
    # print("Example test sample:", json.dumps(test_data[0], indent=2, ensure_ascii=False))


    # --- Image Transformations ---
    # Standard ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("Image transformations defined.")

    # --- Create Datasets and DataLoaders ---
    print('Creating datasets...')
    train_dataset = None
    test_dataset = None
    try:
        # Create training dataset (is_train=True fits the processors)
        train_dataset = ValveDataset(
            train_data, args.image_dir, transform, seq_len=args.seq_len, is_train=True,
            height_proc_method=args.height_proc, valve_proc_method=args.valve_proc
        )

        # Check if processors were successfully fitted
        train_height_proc_valid = train_dataset.height_processor and train_dataset.height_processor.scaler and (hasattr(train_dataset.height_processor.scaler, 'mean_') or hasattr(train_dataset.height_processor.scaler, 'min_'))
        train_valve_proc_valid = train_dataset.valve_processor and train_dataset.valve_processor.scaler and (hasattr(train_dataset.valve_processor.scaler, 'mean_') or hasattr(train_dataset.valve_processor.scaler, 'min_'))

        if not train_height_proc_valid:
             print("ERROR: Height processor for training data failed to fit properly. Check training data.")
             return
        if not train_valve_proc_valid:
             print("ERROR: Valve processor for training data failed to fit properly. Check training data.")
             return

        # Create test dataset (is_train=False)
        test_dataset = ValveDataset(
            test_data, args.image_dir, transform, seq_len=args.seq_len, is_train=False,
            height_proc_method=args.height_proc, valve_proc_method=args.valve_proc
        )
        # Set the processors for the test set using the ones fitted on the training set
        test_dataset.set_processors(train_dataset.height_processor, train_dataset.valve_processor)
        print("Height and valve processors from training set applied to the test dataset.")

    except Exception as e:
        print(f"Error creating datasets: {e}")
        traceback.print_exc()
        return

    print('Datasets created successfully.')
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Error: One or both datasets are empty after initialization. Check data files and paths.")
        return

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'), drop_last=True) # drop_last=True for stability if last batch is small
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))
    print('DataLoaders created.')


    # --- Initialize Model ---
    print('Initializing model...')
    model = ImageSequenceRegressionModel(
        lstm_hidden_size=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout_rate=args.dropout,
        height_lstm_hidden_size=args.height_lstm_hidden,
        num_attention_heads=args.num_heads # Pass attention heads argument
    )
    print("Model structure:")
    # print(model) # Optional: print full model structure, can be verbose
    # Count parameters (optional)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")


    # --- Load Pretrained Weights if specified ---
    if args.pretrained:
        if os.path.exists(args.pretrained):
            print(f"Loading pretrained weights from: {args.pretrained}")
            try:
                # Load the state dict first
                state_dict = torch.load(args.pretrained, map_location=device)

                # If model is already wrapped in DataParallel, need to remove 'module.' prefix
                # Check if keys start with 'module.'
                if list(state_dict.keys())[0].startswith('module.') and not isinstance(model, nn.DataParallel):
                     print("Removing 'module.' prefix from state dict keys...")
                     state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                elif not list(state_dict.keys())[0].startswith('module.') and isinstance(model, nn.DataParallel):
                     print("Adding 'module.' prefix to state dict keys...")
                     state_dict = {'module.' + k: v for k, v in state_dict.items()}


                # Use strict=False to allow loading weights even if some layers (like attention) are new/missing
                # This is useful for fine-tuning or continuing training with modified architectures.
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"Warning: Missing keys in state_dict: {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
                print("Successfully attempted loading pretrained model state dict.")
            except Exception as e:
                 print(f"Error loading pretrained model from {args.pretrained}: {e}. Training from scratch.")
                 traceback.print_exc() # Print details of the loading error
                 # If eval_only mode requires pretrained, exit
                 if args.eval_only:
                     print("Cannot proceed in --eval_only mode without a valid pretrained model.")
                     return
        else:
            print(f"Warning: Pretrained model file not found: {args.pretrained}.")
            if args.eval_only:
                print("Cannot proceed in --eval_only mode as pretrained model file is missing.")
                return
            else:
                print("Training from scratch.")

    # --- Setup Multi-GPU Training (DataParallel) ---
    # Apply DataParallel *after* loading state_dict if loading weights for the base model
    # If state_dict keys have 'module.', apply DP first then load.
    # Re-initializing model after DP application
    model = setup_data_parallel(model, device)


    # --- Evaluation Only Mode ---
    if args.eval_only:
        print("\n=== Running in Evaluation Only Mode ===")
        if not args.pretrained:
             print("Error: --eval_only requires a --pretrained model path.")
             return
        # Ensure test dataset processors are valid (should be set from train dataset's state)
        if test_dataset.valve_processor is None or test_dataset.valve_processor.scaler is None:
            print("ERROR: Cannot evaluate - test dataset valve processor is not validly set.")
            return
        if test_dataset.height_processor is None or test_dataset.height_processor.scaler is None:
             print("ERROR: Cannot evaluate - test dataset height processor is not validly set.")
             return


        print("Evaluating the loaded model on the test set...")
        start_eval_time = time.time()
        # --- MODIFIED: Pass attention plot filename and processors to evaluate_model ---
        eval_metrics = evaluate_model(
            model,
            test_loader,
            device,
            test_dataset.valve_processor, # Pass valve processor
            test_dataset.height_processor, # Pass height processor
            attention_plot_filename=attention_plot_path # Pass attention plot filename
        )
        end_eval_time = time.time()
        print(f"Evaluation completed in {end_eval_time - start_eval_time:.2f} seconds.")
        print(f"Evaluation Results (Original Scale):")
        print(f"  Prediction Error Std Dev: {eval_metrics.get('std', 'N/A'):.4f}")
        print(f"  R² Coefficient:           {eval_metrics.get('r2', 'N/A'):.4f}")
        print(f"  Pearson Correlation (R):  {eval_metrics.get('pearson_r', 'N/A'):.4f}")
        print(f"  Mean Absolute Error (MAE):{eval_metrics.get('mae', 'N/A'):.4f}")

        if eval_metrics.get('actuals_original') and eval_metrics.get('predictions_original'):
            eval_pred_plot_path = pred_plot_path.replace(".png", "_eval_only.png")
            eval_pred_line_plot_path = pred_line_plot_path.replace(".png", "_eval_only.png")
            plot_predictions(eval_metrics['actuals_original'], eval_metrics['predictions_original'], filename=eval_pred_plot_path)
            plot_prediction_lines(eval_metrics['actuals_original'], eval_metrics['predictions_original'], filename=eval_pred_line_plot_path)
        else:
            print("Cannot generate evaluation prediction plots (missing data).")

        print("Evaluation finished (--eval_only mode). Exiting.")
        return # Exit after evaluation


    # --- Training Setup ---
    criterion = nn.MSELoss() # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Learning rate scheduler: Reduces LR if validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True, min_lr=1e-7)

    # --- Training Loop ---
    print("\n=== Starting Training ===")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    start_train_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # --- Training Phase ---
        # train_epoch does not need attention weights
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        # validate does not need attention weights
        # Ensure test processor is valid before validating
        if test_dataset.valve_processor is None or test_dataset.valve_processor.scaler is None or \
           test_dataset.height_processor is None or test_dataset.height_processor.scaler is None:
            print("ERROR: Cannot validate model - test dataset processors invalid. Skipping validation.")
            avg_val_loss = float('inf') # Assign Inf if validation cannot run
        else:
            avg_val_loss = validate(model, test_loader, criterion, device)
        val_losses.append(avg_val_loss)

        # --- Learning Rate Scheduling ---
        # Step the scheduler based on validation loss
        if avg_val_loss != float('inf'):
             scheduler.step(avg_val_loss)
        else:
             print("Skipping LR scheduler step (invalid validation loss).")

        epoch_end_time = time.time()
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Avg Training Loss: {avg_train_loss:.6f}")
        print(f"  Avg Validation Loss: {avg_val_loss:.6f}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current Learning Rate: {current_lr:.6e}")
        print(f"  Epoch Time: {epoch_end_time - epoch_start_time:.2f} seconds")

        # --- Save Best Model ---
        if avg_val_loss < best_val_loss and avg_val_loss != float('inf'):
            best_val_loss = avg_val_loss
            print(f"✅ New best model found (Val Loss: {best_val_loss:.6f}), saving to {model_save_path}...")
            # Save the underlying model state_dict when using DataParallel
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            try:
                torch.save(model_to_save.state_dict(), model_save_path)
                print(f"Model successfully saved.")
            except Exception as e:
                print(f"Error saving model: {e}")
        # --- Early Stopping (Optional) ---
        # Example: Stop if validation loss doesn't improve for 'patience' * N epochs
        # patience_counter = 0
        # if avg_val_loss >= best_val_loss:
        #     patience_counter += 1
        # else:
        #     patience_counter = 0
        # if patience_counter >= 10: # e.g., stop after 10 epochs of no improvement
        #     print("Early stopping triggered.")
        #     break

    # --- Training Finished ---
    end_train_time = time.time()
    total_training_time = end_train_time - start_train_time
    print(f"\n=== Training Finished ===")
    print(f"Total training time: {total_training_time / 3600:.2f} hours ({total_training_time:.2f} seconds)")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")

    # --- Final Evaluation using Best Model ---
    print("\n=== Final Evaluation using Best Model ===")
    if os.path.exists(model_save_path):
        print(f"Loading best model from {model_save_path}...")
        # Initialize a new model instance with the same architecture
        final_model = ImageSequenceRegressionModel(
             lstm_hidden_size=args.lstm_hidden, lstm_layers=args.lstm_layers,
             dropout_rate=args.dropout, height_lstm_hidden_size=args.height_lstm_hidden,
             num_attention_heads=args.num_heads # Crucial: use same num_heads
        )
        try:
            # Load the saved state dict
            state_dict = torch.load(model_save_path, map_location=device)

            # Handle DataParallel prefix if necessary before loading
            if list(state_dict.keys())[0].startswith('module.') and not isinstance(final_model, nn.DataParallel):
                 print("Removing 'module.' prefix from state dict keys for final evaluation model...")
                 state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            elif not list(state_dict.keys())[0].startswith('module.') and isinstance(final_model, nn.DataParallel):
                 # This case is less likely if model was saved correctly, but included for robustness
                 print("Adding 'module.' prefix to state dict keys for final evaluation model...")
                 state_dict = {'module.' + k: v for k, v in state_dict.items()}


            final_model.load_state_dict(state_dict, strict=True) # Strict=True here to ensure exact match
            # Apply DataParallel if needed
            final_model = setup_data_parallel(final_model, device)
            final_model.eval() # Set to evaluation mode

            # Ensure test dataset processor is valid
            if test_dataset.valve_processor is None or test_dataset.valve_processor.scaler is None or \
               test_dataset.height_processor is None or test_dataset.height_processor.scaler is None:
                 print("ERROR: Cannot perform final evaluation - test dataset processors invalid.")
            else:
                 final_eval_start = time.time()
                 # --- MODIFIED: Pass attention plot filename and processors to final evaluate_model call ---
                 final_metrics = evaluate_model(
                    final_model,
                    test_loader,
                    device,
                    test_dataset.valve_processor, # Pass valve processor
                    test_dataset.height_processor, # Pass height processor
                    attention_plot_filename=attention_plot_path # Pass attention plot filename
                 )
                 final_eval_end = time.time()
                 print(f"Final evaluation completed in {final_eval_end - final_eval_start:.2f} seconds.")
                 print(f"Best Model Evaluation Results (Original Scale) on Test Set:")
                 print(f"  Best Validation Loss (Normalized MSE during training): {best_val_loss:.6f}")
                 print(f"  Prediction Error Std Dev: {final_metrics.get('std', 'N/A'):.4f}")
                 print(f"  R² Coefficient:           {final_metrics.get('r2', 'N/A'):.4f}")
                 print(f"  Pearson Correlation (R):  {final_metrics.get('pearson_r', 'N/A'):.4f}")
                 print(f"  Mean Absolute Error (MAE):{final_metrics.get('mae', 'N/A'):.4f}")

                 # Generate final plots (Scatter and Line plots)
                 print("\nGenerating final prediction visualizations...")
                 if final_metrics.get('actuals_original') and final_metrics.get('predictions_original'):
                      plot_predictions(final_metrics['actuals_original'], final_metrics['predictions_original'], filename=pred_plot_path)
                      plot_prediction_lines(final_metrics['actuals_original'], final_metrics['predictions_original'], filename=pred_line_plot_path)
                 else:
                     print("Cannot generate final prediction plots (missing evaluation data).")

                 # Plot loss curves only if training occurred
                 print("Generating loss curve plot...")
                 if train_losses and val_losses:
                     plot_loss_curves(train_losses, val_losses, filename=loss_plot_path)
                 else:
                     print("Cannot generate loss curves plot (no training data available).")

        except FileNotFoundError:
             print(f"Error: Best model file {model_save_path} not found during final evaluation.")
        except Exception as e:
             print(f"Error loading or evaluating best model: {e}")
             traceback.print_exc()
    else:
        print(f"Best model file ({model_save_path}) not found. Cannot perform final evaluation.")

    print("\nScript execution completed.")


# ===== Program Entry Point =====
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nAn unhandled error occurred in main: {e}")
        traceback.print_exc()
