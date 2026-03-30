# =============================================================================
# vq.py — SMQ module - patch based quantization
# Adapted from : https://github.com/lucidrains/vector-quantize-pytorch
# =============================================================================

import torch 
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F

from tslearn.clustering import TimeSeriesKMeans


def laplace_smoothing(x, n_categories, eps=1e-5, dim=-1):
    """
    Apply Laplace smoothing to avoid zero counts in categorical statistics.
    """
    denom = x.sum(dim=dim, keepdim=True)
    return (x + eps) / (denom + n_categories * eps)


def kmeans_time_series(samples, num_clusters, num_iters=10, metric='euclidean', random_state=42):
    """
    Initialize codebook using time-series K-Means clustering.
    """
    samples_np = samples.cpu().detach().numpy()
    
    kmeans = TimeSeriesKMeans(
        n_clusters=num_clusters,
        max_iter=num_iters,
        metric=metric,
        random_state=random_state
    )
    kmeans.fit(samples_np)
    
    means = torch.tensor(kmeans.cluster_centers_, device=samples.device)
    labels = torch.tensor(kmeans.labels_, device=samples.device)

    # Count how many samples were assigned to each cluster
    cluster_sizes = torch.bincount(labels, minlength=num_clusters).float().view(-1, 1, 1)

    return means, cluster_sizes


def euclidean_dist(ts1, ts2):
    """
    Compute pairwise Euclidean distances between two sets of temporal patches.
    """
    # ts1:N1,window,embedding_dim
    # unsqueeze(1):在第 1 维（列方向）升维,eg.(3,)->(3,1)
    ts1 = ts1.unsqueeze(1)  # Shape: (N1, 1, window, embedding_dim)
    ts2 = ts2.unsqueeze(0)  # Shape: (1, N2, window, embedding_dim)

    # Frame-wise squared differences
    # diff: (N1, N2, window, embedding_dim)
    diff = ts1 - ts2
    squared_diff = diff ** 2
    
    # Euclidean distance per frame
    sum_squared_diff = torch.sum(squared_diff, dim=-1)
    distances = torch.sqrt(sum_squared_diff)
    
    # Sum over time window to get patch-level distance
    distances = torch.sum(distances, dim=-1)

    return distances


class SkeletonMotionQuantizer(nn.Module):
    """
    Skeleton Motion Quantization (SMQ)

    Quantizes patches into discrete codebook entries,
    using EMA update and optional time series K-Means initialization. 
    Includes dead-code replacement for stability.
    """
    def __init__(self, num_embeddings, embedding_dim, window, commitment_cost, 
                 decay=0.8, eps=1e-5, threshold_ema_dead_code=10, sampling_quantile=0.5,
                replacement_strategy = "representative", kmeans=False, kmeans_metric='euclidean'):

        super(SkeletonMotionQuantizer, self).__init__()

        # Codebook configuration
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._window = window# 即patch_size，时间分块大小。量化时每次处理的连续帧数。60
        self.kmeans_metric = kmeans_metric# K-Means 度量。计算距离的方式：欧氏距离或动态时间规整。
        self.kmeans = kmeans# 是否启用 K-Means。若开启，则用 K-Means 初始化聚类中心（Codebook）。

        # Codebook: (num_embeddings, window, embedding_dim)
        self._embedding = nn.Parameter(torch.zeros(num_embeddings, window, embedding_dim))
        
        # VQ loss weight
        self._commitment_cost = commitment_cost

        # EMA and dead-code handling parameters
        self.decay = decay
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sampling_quantile = sampling_quantile
        self.replacement_strategy = replacement_strategy

        # EMA buffers
        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('cluster_size', torch.zeros(num_embeddings, 1, 1))
        self.register_buffer('embed_avg', self._embedding.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        """
        Initialize the codebook once, using either K-Means or random weights.
        """
        # 模块内部维护了一个 initted 缓冲区
        # 一旦初始化完成，这个标志位就会被设为 True
        # 后续不断调用forward，初始化逻辑也只会运行一次
        if self.initted:
            return

        if self.kmeans:
            # Initialize codebook using K-Means
            embed, cluster_size = kmeans_time_series(data, self._num_embeddings, num_iters=20,
                                                     metric=self.kmeans_metric)
            embed_sum = embed * cluster_size
            self._embedding.data.copy_(embed)
            self.embed_avg.data.copy_(embed_sum)
            self.cluster_size.data.copy_(cluster_size)
        else:
            # Uniform initialization
            embed = torch.empty_like(self._embedding)
            nn.init.kaiming_uniform_(embed)
            self._embedding.data.copy_(embed)
            self.embed_avg.data.copy_(embed)
            self.cluster_size.data.fill_(1)

        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask, random_generator=None):
        """
        Reinitialize dead codes using patches sampled from the current batch.

        Args:
            batch_samples: (N, W, D) patches from current batch (valid patches)
            batch_mask: (K,) bool mask indicating which codes are dead
            random_generator: torch.Generator or None for deterministic sampling
        """
        dead_code_indices = batch_mask.nonzero(as_tuple=False).flatten()
        num_dead_codes = dead_code_indices.numel()
        if num_dead_codes == 0:
            return

        # Total patches we want to sample
        total_samples_needed = num_dead_codes * self.threshold_ema_dead_code

        # Compute distances between batch_samples and embeddings
        distances = -euclidean_dist(batch_samples, self._embedding)
        min_distances, _ = distances.max(dim=1)

        # Compute quantile
        quantile_value = torch.quantile(min_distances, self.sampling_quantile)

        # Choose candidate patches based on replacement strategy to reinitialize dead codes
        if self.replacement_strategy == "representative":
            candidate_indices = (min_distances >= quantile_value).nonzero(as_tuple=False).flatten() 
        
        elif self.replacement_strategy == "exploratory":
            candidate_indices = (min_distances <= quantile_value).nonzero(as_tuple=False).flatten() 
        else:
            raise ValueError(f"Unknown replacement_strategy: {self.replacement_strategy}")

        # If no indices as candidate, just allow any patch as a candidate
        if candidate_indices.numel() == 0:
            candidate_indices = torch.arange(batch_samples.shape[0], device=batch_samples.device)

        # Select indices to sample
        if len(candidate_indices) < total_samples_needed:
            selected_indices = candidate_indices
        
        else:
            permuted_indices = torch.randperm(len(candidate_indices), generator=random_generator)
            selected_indices = candidate_indices[permuted_indices[:total_samples_needed]]

        sampled = batch_samples[selected_indices]  # (total_needed, W, D) or fewer if not enough

        # If there is not enough patches for all dead codes repeat-to-fill
        if sampled.shape[0] < total_samples_needed:    
            print(
                f"[VQ] Warning: Only {sampled.shape[0]} candidate patches available, "
                f"but {total_samples_needed} are needed to replace {num_dead_codes} dead codes. "
                "Repeating samples to fill the requirement.")
            # Repeat cyclically to fill (only happens when candidate pool is tiny)
            repeat_factor = (total_samples_needed + sampled.shape[0] - 1) // sampled.shape[0]
            sampled = sampled.repeat(repeat_factor, 1, 1)[:total_samples_needed]
        
        # Assign sampled patches to each dead code and update buffers
        sampled = sampled.reshape(num_dead_codes, self.threshold_ema_dead_code, *batch_samples.shape[1:])
        sampled_means = sampled.mean(dim=1)

        # Update embeddings and EMA buffers for the dead codes
        for i, code_idx in enumerate(dead_code_indices):
            self._embedding.data[code_idx] = sampled_means[i]
            self.cluster_size.data[code_idx] = self.threshold_ema_dead_code
            self.embed_avg.data[code_idx] = sampled_means[i] * self.threshold_ema_dead_code

    def expire_codes_(self, batch_samples, random_generator=None):
        """
        Detect dead codes from EMA cluster_size and replace them using batch_samples.
        """
        if self.threshold_ema_dead_code == 0:
            return

        # cluster_size: (K,1,1) -> (K,)
        cluster_size_flat = self.cluster_size.squeeze()
        expired_codes = cluster_size_flat < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        self.replace(batch_samples, expired_codes, random_generator=random_generator)

    def forward(self, x, mask):
        """
        Args:
            x:    (B, T, D) float
            mask: (B, T, D) float/bool, 1=valid, 0=pad
        Returns:
            quantize:        (B, T, D)
            encoding_indices:(B, T)      code id per frame (patch id repeated)
            loss:            scalar      commitment loss
            distances:       (B, T, K)   (negative) distances per frame to each code
        """
        B, T, D = x.shape
        W = self._window# 即patch_size，时间分块大小。量化时每次处理的连续帧数。60
        K = self._num_embeddings# 即num_actions，动作类别数。模型最终需要识别的动作总数。hugadb 为 12

        # Pad time dimension if necessary
        remainder = T % W
        # padding_needed 算出还需要补多少帧才能凑齐最后一个完整的窗口
        padding_needed = (W - remainder) if remainder != 0 else 0

        # F.pad：它从张量的最后一个维度开始匹配，参数成对出现。
        # 第一个对 (0, 0)对应最后一个维度D，第二个对 (0, padding_needed)对应倒数第二个维度T，意思是在
        # mode="constant"：指定填充模式为常数填充。value=0：指定填充的具体数值为 0。
        x_pad = F.pad(x, (0, 0, 0, padding_needed), mode="constant", value=0)
        mask_pad = F.pad(mask, (0, 0, 0, padding_needed), mode="constant", value=0)

        _, T_pad, _ = x_pad.shape
        P = T_pad // W  # patches per sequence

        # Patchify: (B, T_pad, D) -> (B*P, W, D)
        x_patches = x_pad.reshape(B * P, W, D)
        mask_patches = mask_pad.reshape(B * P, W, D)

        # valid patch = contains at least one valid element
        # sum(dim=(1,2)):对每个patch的时间和特征两个维度求和，如果求和=0，说明这个块都是padding的结果，如果大于0，说明是有效块
        # 返回的结果为bool张量，形状：(B*P,)，True表示这是有效块
        valid_patch_mask = mask_patches.sum(dim=(1, 2)) > 0          # (B*P,)
        # valid_patches形状为(N_valid,W,D)，其中N_valid是有效块的数量
        valid_patches = x_patches[valid_patch_mask]                  # (N_valid, W, D)

        # Codebook init
        self.init_embed_(valid_patches)

        # Assign codes to patches
        # 计算每个patch与代码本中所有K个向量之间的负欧氏距离
        # 输出的distances_valid形状为(N_valid,K)
        distances_valid = -euclidean_dist(valid_patches, self._embedding)  # (N_valid, K)
        # 取每一个特征向量与最近的码本向量的码本索引
        encoding_indices_valid = torch.argmax(distances_valid, dim=1)      # (N_valid,)
        # 第i个向量的第j列为1/0，1表示该向量与码本中这个索引的向量距离最近第i个向量的第j列为1/0，1表示该向量与码本中这个索引的向量距离最近
        encoding_onehot_valid = F.one_hot(encoding_indices_valid, K).float()  # (N_valid, K)

        # EMA codebook update
        if self.training:
            # sum(dim=0)得到一个K个元素的向量，第j个元素表示第j个码本向量被选中的次数
            # unsqueeze(1)在第1维增加一个维度，形状为(K,1)
            # unsqueeze(-1)在最一维增加一个维度，形状为(K,1,1)
            cluster_size = encoding_onehot_valid.sum(dim=0).unsqueeze(1).unsqueeze(-1)  # (K,1,1)
            # new= old + (1 - decay)x(current-old)
            self.cluster_size.data.lerp_(cluster_size, 1 - self.decay)

            # (N_valid,W,D) (N_valid, K) -> (K,W,D)
            # 对于Codebook中的第l个向量，它会去看当前Batch中哪些样本（索引为 i）被分类到了它这里。
            # 它把所有指向第l个类别的输入特征 valid_patches[i] 全部加在一起
            embed_sum = einsum('ijk,il->ljk', valid_patches, encoding_onehot_valid)     # (K,W,D)
            # embed_avg <- embed_avg+(1-decay)*(embed_sum-embed_avg)
            # cluster_size:(K,1,1) 每个聚类（Code）的大小
            # embed_avg:(K,W,D) 它记录了所有被归类到某个类的输入特征向量的累计总和
            self.embed_avg.data.lerp_(embed_sum, 1 - self.decay)
            
            # laplace_smoothing：防止后续除以0，进行拉普拉斯平滑
            # Laplace_smoothing:算出来的是一个比例（加起来等于1）
            # 对于很火的 Code：它的值几乎不变，依然很大。
            # 对于“僵尸”Code（计数为 0 的）：它的值会从 0 变成一个很小的正数（比如 $0.0001$）。
            cluster_size = laplace_smoothing(self.cluster_size, K) * self.cluster_size.sum(dim=-1, keepdim=True)
            # 每一个code的平均值，即新的聚类中心
            embed_normalized = self.embed_avg / cluster_size
            self._embedding.data.copy_(embed_normalized) # (K, W, D)
            # 创建了一个独立的随机数生成器对象，并为这个生成器设置了一个固定的“种子” 42
            random_generator = torch.Generator()
            random_generator.manual_seed(42)
            
            # 检测死码：如果一个code的特征向量的数量低于门槛，则被标记为死码；
            # 寻找替代品：
            # 1.
            # - representative（代表性）：挑选那些离所有 K 个编码很近的 patch，加强现有特征的表达。  
            # - exploratory（探索性）：挑选那些离所有 K 个编码很远的 patch，去开拓 Codebook 还没覆盖到的新领域。
            # 2.从符合条件的候选 patch 中随机抓取一部分
            # 3.把死码的数值强行改写成选中的 patch 的平均值，给这个死码开了一个“重生点”
            self.expire_codes_(valid_patches, random_generator=random_generator)

        # Quantization
        # encoding_onehot_valid.unsqueeze(-1).unsqueeze(-1)：(N_valid,K,1,1)
        # _embedding：(K, W, D)
        # 维度自动补齐： _embedding 被视为 (1, K, W, D)
        # 结果形状: (N_valid, K, W, D)。
        # 相乘之后：如果一个patch属于第l个code，那么它在第2维中，第l个位置的W*D矩阵即为该码本中心的向量，其他K-1个位置为W*D的零矩阵
        # sum之后：每个patch就被替换成了它所属code的中心向量，结果形状为(N_valid, W, D)
        quantize_valid = torch.sum(
            encoding_onehot_valid.unsqueeze(-1).unsqueeze(-1) * self._embedding,
            dim=1
        )  # (N_valid, W, D)

        # Put quantized patches back into the full patch tensor (invalid -> 0)
        # 创建一个全为 0 的新张量，形状为 (B*P, W, D)，与 x_patches 形状相同
        quantized_patches = torch.zeros_like(x_patches)         # (B*P, W, D)
        # Python 会扫描 valid_patch_mask。每当遇到一个 True（或 1），它就从 quantize_valid 中按顺序取出一个条目，塞进 quantized_patches 对应的那个位置。
        quantized_patches[valid_patch_mask] = quantize_valid
        quantize_pad = quantized_patches.reshape(B, T_pad, D)   # (B, T_pad, D)

        # Crop back to original length T
        end = -padding_needed if padding_needed > 0 else None
        # 去掉之前填充的部分
        x_crop = x_pad[:, :end, :]
        # quantize形状为(B,T,D)
        quantize = quantize_pad[:, :end, :]

        # Commit loss
        loss_mask = mask[:, :x_crop.shape[1], :]
        # detach()：它会创建一个新的张量，这个新张量与原来的张量共享相同的数据内存，但不会被计算图跟踪。这意味着在反向传播过程中，梯度不会流经这个新张量。
        # 梯度只指向 x_crop，Encoder 会受到惩罚并被更新。码本（Embedding）本身在这里是“置身事外”的，不会因为这个 Loss 而改变
        # 如果不加 .detach() 会怎样？反向传播时，梯度会同时传给生成 x_crop 的 Encoder 和 存储 quantize 的 Codebook。这会导致两个东西都在互相靠近。
        diff = quantize.detach() - x_crop

        #sg[.]：代表 detach() 操作
        # 我们不除以总的元素个数B*T*D，而是除以 loss_mask.sum()（即有效元素的个数）
        commit_loss = torch.sum((diff ** 2) * loss_mask) / loss_mask.sum()
        loss = self._commitment_cost * commit_loss # commitment_cost = 1.0

        # Straight-through estimator
        # 这样在反向传播时，codebook不会因为commit_loss被更新
        quantize = x_crop + (quantize - x_crop).detach()

        # Expand patch-level indices/distances to per-frame
        # Indices: (B*P,) -> (B,P) -> (B,T_pad) -> crop
        # 确保张量在同一个计算设备上，且数据类型正确
        patch_indices = torch.zeros(B * P, dtype=encoding_indices_valid.dtype, device=encoding_indices_valid.device)
        patch_indices[valid_patch_mask] = encoding_indices_valid# (N_valid,)，存储的是码本索引
        # 在时间轴（dim=1）上，将每个 Patch 的索引重复W 次。
        # 形状从(B, P)变成(B, PxW)。
        # 这一步是假设该 Patch 内的所有帧都共享同一个码本索引。
        encoding_indices = patch_indices.reshape(B, P).repeat_interleave(W, dim=1)[:, :end]  # (B, T)

        # Distances: (B*P,K) -> (B,P,K) -> (B,T_pad,K) -> crop
        patch_distances = torch.zeros(B * P, K, device=distances_valid.device)
        # patch_distances：每个patch与代码本中所有K个向量之间的负欧氏距离(B*P,K)
        patch_distances[valid_patch_mask] = distances_valid
        distances = patch_distances.reshape(B, P, K).repeat_interleave(W, dim=1)[:, :end, :]  # (B, T, K)

        # 执行 contiguous() 会在内存中重新开辟一块连续的空间，把张量的数据按顺序拷贝过去，并返回一个内存布局紧凑的新张量
        return quantize.contiguous(), encoding_indices, loss, distances
