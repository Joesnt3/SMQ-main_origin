from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from src.model.smq import SMQModel
from src.model.utils import distance_joints
from src.model.eval_utils import evaluate_local_hungarian, evaluate_global_hungarian

class Trainer:
    
    """Trains SMQ and evaluates with MoF, Edit and F1 scores."""
    
    def __init__(self, in_channels, filters, num_layers, latent_dim, num_actions, 
                 num_joints, num_person, patch_size, kmeans, kmeans_metric, 
                 sampling_quantile, replacement_strategy, decay):
        """Builds the model and loss.

        Args:
            in_channels: Input feature channels per joint (C).
            filters: Base temporal conv width.
            num_layers: Number of dilated residual layers per stage.
            latent_dim: Latent channels per joint (Z).
            num_actions: Codebook size (K).
            num_joints: Number of joints (V).
            patch_size: Temporal window length for VQ (W).
            kmeans: Whether to initialize codebook with KMeans.
            kmeans_metric: Metric for KMeans init ('euclidean' or 'dtw').
            decay: EMA decay for codebook updates.
        """
        
        # Init model and loss
        self.model = SMQModel(in_channels = in_channels, filters = filters, 
                           num_layers = num_layers, latent_dim = latent_dim, 
                           num_actions = num_actions, num_joints = num_joints, 
                           num_person = num_person, patch_size = patch_size,
                           kmeans = kmeans, kmeans_metric = kmeans_metric, 
                           sampling_quantile = sampling_quantile, 
                           replacement_strategy = replacement_strategy, 
                           decay=decay)
        
        self.mse = nn.MSELoss(reduction='none')
        """
        如果你的模型需要计算“关节距离重构损失”（joint_distance_recons），
        你可能需要先拿到每个坐标轴（X, Y, Z）的独立平方差，再按空间维度进行组合（比如计算欧氏距离），
        而不是直接把所有维度的误差混在一起求平均
        
        设置 none: 返回一个与输入形状（Shape）完全相同的张量（Tensor）
        举例：
        predict = torch.tensor([1.0, 2.0])
        target  = torch.tensor([1.0, 5.0])
        计算过程: [(1-1)^2, (2-5)^2]
        输出: tensor([0., 9.])  <-- 保留了每个位置的误差
        """

    def train(self, save_dir, batch_gen, num_epochs, batch_size, 
              learning_rate, commit_weight, mse_loss_weight, device, 
              joint_distance_recons=True):
        
        # Train mode
        self.model.train()# 这行代码不会执行具体的训练动作，它只是拨动了一个“开关”。告诉模型内部的所有层（Layer）：“现在要开始训练了！”
        self.model.to(device)# 先把模型搬到显卡上去（如果 device 是 cuda）

        num_batches = batch_gen.num_batches(batch_size)# batch的数量， 635个文件，每次8个，向上取整就是80个batch
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)# 采用adam优化器，self.model.parameters()告诉优化器去更新哪些东西
    
        for epoch in range(num_epochs):
        # 一个epoch包含80个batch，每个batch包含8个样本（最后一个可能不足8个）。
            pbar = tqdm(
            total=num_batches,
            desc=f"Training [Epoch {epoch+1}]",
            unit="batch",
            leave=False)

            epoch_rec_loss = 0.0
            epoch_commit = 0.0

            while batch_gen.has_next():# 一个batch有8个序列样本
                # batch_input 形状为(N,C,T_max,V,M)的张量，存的是feature，N是当前batch中的样本数量，一般为8
                # mask 形状同上，里面只有0和1，1表示这里是有效数据，0表示这里是padding的数据
                batch_input, mask = batch_gen.next_batch(batch_size)
                # to() 它负责将存储在内存（CPU）中的数据，移动到显卡（GPU）的显存中
                batch_input, mask = batch_input.to(device), mask.to(device)

                optimizer.zero_grad()# 在loss.backward()时，计算出来的梯度，会被累加到每个参数的grad属性上，会导致梯度数值越来越大，无法收敛。
                
                # Forward pass
                # 实际上，它会自动调用你之前定义的 SMQModel 类中的 forward 函数
                # batch_input形状：(N,C,T_max,V,M)
                reconstructed = self.model(batch_input,mask)

                # Reconstruction in joint-distance space
                if joint_distance_recons:
                    # x, x_hat都是关节两两之间的欧几里得距离矩阵，(V, V) 矩阵中，坐标(i, j)的值表示第i个关节到第j个关节的直线距离。
                    x, x_hat = distance_joints(batch_input), distance_joints(reconstructed)

                # Vanilla Reconstruction
                else :
                    x, x_hat = batch_input, reconstructed
                
                # Calculate loss
                # mse记录x,x_hat它们的每一个距离对的平方差
                # mean这里取全局平均
                rec_loss = mse_loss_weight * torch.mean(self.mse(x, x_hat))
                
                # 在 SMQModel 的 forward 函数中
                # quantized, self.indices, self.commit_loss, _ = self.vq(latent, vq_mask)
                commit_loss = commit_weight * self.model.commit_loss
                loss = rec_loss + commit_loss

                # Backprop and update weights
                # 调用loss.backward()时，pytorch会从最后的loss开始，沿着计算图倒着跑，计算出每一个参数的梯度$\frac{\partial Loss}{\partial w}$。
                loss.backward()
                
                # adam更新参数的公式如下：
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                # v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t)^2
                # m_hat_t = m_t / (1 - beta1^t)
                # v_hat_t = v_t / (1 - beta2^t)
                # w_t = w_{t-1} - learning_rate * m_hat_t / (sqrt(v_hat_t) + epsilon)
                # 其中g_t是当前参数的梯度
                optimizer.step()

                epoch_rec_loss += rec_loss.item()# item():只要数字
                epoch_commit += commit_loss.item()

                pbar.update(1)# 手动更新进度。括号里的 1 代表完成了一个步骤

            batch_gen.reset()
            pbar.close()
            
            # Save Every 5 Epochs
            if (epoch + 1) % 5 == 0 :
                # 这不是保存整个模型对象，而是保存一个 Python 字典，里面记录了所有的权重参数（对于模型）或动量参数（对于优化器）
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), save_dir / f"epoch-{epoch+1}.model")
                torch.save(optimizer.state_dict(), save_dir / f"epoch-{epoch+1}.opt")
            
            # epoch_rec_loss 是这一轮里所有 Batch 的 rec_loss 累加和
            print("[epoch %d]: Reconstruction Loss = %f -- Commit Loss = %f" % 
                  (epoch + 1, epoch_rec_loss / num_batches, 
                   epoch_commit / num_batches))

    def eval(self, model_path, features_path, gt_path, mapping_file,
                epoch, vis , plot_dir, device) :
    
        # Eval mode
        self.model.eval()# 这行代码同样是拨动了一个“开关”，告诉模型内部的所有层（Layer）：“现在要开始评估了！” 

        with torch.no_grad():
            # Load model
            self.model.to(device)# 先把模型搬到显卡上去（如果 device 是 cuda）
            self.model.load_state_dict(torch.load(model_path, map_location=device))# load 从硬盘上读取数据，map_location 把数据加载到哪个设备上，load_state_dict把读取到的信息填到model中

            # --- Sequence Level Evaluation ---
            local_mof, local_edit, local_f1_vec, gt_all, prediction_all = evaluate_local_hungarian(
                model=self.model,
                features_path=features_path,
                gt_path=gt_path,
                mapping_file=mapping_file,
                epoch=epoch,
                device=device,
                verbose=True,
            )

            # --- Dataset Level Evaluation ---
            mof, edit, f1_vec, pr2gt = evaluate_global_hungarian(
                model=self.model,
                features_path=features_path,
                gt_path=gt_path,
                device=device,          
                mapping_file=mapping_file,
                epoch = epoch,
                vis=vis,                      
                plot_dir=plot_dir,
                gt_all=gt_all,
                prediction_all=prediction_all,
                verbose=True,
            )
            