# =============================================================================
# eval_utils.py — Evaluation utilities for SMQ
# Provenance:
# - Mapping utils (create_voting_table, create_correspondences) and plotting utils: adapted from CTE.
#   Repo: https://github.com/Annusha/unsup_temp_embed
# - Metrics (get_labels_start_end_time, levenstein, edit_score, f_score): adapted from MS-TCN.
#   Repo: https://github.com/yabufarha/ms-tcn
# =============================================================================

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

import torch

####################### Mapping utils #######################

def create_voting_table(ground_truth, predicted):
    
    """Filling table with assignment scores.

    Create table which represents paired label assignments, i.e. each
    cell comprises score for corresponding label assignment"""
    
    gt_label2index = dict()
    gt_index2label = dict()
    pr_label2index = dict()
    pr_index2label = dict()

    exclude = dict()

    size = max(len(np.unique(ground_truth)),
                len(np.unique(predicted)))
    
    voting_table = np.zeros((size, size))

    for idx_gt, gt_label in enumerate(np.unique(ground_truth)):
        gt_label2index[gt_label] = idx_gt
        gt_index2label[idx_gt] = gt_label

    if len(gt_label2index) < size:
        for idx_gt in range(len(np.unique(ground_truth)), size):
            gt_label = idx_gt
            while gt_label in gt_label2index:
                gt_label += 1
            gt_label2index[gt_label] = idx_gt
            gt_index2label[idx_gt] = gt_label

    for idx_pr, pr_label in enumerate(np.unique(predicted)):
        pr_label2index[pr_label] = idx_pr
        pr_index2label[idx_pr] = pr_label

    if len(pr_label2index) < size:
        for idx_pr in range(len(np.unique(predicted)), size):
            pr_label = idx_pr
            while pr_label in pr_label2index:
                pr_label += 1
            pr_label2index[pr_label] = idx_pr
            pr_index2label[idx_pr] = pr_label

    for idx_gt, gt_label in enumerate(np.unique(ground_truth)):
        if gt_label in list(exclude.keys()):
            continue
        gt_mask = ground_truth == gt_label
        for idx_pr, pr_label in enumerate(np.unique(predicted)):
            if pr_label in list(exclude.values()):
                continue
            voting_table[idx_gt, idx_pr] = np.sum(predicted[gt_mask] == pr_label, dtype=float)
    for key, val in exclude.items():
        # works only if one pair in exclude
        assert len(exclude) == 1
        try:
            voting_table[gt_label2index[key], pr_label2index[val[0]]] = size * np.max(voting_table)
        except KeyError:
            print('No background!')
            voting_table[gt_label2index[key], -1] = size * np.max(voting_table)
            pr_index2label[size - 1] = val[0]
            pr_label2index[val[0]] = size - 1
    return voting_table, gt_index2label, pr_index2label


def create_correspondences(ground_truth, predicted, method='hungarian', optimization='max', mapping = False):
    
    """ Find output labels which correspond to ground truth labels.

    Hungarian method finds one-to-one mapping: if there is squared matrix
    given, then for each output label -> gt label. If not, some labels will
    be without correspondences.
    Args:
        method: hungarian or max
        optimization: for hungarian method usually min problem but here
            is max, hence convert to min
        where: if some actions are not in the video collection anymore
    """

    gt2cluster = defaultdict(list)
    voting_table, gt_index2label, pr_index2label = create_voting_table(ground_truth=ground_truth, predicted=predicted)

    if method == 'hungarian':
        try:
            assert voting_table.shape[0] == voting_table.shape[1]
        except AssertionError:
            print('voting table non squared')
            raise AssertionError('bum tss')
        if optimization == 'max':
            # convert max problem to minimization problem
            voting_table *= -1
        
        x, y = linear_sum_assignment(voting_table)
        for idx_gt, idx_pr in zip(x, y):
            gt2cluster[gt_index2label[idx_gt]] = [pr_index2label[idx_pr]]
    if method == 'max':
        # maximum voting, won't create exactly one-to-one mapping
        max_responses = np.argmax(voting_table, axis=0)
        for idx, c in enumerate(max_responses):
            # c is index of gt label
            # idx is predicted cluster label
            gt2cluster[gt_index2label[c]].append(idx)
    
    #get the mapped predicted
    pr2gt = {value[0]: key for key, value in sorted(gt2cluster.items(), key=lambda x: x[-1])}
    predicted_mapped = np.vectorize(pr2gt.get)(predicted)
    
    if mapping == False :
        return predicted_mapped
    else :
        return predicted_mapped, pr2gt


########################## Metrics ##########################

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    
    """Converts framewise labels to segment triplets."""
    # bg_class（默认是 "background"），它会自动无视掉背景帧
    # 输入：每一帧的标签数组（例如：[走, 走, 走, 跑, 跑, 背景, 坐, 坐]）。
    # labels: 动作顺序列表（例如：[走, 跑, 坐]）
    # starts / ends: 每个动作段在原始数组中开始和结束的下标。
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    
    """Levenshtein edit distance (optionally normalized).

    Args:
        p (list): Predicted sequence.
        y (list): Ground-truth sequence.
        norm (bool): If True, return 0–100 score.

    Returns:
        float: Distance or normalized score.
    """

    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    
    """Edit score between predicted and GT segments."""
    # P、Y: 动作顺序列表（例如：[走, 跑, 坐]）
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    
    """F1 components using segment IoU threshold."""
    
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
        #判定为 TP (真正例) 的条件（必须同时满足）：1.重合度够高: IoU[idx] >= overlap（即达到了你之前问的那个阈值 $o$）。2.未被占用: not hits[idx]。即这个真实的动作段还没有被之前的预测段匹配过。结果: tp += 1 且标记该真实段已被占用 hits[idx] = 1。
            tp += 1
            hits[idx] = 1
        else:
        # 如果 IoU 太低，或者该动作段已经被匹配过了，则 fp += 1
            fp += 1
    fn = len(y_label) - sum(hits)# 遍历完所有预测段后，那些始终没有被匹配上的真实动作段（即 hits 为 0 的位）就被计为 FN (假负例/漏报)。
    return float(tp), float(fp), float(fn)


#################### Evaluation Pipeline ####################

def read_mapping_file(mapping_file: str):
    """
    Loads mapping.txt that has lines like:
        <index> <action_name>

    Returns:
        dict[int, str]: {index: action_name}
    """
    activities = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            index, activity = line.strip().split(' ', 1)
            activities[int(index)] = activity
    return activities


def get_framewise_predictions(
    model: torch.nn.Module, 
    vid: str, 
    features_path: str, 
    gt_path: str, 
    actions_dict: dict, 
    device: torch.device):
    
    """Runs inference for one sequence and returns GT and raw predictions for segmentation.

    Returns:
        tuple[np.ndarray, np.ndarray]: (gt_array, prediction_array).
    """

    # Load features and gt labels
    features = np.load(os.path.join(features_path, vid))
    gt_array = np.array([actions_dict[action] for action in open(os.path.join(gt_path, vid.replace('npy', 'txt'))).read().splitlines()])
    #eg. gt_array为array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    # Prepare the input
    input_x = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)
    mask = torch.ones(input_x.size(), device=device)
    
    # Feed input to the model and get segmentation predictions
    _ = model(input_x, mask)# 重构后的骨架特征，包含3轴加速度和角速度信息，
    # _ 通常作为一个“丢弃变量”的名词。因为在评估阶段，我们最关心的是模型内部生成的聚类索引（即动作预测）
    prediction_array = model.indices.squeeze(0).cpu().numpy()# 预测的动作标签（聚类索引）数组
 
    return gt_array, prediction_array


def evaluate_predictions(
    gt_array: np.ndarray, 
    prediction_array : np.ndarray, 
    overlap: list[float]):

    """Local (per-sequence) mapping + metrics.

    Returns:
        tuple: (correct, total, edit, tp_vec, fp_vec, fn_vec).
    """
    # 针对每个视频独立进行“局部映射（Local Mapping）”
    # 返回的是“翻译”后的预测序列
    prediction_mapped = create_correspondences(gt_array, prediction_array)
    total = len(gt_array)# 该视频序列的总帧数
    correct = sum(1 for gt, pred in zip(gt_array, prediction_mapped) if gt == pred)# 依次比较GT和预测的每一帧，若相等则返回1，最后求和得到正确预测的帧数
    edit = edit_score(prediction_mapped, gt_array)# 它衡量的是模型是否抓住了“先做什么，后做什么”的逻辑关系。
    # overlaps = [.1, .25, .5]
    tp, fp, fn = np.zeros(len(overlap)), np.zeros(len(overlap)), np.zeros(len(overlap))
    for i, o in enumerate(overlap):
        tp[i], fp[i], fn[i] = f_score(prediction_mapped, gt_array, o)

    return correct, total, edit, tp, fp, fn


def evaluate_local_hungarian(
    model: torch.nn.Module,
    features_path: str,
    gt_path: str,
    mapping_file: str,
    epoch : int, 
    device: torch.device,
    verbose: bool = True):
    
    """
    Runs local (per-video) Hungarian evaluation loop over all videos.

    Returns:
        mof               : float in [0, 100]
        edit_mean         : float (mean edit score over videos)
        f1_vec            : np.ndarray (F1 scores for IoU thresholds [0.10, 0.25, 0.50])
        gt_all            : list of np.ndarray (framewise GT per video)
        prediction_all    : list of np.ndarray (raw framewise predictions per video)
    """
    
    # List of sequences
    list_of_vids = sorted(os.listdir(features_path))# 对文件名进行升序排序，返回一个包含所有文件名的列表
    num_videos = len(list_of_vids)# 文件个数635

    # Cretate actions_dict
    mapping = read_mapping_file(mapping_file)# 建立数字与名称之间的对应关系，返回一个字典，eg.{0: "background", 1: "idle", 2: "walk", 3: "run"}
    actions_dict = {v: k for k, v in mapping.items()}# 变成{"background": 0, "walk": 1}

    # Overlap for F1@0.10 - F1@0.25 - F1@0.50
    overlaps = [.1, .25, .5] 

    total_correct, total_frames, total_edit = 0, 0, 0.0
    total_tp = np.zeros(len(overlaps))# [0.0, 0.0, 0.0]
    total_fp = np.zeros(len(overlaps))
    total_fn = np.zeros(len(overlaps))

    gt_all: List[np.ndarray] = []# 该变量gt_all预期是一个列表List，初始化为空列表，里面的元素预期是一个numpy数组
    prediction_all: List[np.ndarray] = []

    # 加上 tqdm 后，终端会显示一个动态进度条，实时更新当前的百分比、处理速度和预计剩余时间。
    # list_of_vids：包含所有视频文件名的列表，qdm 会自动计算这个列表的长度来确定进度条的总量。
    # desc="Local Hungarian eval"，进度条左侧会显示这段文字，告诉你当前正在执行的是“局部匈牙利评估”任务。
    # unit="sequence"：... sequence/s（每秒处理多少个视频序列）
    for vid in tqdm(list_of_vids, desc="Local Hungarian eval", unit="sequence"):
        # 每次处理一个序列
        # get GT and raw predictions
        gt_array, prediction_array = get_framewise_predictions(model, vid, features_path, gt_path, actions_dict, device)
        gt_all.append(gt_array)
        prediction_all.append(prediction_array)

        # per-sequence Hungarian mapping + metrics
        correct, total, edit, tp, fp, fn = evaluate_predictions(gt_array, prediction_array, overlaps)

        total_correct += correct
        total_frames  += total
        total_edit    += edit
        total_tp      += tp
        total_fp      += fp
        total_fn      += fn

    # MoF and Edit score
    mof = 100.0 * float(total_correct) / total_frames if total_frames else 0.0
    edit_mean = total_edit / num_videos if num_videos else 0.0

    # --- F1 vector (for [.10, .25, .50]) ---
    f1_vec = np.zeros(len(overlaps))
    for i in range(len(overlaps)):
        precision = total_tp[i] / float(total_tp[i] + total_fp[i]) if (total_tp[i] + total_fp[i]) else 0.0
        recall    = total_tp[i] / float(total_tp[i] + total_fn[i]) if (total_tp[i] + total_fn[i]) else 0.0
        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
        f1_vec[i] = np.nan_to_num(f1) * 100.0

    if verbose:
        print_metrics_table("Local", mof, edit_mean, f1_vec, overlaps, epoch)

    return mof, edit_mean, f1_vec, gt_all, prediction_all


def global_mapping(
    gt_all: list[np.ndarray], 
    prediction_all: list[np.ndarray]):
    
    """
    Dataset-level cluster to GT mapping.

    Returns:
        dict: pr2gt mapping.
    """

    gts = np.concatenate(gt_all)
    preds = np.concatenate(prediction_all)
    _ , pr2gt = create_correspondences(gts, preds, mapping=True)

    return pr2gt


def evaluate_with_global_mapping(
    model: torch.nn.Module, 
    list_of_vids: list, 
    features_path: str, 
    gt_path: str, 
    device: torch.device, 
    pr2gt: dict, 
    overlap: list, 
    mapping_file: str, 
    unique_gts: np.ndarray, 
    vis: bool, 
    plot_dir: str):

    """
    Evaluate all sequences using a fixed global cluster-to-ground-truth mapping.

    Returns:
        (tuple): A tuple containing:
            - correct: int (Total number of correctly predicted frames)
            - total: int (Total number of evaluated frames)
            - edit: float (Cumulative edit score across all sequences)
            - tp: np.ndarray (True positive counts for each IoU threshold)
            - fp: np.ndarray (False positive counts for each IoU threshold)
            - fn: np.ndarray (False negative counts for each IoU threshold)
    """

    correct = 0
    total = 0
    edit = 0
    tp, fp, fn = np.zeros(len(overlap)), np.zeros(len(overlap)), np.zeros(len(overlap))

    activities = read_mapping_file(mapping_file)
    actions_dict = {v: k for k, v in activities.items()}

    # mjc
    # --- 新增：如果开启可视化，则准备汇总的文本文件 ---
    summary_txt_path = os.path.join(plot_dir, "all_sequences_results.txt")
    f_summary = None
    if vis:
        f_summary = open(summary_txt_path, 'w', encoding='utf-8')

    for vid in tqdm(list_of_vids, desc="Global Hungarian eval", unit="sequence"):
        gt_array, prediction_array = get_framewise_predictions(model, vid, features_path, gt_path, actions_dict, device)
        prediction_mapped = np.vectorize(pr2gt.get)(prediction_array)

        correct += sum(1 for gt, pred in zip(gt_array, prediction_mapped) if gt == pred)
        total += len(gt_array)
        edit += edit_score(prediction_mapped, gt_array)

        for i, o in enumerate(overlap):
            tp1, fp1, fn1 = f_score(prediction_mapped, gt_array, o)
            tp[i] += tp1
            fp[i] += fp1
            fn[i] += fn1
        
     
        # mjc
        # --- 修改：写入汇总文件 ---
        if vis and f_summary:
            # 将数字索引转换为标签名称
            gt_names = [activities.get(idx, str(idx)) for idx in gt_array]
            pred_names = [activities.get(idx, str(idx)) for idx in prediction_mapped]
            
            f_summary.write(f"{vid}\n")
            f_summary.write(f"GT  {' '.join(gt_names)}\n")
            f_summary.write(f"SMQ {' '.join(pred_names)}\n")
            f_summary.write("-" * 20 + "\n") # 分隔线，方便阅读
            
            plot_path = os.path.join(plot_dir, os.path.basename(vid.replace('.npy', '.png')))
            color_dict = get_color(unique_gts)
            plot_segm(gt_array, prediction_mapped, colors=color_dict, activities = activities, path=plot_path)

    # 别忘了关闭文件
    if f_summary:
        f_summary.close()
        
    return correct, total, edit, tp, fp, fn


def evaluate_global_hungarian(
    model: torch.nn.Module,
    features_path: str,
    gt_path: str,
    device: torch.device,
    mapping_file: str,
    epoch : int,
    vis: bool,
    plot_dir: str,
    gt_all: List[np.ndarray],
    prediction_all: List[np.ndarray],
    verbose: bool = True):
    
    """
    Runs global (dataset-level) Hungarian evaluation with a single call.

    Returns:
        mof               : float in [0, 100]
        edit_mean         : float (mean edit over videos)
        f1_vec            : np.ndarray (F1 scores for IoU thresholds [0.10, 0.25, 0.50])
        pr2gt             : dict mapping predicted cluster id -> GT label
    """
    
    # Build a dataset-level predicted->GT mapping using all sequences (exactly as before)
    pr2gt = global_mapping(gt_all, prediction_all)

    # List of sequences
    list_of_vids = sorted(os.listdir(features_path))

    # Useful for colors/legend; identical to original usage
    unique_gts = np.unique(np.concatenate(gt_all))

    # Overlap for F1@0.10 - F1@0.25 - F1@0.50
    overlaps = [.1, .25, .5] 

    # Evaluate the dataset once using the fixed global mapping
    correct, total, edit, tp, fp, fn = evaluate_with_global_mapping(
        model=model,
        list_of_vids=list_of_vids,
        features_path=features_path,
        gt_path=gt_path,
        device=device,
        pr2gt=pr2gt,
        overlap=overlaps,
        mapping_file=mapping_file,
        unique_gts=unique_gts,
        vis=vis,
        plot_dir=plot_dir,
    )

    # MoF and Edit score
    mof = 100.0 * float(correct) / total if total else 0.0
    edit_mean = float(edit) / len(list_of_vids) if len(list_of_vids) else 0.0

    # --- F1 vector (for [.10, .25, .50]) ---
    f1_vec = np.zeros(len(overlaps))
    for i in range(len(overlaps)):
        precision = tp[i] / float(tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
        recall    = tp[i] / float(tp[i] + fn[i]) if (tp[i] + fn[i]) else 0.0
        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
        f1_vec[i] = np.nan_to_num(f1) * 100.0

    if verbose:
        print_metrics_table("Global", mof, edit_mean, f1_vec, overlaps,epoch)

    return mof, edit_mean, f1_vec, pr2gt

def print_metrics_table(
    title: str, 
    mof: float, 
    edit: float, 
    f1_vec: np.ndarray, 
    overlaps: list[float], 
    epoch: int):
    
    """Print metrics as a table."""
    
    rows = [
        ("MoF",  f"{mof:6.4f}"),
        ("Edit", f"{edit:6.4f}"),
    ]
    for o, f1 in zip(overlaps, f1_vec):
        rows.append((f"F1@{o:0.2f}", f"{f1:6.4f}"))

    epoch_str = f"Epoch {epoch}" if epoch is not None else ""
    name = f"— {epoch_str} — {title} Hungarian Matching Results"

    width_key = max(len(k) for k, _ in rows + [("Metric", "")])
    width_val = max(len(v) for _, v in rows + [("", "Value")])
    bar = "+" + "-"*(width_key+2) + "+" + "-"*(width_val+2) + "+"

    print("\n" + name)
    print(bar)
    print(f"| {'Metric'.ljust(width_key)} | {'Value'.rjust(width_val)} |")
    print(bar)
    for k, v in rows:
        print(f"| {k.ljust(width_key)} | {v.rjust(width_val)} |")
    print(bar + "\n")

####################### Plotting utils #######################

def get_color(unique_gts) :
    color_dict = {}
    cmap = plt.get_cmap('tab20')
    for label_idx, label in enumerate(sorted(unique_gts)):
        if label == -1:
            color_dict[label] = (0, 0, 0)
        else:
            color_dict[label] = cmap(label_idx / len(unique_gts))
    
    return color_dict

def bounds(segm):
    start_label = segm[0]
    start_idx = 0
    idx = 0
    while idx < len(segm):
        try:
            while start_label == segm[idx]:
                idx += 1
        except IndexError:
            yield start_idx, idx, start_label
            break

        yield start_idx, idx, start_label
        start_idx = idx
        start_label = segm[start_idx]

def plot_segm(gt_segm, pred_segm, colors, activities, path='output.png', name='', legend = False):
    fig = plt.figure(figsize=(16, 4))
    plt.axis('off')
    plt.title(name, fontsize=20)

    plots_number = 2  # Ground truth and Prediction

    # Plot Ground Truth
    ax_idx = 1
    ax = fig.add_subplot(plots_number, 1, ax_idx)
    ax.set_ylabel('GT', fontsize=20, rotation=0, labelpad=40, verticalalignment='center')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    v_len = len(gt_segm)
    for start, end, label in bounds(gt_segm):
        ax.axvspan(start / v_len, end / v_len, facecolor=colors[label], alpha=1.0)

    ax.set_xlim(0, end / v_len)
    # Plot Prediction
    ax_idx += 1
    ax = fig.add_subplot(plots_number, 1, ax_idx)
    ax.set_ylabel('SMQ', fontsize=20, rotation=0, labelpad=40, verticalalignment='center')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xlim(0, end / v_len)

    for start, end, label in bounds(pred_segm):
        ax.axvspan(start / v_len, end / v_len, facecolor=colors[label], alpha=1.0)

    if legend :
        # Create Legend for Ground Truth
        plt.subplots_adjust(bottom=0.2)

        gt_labels = set(label for _, _, label in bounds(gt_segm))
        gt_legend_handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in gt_labels]
        gt_legend_labels = [activities[label] for label in gt_labels]
        ax.legend(gt_legend_handles, gt_legend_labels, loc='lower center', bbox_to_anchor=(0.5, -1.2), ncol=len(gt_legend_labels)//2, fontsize=12)

    fig.savefig(path, transparent=False)
    plt.close(fig)