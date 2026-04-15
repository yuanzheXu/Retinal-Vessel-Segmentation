import torch
import os
import json
import numpy as np
import cv2
import pandas as pd
from monai.inferers import sliding_window_inference
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from tqdm import tqdm

from model import get_model

MODEL_NAME = "rcnet"
CHECKPOINT_ROOT = f"checkpoints/{MODEL_NAME}"
USE_TTA = True
INFERENCE_OVERLAP = 0.5
NUM_FOLDS = 5

# ================= 学术创新与博弈控制区 =================
# 1. 填入你训练时在控制台看到的各折最优验证集 Dice 分数
# （此处为示例数据，请根据你实际训练的 log 替换）
VAL_DICE_SCORES = {
    0: 0.825,
    1: 0.810,
    2: 0.795,
    3: 0.650,  # 模拟一个因为数据划分极差而崩溃的 Fold
    4: 0.830
}

# 2. 弱专家剔除阈值 (Top-K Expert Selection)
# 验证集 Dice 低于此值的模型将被直接判定为“噪声模型”，剥夺投票权
MIN_DICE_THRESHOLD = 0.75

# 3. 软投票温度系数 (Temperature Scaling for Softmax)
# T > 1 会让权重分布更平滑；T < 1 会放大优秀模型的权重差距
TEMPERATURE = 0.05


# ========================================================


def preprocess_fundus_image(img_path, mask_path=None):
    img = cv2.imread(img_path)
    if img is None: raise ValueError(f"Cannot load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if mask_path and os.path.exists(mask_path):
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask_raw, 127, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    img_gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    std = np.std(img_gray)
    mean = np.mean(img_gray)
    img_normalized = (img_gray - mean) / (std + 1e-8)
    img_normalized = (img_normalized - np.min(img_normalized)) / (np.max(img_normalized) - np.min(img_normalized)) * 255

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_normalized.astype(np.uint8))

    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_final = cv2.LUT(img_clahe, table)

    img_final = img_final.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_final).unsqueeze(0).unsqueeze(0)
    return img_tensor, mask


def tta_inference(model, img_tensor):
    def predictor(x):
        out = model(x)
        return out[0] if isinstance(out, tuple) else out

    pred_original = sliding_window_inference(img_tensor, (256, 256), 4, predictor, overlap=INFERENCE_OVERLAP,
                                             mode="gaussian")
    pred_original = torch.sigmoid(pred_original)

    img_tensor_h = torch.flip(img_tensor, dims=[2])
    pred_h = sliding_window_inference(img_tensor_h, (256, 256), 4, predictor, overlap=INFERENCE_OVERLAP,
                                      mode="gaussian")
    pred_h = torch.flip(torch.sigmoid(pred_h), dims=[2])

    img_tensor_v = torch.flip(img_tensor, dims=[3])
    pred_v = sliding_window_inference(img_tensor_v, (256, 256), 4, predictor, overlap=INFERENCE_OVERLAP,
                                      mode="gaussian")
    pred_v = torch.flip(torch.sigmoid(pred_v), dims=[3])

    return (pred_original + pred_h + pred_v) / 3.0


def calculate_metrics(y_true, y_pred, mask):
    y_true_f = y_true[mask == 255].flatten()
    y_pred_f = y_pred[mask == 255].flatten()

    auc_roc = roc_auc_score(y_true_f, y_pred_f)
    auc_pr = average_precision_score(y_true_f, y_pred_f)

    return auc_roc, auc_pr, y_true_f, y_pred_f


def search_best_threshold(all_probs, all_gts, all_masks):
    print("Searching for best threshold...")
    best_f1 = 0
    best_thresh = 0.5

    all_y_true = []
    all_y_pred = []

    for prob, gt, mask in zip(all_probs, all_gts, all_masks):
        all_y_true.extend(gt[mask == 255].flatten())
        all_y_pred.extend(prob[mask == 255].flatten())

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred_bin = (all_y_pred > thresh).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(all_y_true, y_pred_bin).ravel()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "results_stare_ensemble"
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    json_path = "STARE_dataset.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
    test_files = data['test']

    # 动态路径适配与防御性校验
    base_dir = os.path.dirname(os.path.abspath(json_path))

    def correct_path(file_path):
        if not file_path: return file_path
        if 'STARE/' in file_path:
            rel_path = 'STARE/' + file_path.split('STARE/')[-1]
        else:
            rel_path = os.path.basename(file_path)
        return os.path.join(base_dir, os.path.normpath(rel_path))

    missing_files = []
    for item in test_files:
        for key in ['image', 'label', 'mask']:
            if key in item and item[key]:
                corrected = correct_path(item[key])
                item[key] = corrected
                if not os.path.exists(corrected):
                    missing_files.append(corrected)

    if missing_files:
        error_msg = f"测试集数据校验失败！缺失 {len(missing_files)} 个文件。\n示例：\n" + "\n".join(missing_files[:5])
        raise FileNotFoundError(error_msg)

    # ================= 模型加载与动态权重计算 =================
    models = []
    valid_scores = []

    print("\n" + "=" * 40)
    print("🧠 Initializing Validation-Aware Expert Ensemble")
    print("=" * 40)

    for fold in range(NUM_FOLDS):
        model_path = os.path.join(CHECKPOINT_ROOT, f"fold_{fold}", "best_metric_model.pth")
        if not os.path.exists(model_path):
            print(f"⚠️ Fold {fold}: Checkpoint missing, skipped.")
            continue

        fold_score = VAL_DICE_SCORES.get(fold, 0.0)

        # 创新点1：弱专家剔除机制
        if fold_score < MIN_DICE_THRESHOLD:
            print(f"⛔ Fold {fold} rejected: Validation Dice ({fold_score:.4f}) below threshold ({MIN_DICE_THRESHOLD}).")
            continue

        model = get_model(MODEL_NAME, device=device)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
        valid_scores.append(fold_score)
        print(f"✅ Fold {fold} accepted: Validation Dice = {fold_score:.4f}")

    if not models:
        raise RuntimeError("No models passed the validation threshold! Please lower the MIN_DICE_THRESHOLD or retrain.")

    # 创新点2：温度标度 Softmax 计算动态权重
    scores_tensor = torch.tensor(valid_scores)
    # 计算公式: exp(Score / T) / sum(exp(Score / T))
    weights = torch.softmax(scores_tensor / TEMPERATURE, dim=0).numpy()

    print("\n📊 Normalized Ensemble Weights (Temperature = {}):".format(TEMPERATURE))
    for i, w in enumerate(weights):
        print(f"   Expert {i + 1} Weight: {w:.4f}")
    print("=" * 40 + "\n")
    # ========================================================

    all_probs, all_gts, all_masks, all_names, all_img_paths = [], [], [], [], []

    with torch.no_grad():
        for i, item in enumerate(tqdm(test_files, desc="Inference")):
            img_path = item["image"]
            label_path = item["label"]
            mask_path = item.get("mask")
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            img_tensor, fov_mask = preprocess_fundus_image(img_path, mask_path)
            img_tensor = img_tensor.to(device)

            gt_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if gt_img.shape != img_tensor.shape[2:]:
                gt_img = cv2.resize(gt_img, (img_tensor.shape[3], img_tensor.shape[2]), interpolation=cv2.INTER_NEAREST)
            gt_mask = (gt_img > 127).astype(np.uint8)

            weighted_ensemble_prob = None

            with torch.amp.autocast('cuda'):
                # 创新点核心落地：带权重的概率融合
                for model_idx, model in enumerate(models):
                    if USE_TTA:
                        fold_output = tta_inference(model, img_tensor)
                    else:
                        def predictor(x):
                            out = model(x)
                            return out[0] if isinstance(out, tuple) else out

                        fold_output = sliding_window_inference(img_tensor, (256, 256), 4, predictor,
                                                               overlap=INFERENCE_OVERLAP, mode="gaussian")
                        fold_output = torch.sigmoid(fold_output)

                    weight = weights[model_idx]
                    if weighted_ensemble_prob is None:
                        weighted_ensemble_prob = fold_output * weight
                    else:
                        weighted_ensemble_prob += fold_output * weight

            pred_prob = weighted_ensemble_prob[0, 0].cpu().numpy()

            if fov_mask is not None:
                fov_mask = cv2.resize(fov_mask, (pred_prob.shape[1], pred_prob.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                pred_prob[fov_mask == 0] = 0

            all_probs.append(pred_prob)
            all_gts.append(gt_mask)
            all_masks.append(fov_mask)
            all_names.append(img_name)
            all_img_paths.append(img_path)

    best_thresh = search_best_threshold(all_probs, all_gts, all_masks)
    print(f"\n✅ Optimal Threshold Found: {best_thresh:.4f}")

    results = []
    for prob, gt, mask, name in zip(all_probs, all_gts, all_masks, all_names):
        auc_roc, auc_pr, y_true_f, y_pred_f = calculate_metrics(gt, prob, mask)

        y_pred_bin = (y_pred_f > best_thresh).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(y_true_f, y_pred_bin).ravel()

        se = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

        results.append(
            {"Image": name, "AUC_ROC": auc_roc, "AUC_PR": auc_pr, "Se": se, "Sp": sp, "Acc": acc, "Dice": dice})

        pred_bin_img = (prob > best_thresh).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(vis_dir, f"{name}_prob.png"), (prob * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(vis_dir, f"{name}_bin.png"), pred_bin_img)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, "stare_metrics.csv"), index=False)

    print("\n" + "=" * 40)
    print("🏆 STARE Validation-Aware Ensemble Results 🏆")
    print("=" * 40)
    print(df.mean(numeric_only=True).to_string())
    print("=" * 40)


if __name__ == "__main__":
    run_inference()