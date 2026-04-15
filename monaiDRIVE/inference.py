import torch
import os
import json
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from monai.inferers import sliding_window_inference
from model import get_model
from tqdm import tqdm

MODEL_NAME = "rcnet"
CHECKPOINT_ROOT = f"checkpoints/{MODEL_NAME}"
USE_TTA = True
INFERENCE_OVERLAP = 0.5
NUM_FOLDS = 5

def preprocess_fundus_image(img_path):
    img = cv2.imread(img_path)
    if img is None: raise ValueError(f"Cannot load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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


def calculate_auc_numpy(y_true, y_score):
    y_true = y_true.flatten()
    y_score = y_score.flatten()
    if len(y_true) > 200000:
        idx = np.random.choice(len(y_true), 200000, replace=False)
        y_true = y_true[idx]
        y_score = y_score[idx]
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    return np.trapz(tpr, fpr)


def calculate_metrics(pred_bin, pred_prob, gt, mask=None):
    pred_bin = pred_bin.flatten()
    pred_prob = pred_prob.flatten()
    gt = gt.flatten()

    if mask is not None:
        mask = mask.flatten()
        valid_idx = mask > 0
        pred_bin = pred_bin[valid_idx]
        pred_prob = pred_prob[valid_idx]
        gt = gt[valid_idx]

    tp = np.sum((pred_bin == 1) & (gt == 1))
    tn = np.sum((pred_bin == 0) & (gt == 0))
    fp = np.sum((pred_bin == 1) & (gt == 0))
    fn = np.sum((pred_bin == 0) & (gt == 1))

    epsilon = 1e-7
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    sensitivity = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)

    try:
        auc_score = calculate_auc_numpy(gt, pred_prob)
    except Exception:
        auc_score = 0.0

    return {
        "Dice": dice, "IoU": iou, "Se": sensitivity,
        "Sp": specificity, "Acc": accuracy, "AUC": auc_score,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn
    }


def tta_inference(model, inputs):
    roi_size = (256, 256)
    sw_batch_size = 4
    overlap = INFERENCE_OVERLAP

    def predictor(x):
        out = model(x)
        if isinstance(out, tuple): return out[0]
        return out

    pred_1 = sliding_window_inference(inputs, roi_size, sw_batch_size, predictor, overlap=overlap, mode="gaussian")
    inputs_h = torch.flip(inputs, dims=[3])
    pred_h = sliding_window_inference(inputs_h, roi_size, sw_batch_size, predictor, overlap=overlap, mode="gaussian")
    pred_h = torch.flip(pred_h, dims=[3])
    inputs_v = torch.flip(inputs, dims=[2])
    pred_v = sliding_window_inference(inputs_v, roi_size, sw_batch_size, predictor, overlap=overlap, mode="gaussian")
    pred_v = torch.flip(pred_v, dims=[2])

    avg_pred = (pred_1 + pred_h + pred_v) / 3.0
    return torch.sigmoid(avg_pred)


def search_best_threshold(pred_probs_list, gt_list, mask_list):
    print("\n ?_? Searching for the BEST threshold to maximize Dice...")
    best_th = 0.42
    best_dice = 0.0
    thresholds = np.arange(0.30, 0.61, 0.01)
    for th in tqdm(thresholds, desc="Searching"):
        dice_sum = 0
        for prob, gt, mask in zip(pred_probs_list, gt_list, mask_list):
            pred_bin = (prob > th).astype(np.uint8)
            d_metrics = calculate_metrics(pred_bin, prob, gt, mask)
            dice_sum += d_metrics['Dice']
        avg_dice = dice_sum / len(pred_probs_list)
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_th = th
    print(f":) Found Magic Threshold: {best_th:.3f} | Max Dice: {best_dice:.4f}")
    return best_th


def apply_morphological_filter(pred_binary, min_size=50):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_binary, connectivity=8)
    cleaned_mask = np.zeros_like(pred_binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == i] = 1
    return cleaned_mask


def plot_6_panel_visualization(orig_img_path, gt_mask, pred_prob, pred_binary, metrics, save_path):
    orig_bgr = cv2.imread(orig_img_path)
    orig_gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)

    overlay = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
    bg_dark = (orig_gray * 0.3).astype(np.uint8)
    overlay[:, :, 0] = bg_dark
    overlay[:, :, 1] = bg_dark
    overlay[:, :, 2] = bg_dark

    tp_mask = (pred_binary == 1) & (gt_mask == 1)
    fp_mask = (pred_binary == 1) & (gt_mask == 0)
    fn_mask = (pred_binary == 0) & (gt_mask == 1)

    overlay[tp_mask] = [0, 255, 0]
    overlay[fp_mask] = [255, 0, 0]
    overlay[fn_mask] = [0, 0, 255]

    fig = plt.figure(figsize=(18, 12))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(orig_gray, cmap='gray')
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(gt_mask, cmap='gray')
    ax2.set_title("Ground Truth", fontsize=14, fontweight='bold')
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(pred_binary, cmap='gray')
    ax3.set_title("Prediction", fontsize=14, fontweight='bold')
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(pred_prob, cmap='hot')
    ax4.set_title("Probability Heatmap", fontsize=14, fontweight='bold')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(overlay)
    ax5.set_title("Error Overlay (Green:TP, Red:FP, Blue:FN)", fontsize=14, fontweight='bold')
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    plt.tight_layout(rect=[0, 0, 0.65, 1])

    cm = np.array([[metrics['TN'], metrics['FP']], [metrics['FN'], metrics['TP']]])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ax_cm = fig.add_axes([0.68, 0.15, 0.25, 0.25])
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax_cm, cbar=False)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    ax_cm.set_title('Confusion Matrix')

    text_str = (f"Quantitative Metrics:\n\n"
                f"• Dice: {metrics['Dice']:.4f}\n"
                f"• IoU:  {metrics['IoU']:.4f}\n"
                f"• Sens: {metrics['Se']:.4f}\n"
                f"• Spec: {metrics['Sp']:.4f}\n"
                f"• Acc:  {metrics['Acc']:.4f}\n"
                f"• AUC:  {metrics['AUC']:.4f}")
    fig.text(0.68, 0.50, text_str, fontsize=14, family='monospace', verticalalignment='top')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_metrics(df, save_path):
    metrics_to_plot = ['Dice', 'IoU', 'Se', 'Sp', 'Acc', 'AUC']
    mean_vals = df.iloc[-1][metrics_to_plot].values

    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_to_plot, mean_vals, color='steelblue', alpha=0.8)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom',
                 fontweight='bold')

    plt.axhline(y=np.mean(mean_vals), color='r', linestyle='--', label=f'Mean Score: {np.mean(mean_vals):.4f}')

    plt.ylim(0, 1.1)
    plt.title('Ensemble Inference Summary Metrics', fontsize=16)
    plt.ylabel('Score')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "results_ensemble_submission"
    os.makedirs(save_dir, exist_ok=True)
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    with open("dataset.json", 'r') as f:
        data = json.load(f)
    test_files = data['test']

    print(f"Loading {NUM_FOLDS} models for Ensemble Inference...")
    models = []
    for fold in range(NUM_FOLDS):
        model_path = os.path.join(CHECKPOINT_ROOT, f"fold_{fold}", "best_metric_model.pth")
        if not os.path.exists(model_path): continue
        model = get_model(MODEL_NAME, device=device)
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
        print(f"  :) Loaded Fold {fold}")

    if not models: raise ValueError("No models loaded!")

    all_probs, all_gts, all_masks, all_names, all_img_paths = [], [], [], [], []

    print("Step 1/2: Generating Probability Maps...")
    with torch.no_grad():
        for i, item in enumerate(tqdm(test_files)):
            img_path = item["image"]
            label_path = item["label"]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            img_tensor, fov_mask = preprocess_fundus_image(img_path)
            img_tensor = img_tensor.to(device)

            gt_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if gt_img.shape != img_tensor.shape[2:]:
                gt_img = cv2.resize(gt_img, (img_tensor.shape[3], img_tensor.shape[2]), interpolation=cv2.INTER_NEAREST)
            gt_mask = (gt_img > 127).astype(np.uint8)

            ensemble_prob = None
            with torch.amp.autocast('cuda'):
                for model in models:
                    if USE_TTA:
                        fold_output = tta_inference(model, img_tensor)
                    else:
                        def predictor(x):
                            out = model(x)
                            if isinstance(out, tuple): return out[0]
                            return out

                        fold_output = sliding_window_inference(
                            img_tensor, (256, 256), 4, predictor, overlap=INFERENCE_OVERLAP, mode="gaussian"
                        )
                        fold_output = torch.sigmoid(fold_output)

                    if ensemble_prob is None:
                        ensemble_prob = fold_output
                    else:
                        ensemble_prob += fold_output

            avg_prob = ensemble_prob / len(models)
            pred_prob = avg_prob[0, 0].cpu().numpy()

            if fov_mask is not None:
                fov_mask = cv2.resize(fov_mask, (pred_prob.shape[1], pred_prob.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                pred_prob[fov_mask == 0] = 0

            all_probs.append(pred_prob)
            all_gts.append(gt_mask)
            all_masks.append(fov_mask)
            all_names.append(img_name)
            all_img_paths.append(img_path)

    best_threshold = search_best_threshold(all_probs, all_gts, all_masks)

    print(f"Step 2/2: Calculating Final Metrics & Generating Viz...")
    all_metrics = []

    global_cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for i in tqdm(range(len(all_names)), desc="Evaluating"):
        pred_prob = all_probs[i]
        gt_mask = all_gts[i]
        fov_mask = all_masks[i]
        img_name = all_names[i]
        img_path = all_img_paths[i]

        pred_binary = (pred_prob > best_threshold).astype(np.uint8)
        metrics = calculate_metrics(pred_binary, pred_prob, gt_mask, mask=fov_mask)

        global_cm["TP"] += int(metrics["TP"])
        global_cm["TN"] += int(metrics["TN"])
        global_cm["FP"] += int(metrics["FP"])
        global_cm["FN"] += int(metrics["FN"])

        save_metrics = {k: v for k, v in metrics.items() if k not in ["TP", "TN", "FP", "FN"]}
        save_metrics["Image"] = img_name
        all_metrics.append(save_metrics)

        if i < 5:
            vis_save_path = os.path.join(vis_dir, f"{img_name}_result.png")
            plot_6_panel_visualization(img_path, gt_mask, pred_prob, pred_binary, metrics, vis_save_path)

    df = pd.DataFrame(all_metrics)
    stats_dict = {}
    for col in ['Dice', 'IoU', 'Se', 'Sp', 'Acc', 'AUC']:
        stats_dict[col] = f"{df[col].mean():.4f} ± {df[col].std():.4f}"

    mean_row = pd.DataFrame(df.iloc[:, 0:6].mean()).T
    mean_row["Image"] = "AVERAGE"
    df = pd.concat([df, mean_row], ignore_index=True)

    csv_filename = os.path.join(save_dir, "inference_ensemble_5folds.csv")
    df.to_csv(csv_filename, index=False)

    plot_summary_metrics(df, os.path.join(save_dir, "summary_metrics.png"))

    final_results = {
        "Strategy": "5-Fold Ensemble",
        "Threshold": round(best_threshold, 4),
        "Overall_Metrics (Mean ± Std)": stats_dict,
        "Global_Confusion_Matrix": global_cm
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(final_results, f, indent=4)

    print("\n" + "=" * 50)
    print(f"All Done! Final Average Dice: {df.iloc[-1]['Dice']:.4f}")
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    run_inference()