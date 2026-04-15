import torch
import os
import torch.backends.cudnn as cudnn
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from tqdm import tqdm
import gc

from data_utils import get_dataloaders
from model import get_model  # 确保你的 model.py 中有这个工厂函数

cudnn.benchmark = True
cudnn.deterministic = False

MODEL_NAME = "rcnet"
MAX_EPOCHS = 800
LOADER_BATCH_SIZE = 1
NUM_SAMPLES = 4
ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
VAL_INTERVAL = 10
PATCH_SIZE = (256, 256)
NUM_FOLDS = 5


def train_one_epoch(model, loader, optimizer, loss_func, scaler, device):
    model.train()
    epoch_loss = 0
    step = 0
    pbar = tqdm(loader, desc=f"Training {MODEL_NAME}", leave=False)

    optimizer.zero_grad()
    for i, batch_data in enumerate(pbar):
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        if inputs.dim() == 5:
            inputs = inputs.view(-1, *inputs.shape[2:])
            labels = labels.view(-1, *labels.shape[2:])

        with torch.amp.autocast('cuda'):
            logits, aux_logits = model(inputs)
            loss = loss_func(logits, labels) + 0.4 * loss_func(aux_logits, labels)

        if torch.isnan(loss):
            print("❌ Warning: Loss is NaN, skipping batch")
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1

        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return epoch_loss / len(loader)


def validate_epoch(model, loader, dice_metric, confusion_metric, device):
    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            with torch.amp.autocast('cuda'):
                def predictor(x):
                    out = model(x)
                    if isinstance(out, tuple): return out[0]
                    return out

                val_outputs = sliding_window_inference(
                    inputs=inputs, roi_size=PATCH_SIZE, sw_batch_size=4,
                    predictor=predictor, overlap=0.5
                )

            pred_discrete = (val_outputs.sigmoid() > 0.5).float()
            dice_metric(y_pred=pred_discrete, y=labels)
            confusion_metric(y_pred=pred_discrete, y=labels)

    dice_score = dice_metric.aggregate().item()
    results = confusion_metric.aggregate()
    return dice_score, results[0].item(), results[1].item()


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(seed=3407)

    for fold in range(NUM_FOLDS):
        print(f"\n{'=' * 20} Fold {fold + 1}/{NUM_FOLDS} {'=' * 20}")
        torch.cuda.empty_cache()
        gc.collect()

        save_dir = os.path.join("checkpoints", MODEL_NAME, f"fold_{fold}")
        os.makedirs(save_dir, exist_ok=True)

        train_loader, val_loader = get_dataloaders(
            "STARE_dataset.json",
            batch_size=LOADER_BATCH_SIZE, num_samples=NUM_SAMPLES,
            fold=fold, num_folds=NUM_FOLDS
        )

        model = get_model(MODEL_NAME, device=device)
        loss_function = DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0, to_onehot_y=False, batch=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda')

        dice_metric = DiceMetric(include_background=False, reduction="mean")
        confusion_metric = ConfusionMatrixMetric(include_background=False, metric_name=["sensitivity", "specificity"],
                                                 reduction="mean")

        best_dice = -1

        try:
            for epoch in range(MAX_EPOCHS):
                train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, scaler, device)
                scheduler.step()

                if (epoch + 1) % VAL_INTERVAL == 0:
                    val_dice, val_se, val_sp = validate_epoch(model, val_loader, dice_metric, confusion_metric, device)
                    dice_metric.reset()
                    confusion_metric.reset()

                    print(
                        f"Fold {fold + 1} | Ep {epoch + 1} | Loss: {train_loss:.4f} | Dice: {val_dice:.4f} | Se: {val_se:.4f}")

                    if val_dice > best_dice:
                        best_dice = val_dice
                        torch.save(model.state_dict(), os.path.join(save_dir, "best_metric_model.pth"))
                        print(f"  >>> ✅ Fold {fold + 1} Best Dice: {best_dice:.4f}")

        except Exception as e:
            print(f"❌ Error in Fold {fold + 1}: {e}")
            torch.save(model.state_dict(), os.path.join(save_dir, "crash_backup.pth"))
        finally:
            del model, optimizer, scheduler, scaler, train_loader, val_loader
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    run_training()