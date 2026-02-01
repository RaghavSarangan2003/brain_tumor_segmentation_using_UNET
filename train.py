import os
import glob
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter

from models.unet import UNet
from datasets.brain_dataset import BrainTumorDataset
from losses.segmentation_losses import BCEDiceLoss
from metrics.segmentation_metrics import (
    dice_score,
    iou_score,
    pixel_accuracy
)

import config

# Transforms for images

def get_transforms():
    if config.USE_AUGMENTATION:
        return A.Compose([
            A.Resize(*config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(*config.IMAGE_SIZE),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2(),
        ])


def verify_image_mask_pairs(image_paths, mask_paths):
    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"Mismatch: {len(image_paths)} images found for {len(mask_paths)} masks"
        )

    print(
        f"Dataset integrity check passed"
        f"(All images and masks are present: {len(image_paths)} pairs found)"
    )


def load_image_mask_paths():
    image_paths = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "*")))
    mask_paths = sorted(glob.glob(os.path.join(config.MASK_DIR, "*")))

    verify_image_mask_pairs(image_paths, mask_paths)
    return image_paths, mask_paths




def train():
    torch.manual_seed(config.RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    images, masks = load_image_mask_paths()

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        images,
        masks,
        train_size=config.TRAIN_VAL_SPLIT,
        random_state=config.RANDOM_SEED,
        shuffle=True,
    )

    transform = get_transforms()
    train_dataset = BrainTumorDataset(train_imgs, train_masks, transform)
    val_dataset = BrainTumorDataset(val_imgs, val_masks, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    model = UNet(in_channels=config.CHANNELS).to(device)
    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    writer = SummaryWriter(
        log_dir=str(os.path.join(config.LOG_DIR, config.EXPERIMENT_NAME))
    )

    best_dice = 0.0

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{config.EPOCHS}]")

        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_dice = 0.0

        train_bar = tqdm(
            train_loader,
            desc="Training",
            leave=False
        )

        for imgs, masks in train_bar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += pixel_accuracy(
                logits, masks, threshold=config.THRESHOLD
            )
            train_dice += dice_score(logits, masks, threshold=config.THRESHOLD)

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_dice /= len(train_loader)

        model.eval()
        val_dice = 0.0
        val_iou = 0.0
        val_acc = 0.0

        val_bar = tqdm(
            val_loader,
            desc="Validation",
            leave=False
        )

        with torch.no_grad():
            for imgs, masks in val_bar:
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)

                val_dice += dice_score(
                    logits, masks, threshold=config.THRESHOLD
                )
                val_iou += iou_score(
                    logits, masks, threshold=config.THRESHOLD
                )
                val_acc += pixel_accuracy(
                    logits, masks, threshold=config.THRESHOLD
                )

        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        val_acc /= len(val_loader)


        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("Dice/Train", train_dice, epoch)
        writer.add_scalar("Dice/Val", val_dice, epoch)
        writer.add_scalar("IoU/Val", val_iou, epoch)

        print(
            f"[Epoch {epoch:03d}] "
            f"TrainAcc={train_acc:.4f} | "
            f"TrainDice={train_dice:.4f} | "
            f"ValAcc={val_acc:.4f} | "
            f"ValDice={val_dice:.4f} | "
            f"IoU={val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            best_model_path = os.path.join(
                config.MODEL_DIR,
                f"{config.EXPERIMENT_NAME}_best_model.pt"
            )
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved: {best_model_path}")

    writer.close()
    print("Training complete")


if __name__ == "__main__":
    train()
