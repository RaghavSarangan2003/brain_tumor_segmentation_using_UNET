import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.unet import UNet
import config



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])



def dice_score(pred, gt, smooth=1e-6):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    return (2. * intersection + smooth) / (union + smooth)


def iou_score(pred, gt, smooth=1e-6):
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return (intersection + smooth) / (union + smooth)



def run_inference_and_analysis():

    # Load model
    model = UNet(in_channels=config.CHANNELS).to(DEVICE)
    model_path = os.path.join(
        config.MODEL_DIR,
        f"{config.EXPERIMENT_NAME}_best_model.pt"
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    #  Output directories for saving
    binary_mask_dir = os.path.join(config.PREDICTION_ROOT, "binary_masks")
    segmented_dir = os.path.join(config.PREDICTION_ROOT, "segmented_images")
    overlay_dir = os.path.join(config.PREDICTION_ROOT, "overlay_images")

    os.makedirs(binary_mask_dir, exist_ok=True)
    os.makedirs(segmented_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # CSV file saving
    csv_path = os.path.join(config.PREDICTION_ROOT, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["Image", "Dice", "IoU"])

    print(f"Saving results to {config.PREDICTION_ROOT}")


    image_files = sorted(os.listdir(config.TEST_IMAGE_DIR))

    for filename in image_files:
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(config.TEST_IMAGE_DIR, filename)
        mask_path = os.path.join(config.TEST_MASK_DIR, filename)  # GT mask

        # Load image and corresponding mask
        image = Image.open(image_path).convert("RGB")
        gt_mask = Image.open(mask_path).convert("L")

        original_size = image.size

        input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

        # Do inference
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        binary_mask = (probs > config.THRESHOLD).astype(np.uint8)

        # Resize masks to original image size
        binary_mask_img = Image.fromarray(binary_mask * 255).resize(
            original_size, Image.NEAREST
        )

        gt_mask = gt_mask.resize(original_size, Image.NEAREST)
        gt_array = (np.array(gt_mask) > 0).astype(np.uint8)

        pred_array = (np.array(binary_mask_img) > 0).astype(np.uint8)

        # Metrics calculation
        dice = dice_score(pred_array, gt_array)
        iou = iou_score(pred_array, gt_array)

        writer.writerow([filename, f"{dice:.4f}", f"{iou:.4f}"])

        # Save the binary mask
        binary_mask_img.save(
            os.path.join(binary_mask_dir, filename)
        )

        # Segmented image extraction
        image_np = np.array(image)
        segmented = image_np * pred_array[:, :, None]
        Image.fromarray(segmented).save(
            os.path.join(segmented_dir, filename)
        )

        # Overlay mask over image
        overlay = image_np.copy()
        overlay[pred_array == 1] = [255, 0, 0]  # red overlay

        blended = (0.7 * image_np + 0.3 * overlay).astype(np.uint8)
        Image.fromarray(blended).save(
            os.path.join(overlay_dir, filename)
        )

        print(
            f"[{filename}] Dice={dice:.4f}, IoU={iou:.4f}"
        )

    csv_file.close()
    print("Inference and analysis completed.")




if __name__ == "__main__":
    run_inference_and_analysis()
