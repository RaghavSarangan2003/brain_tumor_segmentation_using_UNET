import torch


def dice_score(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    return (2 * intersection) / (union + 1e-6)


def iou_score(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return intersection / (union + 1e-6)


def pixel_accuracy(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    correct = (preds == targets).float().sum()
    total = targets.numel()
    return correct / total
