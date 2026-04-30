import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd

# PATHS

IMAGE_DIR = "Images"
GT_DIR = "Groundtruth"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# LOAD GROUND TRUTH

def load_ground_truth(mat_path):
    mat = loadmat(mat_path)
    gts = mat['groundTruth']

    combined = None

    for i in range(gts.shape[1]):
        boundary = gts[0][i]['Boundaries'][0][0]

        if combined is None:
            combined = boundary.astype(float)
        else:
            combined += boundary

    combined = combined / gts.shape[1]
    combined = combined > 0.05

    return combined.astype(np.uint8)


# METRICS
def compute_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, ~gt).sum()
    FN = np.logical_and(~pred, gt).sum()
    TN = np.logical_and(~pred, ~gt).sum()

    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    iou = TP / (TP + FP + FN + 1e-6)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    return dice, iou, accuracy, precision, recall

# STORAGE

dice_scores, iou_scores = [], []
accuracies, precisions, recalls = [], [], []


results = []


# MAIN LOOP

for file in os.listdir(IMAGE_DIR):

    if not file.endswith(".jpg"):
        continue

    img_path = os.path.join(IMAGE_DIR, file)
    mat_path = os.path.join(GT_DIR, file.replace(".jpg", ".mat"))

    if not os.path.exists(mat_path):
        print(f"Skipping {file}")
        continue

    # Load image
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load GT
    gt = load_ground_truth(mat_path)
    gt = cv2.resize(gt, (gray.shape[1], gray.shape[0]))

    
    # CANNY + MORPHOLOGY
    
    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.dilate(edges, kernel, iterations=1)

    pred = edges > 0

    gt = cv2.dilate(gt.astype(np.uint8), kernel, iterations=2)
    gt = gt > 0

    
    # METRICS
    
    d, i, a, p, r = compute_metrics(pred, gt)

    dice_scores.append(d)
    iou_scores.append(i)
    accuracies.append(a)
    precisions.append(p)
    recalls.append(r)

    
    results.append([file, d, i, a, p, r])

    
    # VISUALIZATION
    
    overlay = image.copy()
    overlay[pred] = [255, 0, 0]
    overlay[gt] = [0, 255, 0]

    fig, axs = plt.subplots(1, 4, figsize=(15,4))

    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(gt, cmap='gray')
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(pred, cmap='gray')
    axs[2].set_title("Improved Canny")
    axs[2].axis("off")

    axs[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axs[3].set_title("Overlay")
    axs[3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"result_{file}"))
    plt.close()

    print(f"Processed: {file}")


# FINAL METRICS

print("\n===== FINAL RESULTS =====")
print(f"Dice Coefficient : {np.mean(dice_scores):.4f}")
print(f"IoU (Jaccard)   : {np.mean(iou_scores):.4f}")
print(f"Pixel Accuracy  : {np.mean(accuracies):.4f}")
print(f"Precision       : {np.mean(precisions):.4f}")
print(f"Recall          : {np.mean(recalls):.4f}")


# RESULTS TABLE (NEW)

df = pd.DataFrame(results, columns=[
    "Image", "Dice", "IoU", "Accuracy", "Precision", "Recall"
])

#  serial number
df.insert(0, "Sl No.", range(1, len(df)+1))

#  average row
avg = df.mean(numeric_only=True)
avg["Image"] = "Average"
df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)

# Save CSV
df.to_csv("results_table.csv", index=False)

print("\n===== RESULTS TABLE =====")
print(df.to_string(index=False))


# GRAPH

metrics = ['Dice', 'IoU', 'Accuracy', 'Precision', 'Recall']
values = [
    np.mean(dice_scores),
    np.mean(iou_scores),
    np.mean(accuracies),
    np.mean(precisions),
    np.mean(recalls)
]

plt.figure(figsize=(8,5))
plt.bar(metrics, values)
plt.title("Segmentation Performance Metrics")
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.grid()

plt.savefig(os.path.join(OUTPUT_DIR, "final_metrics.png"))
plt.show()