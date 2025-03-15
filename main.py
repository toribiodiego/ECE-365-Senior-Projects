import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
import io
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import wandb
import plotly.express as px  # for interactive plotting

####################################
# CONFIGURATION DICTIONARY
####################################
# For testing the ResNet model via torch.hub.load, set model_type to "resnet18".
# Note: The checkpoint "resnet18_110.pth" is for a grayscale model.
# If you wish to use it as-is, set "grayscale": True.
config = {
    "pretrained_path": "resnet18_110.pth",  # Local path to your ResNet weights
    "model_type": "resnet18",               # Options: "resnet18" or "edgeface"
    "edgeface_variant": "edgeface_xxs_q",   # Not used when model_type is "resnet18"
    "use_se": False,                        # Whether to use Squeeze-Excitation blocks (for ResNetFace)
    "grayscale": False,                     # Use color images for ResNetFace. If set to False, the hubconf will replicate grayscale weights.
    "image_size": 128,                      # Input image size (width, height)
    "batch_size": 64,                       # Batch size for evaluation (if needed)
    "data_root": "align/lfw-align-128",      # Root directory of processed images
    "pairs_file": "lfw_test_pair.txt",       # File with image pair information
    "embedding_size": 512,                  # Expected embedding dimension from the model
    "nrof_folds": 10,                       # Number of folds for ROC evaluation
    "threshold_range": (0, 3, 0.01),         # (start, stop, step) for threshold sweep
    # WandB configuration
    "wandb_api_key": "bf8d1a3f64bd6397782ed9ec70231089c9deaefa",
    "wandb_project": "LPFMC",
    "wandb_entity": "Cooper-Union"
}

####################################
# 2) MODEL LOADER CLASS
####################################
class FaceModelLoader:
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def load_model(self):
        if self.config["model_type"] == "resnet18":
            # Load the model using our local hubconf.py entrypoint
            model = torch.hub.load(
                '.',                         # Local directory (i.e. your current repo)
                'resnet18_face',            # Function name exposed in hubconf.py
                source='local',
                pretrained=True,            # Load local weights
                use_se=self.config["use_se"],
                grayscale=self.config["grayscale"],
                embedding_size=self.config["embedding_size"],
                weights_path=self.config["pretrained_path"]  # e.g. "resnet18_110.pth"
            )
            model = model.to(self.device)
            model.eval()
            return model
        elif self.config["model_type"] == "edgeface":
            model = torch.hub.load(
                repo_or_dir='otroshi/edgeface',
                model=self.config["edgeface_variant"],
                source='github',
                pretrained=True
            )
            # Force quantized models to run on CPU
            if "q" in self.config["edgeface_variant"]:
                model.to("cpu")
            else:
                model.to(self.device)
            model.eval()
            return model
        else:
            raise ValueError("Unsupported model type: {}".format(self.config["model_type"]))

####################################
# 3) IMAGE PROCESSOR CLASS
####################################
class ImageProcessor:
    def __init__(self, config):
        self.config = config

    def process_img(self, img_path):
        # Use color images for edgeface; for ResNetFace, use grayscale if specified.
        if self.config["model_type"] == "edgeface":
            img = cv2.imread(img_path)  # color image
        else:
            img = cv2.imread(img_path, 0) if self.config["grayscale"] else cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image at: {img_path}")
        size = self.config["image_size"]
        img = cv2.resize(img, (size, size))
        if self.config["grayscale"] and self.config["model_type"] != "edgeface":
            img = img.reshape((size, size, 1))
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img = (img - 127.5) / 127.5
        return torch.from_numpy(img).float()

####################################
# 4) EVALUATOR CLASS WITH WANDB LOGGING
####################################
class FaceModelEvaluator:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.img_processor = ImageProcessor(config)
        self.embedding_size = config["embedding_size"]

    def extract_embeddings(self):
        with open(self.config["pairs_file"], 'r') as fd:
            lines = fd.readlines()
        N = len(lines)
        embeddings = np.zeros([2 * N, self.embedding_size], dtype=np.float32)
        issame = []
        idx = 0
        inference_times = []  # store inference time per image pair
        for line in lines:
            line = line.strip()
            splits = line.split()
            if len(splits) < 3:
                print(f"Skipping line: {line}")
                continue
            pathA = os.path.join(self.config["data_root"], splits[0].lstrip('/'))
            pathB = os.path.join(self.config["data_root"], splits[1].lstrip('/'))
            label = float(splits[2])
            imgA = self.img_processor.process_img(pathA).unsqueeze(0)
            imgB = self.img_processor.process_img(pathB).unsqueeze(0)
            # If using a quantized model on CPU, force inputs to CPU.
            device_for_inputs = self.device
            if "q" in self.config["edgeface_variant"]:
                device_for_inputs = torch.device("cpu")
            inputs = torch.cat([imgA, imgB], dim=0).to(device_for_inputs)
            start_time = time.time()
            with torch.no_grad():
                output = self.model(inputs)
                if isinstance(output, dict) and "fea" in output:
                    embeddings_tensor = output["fea"]
                else:
                    embeddings_tensor = output
            inference_time = time.time() - start_time
            inference_times.append(inference_time / 2)  # per image
            out_np = embeddings_tensor.cpu().numpy()
            embeddings[2 * idx] = out_np[0]
            embeddings[2 * idx + 1] = out_np[1]
            issame.append(label)
            idx += 1

        embeddings = embeddings[:2 * idx]
        issame = issame[:idx]
        embeddings = normalize(embeddings)
        avg_inference_time = np.mean(inference_times)
        return embeddings, issame, avg_inference_time

    def calculate_accuracy(self, threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
        tpr = 0 if (tp + fn == 0) else float(tp) / (tp + fn)
        fpr = 0 if (fp + tn == 0) else float(fp) / (fp + tn)
        acc = float(tp + tn) / dist.size
        return tpr, fpr, acc

    def calculate_roc(self, thresholds, embeddings1, embeddings2, actual_issame):
        k_fold = KFold(n_splits=self.config["nrof_folds"], shuffle=False)
        nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
        tprs = np.zeros((self.config["nrof_folds"], len(thresholds)))
        fprs = np.zeros((self.config["nrof_folds"], len(thresholds)))
        accuracy = np.zeros(self.config["nrof_folds"])
        best_thresholds = np.zeros(self.config["nrof_folds"])
        diff = embeddings1 - embeddings2
        dist = np.sum(np.square(diff), axis=1)
        indices = np.arange(nrof_pairs)
        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            acc_train = np.zeros(len(thresholds))
            for t_idx, threshold in enumerate(thresholds):
                _, _, acc_train[t_idx] = self.calculate_accuracy(threshold, dist[train_set], np.asarray(actual_issame)[train_set])
            best_threshold_index = np.argmax(acc_train)
            best_thresholds[fold_idx] = thresholds[best_threshold_index]
            for t_idx, threshold in enumerate(thresholds):
                tprs[fold_idx, t_idx], fprs[fold_idx, t_idx], _ = self.calculate_accuracy(threshold, dist[test_set], np.asarray(actual_issame)[test_set])
            _, _, accuracy[fold_idx] = self.calculate_accuracy(thresholds[best_threshold_index], dist[test_set], np.asarray(actual_issame)[test_set])
        return np.mean(tprs, axis=0), np.mean(fprs, axis=0), accuracy, best_thresholds

    def gen_plot(self, fpr, tpr):
        plt.figure()
        plt.xlabel("FPR", fontsize=14)
        plt.ylabel("TPR", fontsize=14)
        plt.title("ROC Curve", fontsize=14)
        plt.plot(fpr, tpr, linewidth=2)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred:Different", "Pred:Same"],
                    yticklabels=["True:Different", "True:Same"])
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf

    def evaluate(self):
        start_eval = time.time()
        embeddings, issame, avg_inference_time = self.extract_embeddings()
        total_eval_time = time.time() - start_eval

        # If the device is CPU, we skip GPU memory logging.
        if self.device.type == "cuda":
            gpu_memory = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
        else:
            gpu_memory = 0.0

        thresholds = np.arange(*self.config["threshold_range"])
        embeddings1 = embeddings[0::2]
        embeddings2 = embeddings[1::2]
        tpr, fpr, accuracy, best_thresholds = self.calculate_roc(thresholds, embeddings1, embeddings2, issame)
        mean_acc = accuracy.mean()
        std_acc = accuracy.std()
        best_thresh = best_thresholds.mean()

        print(f"[Evaluation] #pairs = {len(issame)}")
        print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Best threshold: {best_thresh:.4f}")
        print(f"Avg inference time per image: {avg_inference_time*1000:.2f} ms")
        print(f"Total evaluation time: {total_eval_time:.2f} s")
        print(f"Max GPU memory allocated: {gpu_memory:.2f} MB")

        # Generate confusion matrix image.
        cm_buf = self.plot_confusion_matrix(confusion_matrix(
            np.array(issame, dtype=bool),
            np.sum(np.square(embeddings1 - embeddings2), axis=1) < best_thresh))
        cm_img = Image.open(cm_buf)

        metrics_str = (
            f"Accuracy Mean: {mean_acc:.4f}\n"
            f"Accuracy Std: {std_acc:.4f}\n"
            f"Best Threshold: {best_thresh:.4f}\n"
            f"Avg Inference Time (ms): {avg_inference_time*1000:.2f}\n"
            f"Total Evaluation Time (s): {total_eval_time:.2f}\n"
            f"GPU Memory (MB): {gpu_memory:.2f}\n"
        )
        with open("evaluation_metrics.txt", "w") as f:
            f.write(metrics_str)
        wandb.save("evaluation_metrics.txt")

        wandb.run.summary.update({
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "best_threshold": best_thresh,
            "avg_inference_time_ms": avg_inference_time * 1000,
            "total_eval_time_s": total_eval_time,
            "gpu_memory_MB": gpu_memory
        })

        wandb.log({"confusion_matrix_chart": wandb.Image(cm_img, caption="Confusion Matrix")})

        # Log the ROC curve as an interactive Plotly plot.
        if tpr.ndim == 2:
            avg_tpr = np.mean(tpr, axis=0)
            avg_fpr = np.mean(fpr, axis=0)
        else:
            avg_tpr = tpr
            avg_fpr = fpr

        fig = px.line(x=avg_fpr, y=avg_tpr, labels={'x': 'FPR', 'y': 'TPR'},
                      title="ROC Curve (Interactive)")
        wandb.log({"ROC Interactive Plot": fig})

        # Log evaluation metrics table.
        metrics_table = wandb.Table(columns=["Metric", "Value"])
        metrics_table.add_data("Accuracy Mean", mean_acc)
        metrics_table.add_data("Accuracy Std", std_acc)
        metrics_table.add_data("Best Threshold", best_thresh)
        metrics_table.add_data("Avg Inference Time (ms)", avg_inference_time * 1000)
        metrics_table.add_data("Total Evaluation Time (s)", total_eval_time)
        metrics_table.add_data("GPU Memory (MB)", gpu_memory)
        wandb.log({"Evaluation Metrics Table": metrics_table})

        return mean_acc, std_acc, best_thresh

if __name__ == "__main__":
    # Log in to Weights & Biases.
    wandb.login(key=config["wandb_api_key"])
    # If using a quantized model for EdgeFace, force CPU; otherwise, use available device.
    if "q" in config["edgeface_variant"]:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config, reinit=True)

    model_loader = FaceModelLoader(config, device)
    model = model_loader.load_model()
    evaluator = FaceModelEvaluator(model, config, device)
    evaluator.evaluate()
