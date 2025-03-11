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

####################################
# CONFIGURATION DICTIONARY
####################################
# For EdgeFace, set model_type to "edgeface" and use color images (grayscale=False).
config = {
    "pretrained_path": "resnet18_110.pth",  # For ResNetFace. Not used for EdgeFace.
    "model_type": "edgeface",               # Options: "resnet18" or "edgeface"
    "edgeface_variant": "edgeface_xxs",     # Only used if model_type is "edgeface"
    "use_se": False,                        # Whether to use Squeeze-Excitation blocks (ResNetFace)
    "grayscale": False,                     # For EdgeFace, use color images
    "image_size": 128,                      # Input image size (width, height)
    "batch_size": 64,                       # Batch size for evaluation (if needed)
    "data_root": "align/lfw-align-128",       # Root directory of processed images
    "pairs_file": "lfw_test_pair.txt",        # File with image pair information
    "embedding_size": 512,                  # Expected embedding dimension from the model
    "nrof_folds": 10,                       # Number of folds for ROC evaluation
    "threshold_range": (0, 3, 0.01),          # (start, stop, step) for threshold sweep
    # WandB configuration
    "wandb_api_key": "bf8d1a3f64bd6397782ed9ec70231089c9deaefa",
    "wandb_project": "LPFMC",
    "wandb_entity": "Cooper-Union"
}

####################################
# 1) MODEL DEFINITION (ResNetFace)
####################################
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class IRBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)
    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.prelu(out)
        return out

class ResNetFace(nn.Module):
    def __init__(self, block, layers, use_se=True, grayscale=True):
        super(ResNetFace, self).__init__()
        self.inplanes = 64
        in_ch = 1 if grayscale else 3
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc5 = nn.Linear(512 * 8 * 8, config["embedding_size"])
        self.bn5 = nn.BatchNorm1d(config["embedding_size"])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=config["use_se"]))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=config["use_se"]))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.bn5(x)
        return {'fea': x}

def resnet_face18(use_se=True, grayscale=True):
    return ResNetFace(IRBlock, [2, 2, 2, 2], use_se=use_se, grayscale=grayscale)

####################################
# 2) MODEL LOADER CLASS
####################################
class FaceModelLoader:
    def __init__(self, config, device):
        self.config = config
        self.device = device

    def load_model(self):
        if self.config["model_type"] == "resnet18":
            model = resnet_face18(use_se=self.config["use_se"], grayscale=self.config["grayscale"]).to(self.device)
            state_dict = torch.load(self.config["pretrained_path"], map_location=self.device)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_key = k.replace("module.", "")
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
            model.eval()
            return model
        elif self.config["model_type"] == "edgeface":
            model = torch.hub.load(repo_or_dir='otroshi/edgeface', model=self.config["edgeface_variant"], source='github', pretrained=True)
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
        if self.config["model_type"] == "edgeface":
            img = cv2.imread(img_path)  # color
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
            inputs = torch.cat([imgA, imgB], dim=0).to(self.device)
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
        gpu_memory = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

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

        # Generate confusion matrix image
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

        # Update summary with numeric metrics
        wandb.run.summary.update({
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "best_threshold": best_thresh,
            "avg_inference_time_ms": avg_inference_time * 1000,
            "total_eval_time_s": total_eval_time,
            "gpu_memory_MB": gpu_memory
        })

        wandb.log({"confusion_matrix_chart": wandb.Image(cm_img, caption="Confusion Matrix")})

        # Log the ROC curve as an interactive plot.
        # If tpr is 2D (from multiple folds), average it:
        if tpr.ndim == 2:
            avg_tpr = np.mean(tpr, axis=0)
            avg_fpr = np.mean(fpr, axis=0)
        else:
            avg_tpr = tpr
            avg_fpr = fpr

        roc_data = [[float(fpr_val), float(tpr_val)] for fpr_val, tpr_val in zip(avg_fpr, avg_tpr)]
        roc_table = wandb.Table(data=roc_data, columns=["FPR", "TPR"])
        wandb.log({"ROC Interactive Plot": wandb.plot.line(roc_table, "FPR", "TPR", title="ROC Curve (Interactive)")})

        return mean_acc, std_acc, best_thresh


if __name__ == "__main__":
    # Ensure every run is logged as a new run
    wandb.login(key=config["wandb_api_key"])
    wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config, reinit=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loader = FaceModelLoader(config, device)
    model = model_loader.load_model()
    evaluator = FaceModelEvaluator(model, config, device)
    evaluator.evaluate()
