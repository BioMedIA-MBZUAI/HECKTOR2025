import os
import pandas as pd
import numpy as np
import torch
import csv 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, Resized
)
from monai.networks.nets import resnet18

PATH_TO_TRAINING_IMAGES = "Task_3/Training_Images"
PATH_TO_EHR = "Task_3/HECKTOR_2025_Training_Task_3.csv"

# 1) Dataset -----------------------------------------------------
class HecktorDataset(Dataset):
    def __init__(self, csv_file, img_dirs, transforms, patient_ids=None, scaler=None, ohe=None):
        self.df_full = pd.read_csv(csv_file)
        self.img_dirs = img_dirs
        self.transforms = transforms

        # Filter to specified patients if provided
        if patient_ids is not None:
            self.df = self.df_full[self.df_full["PatientID"].isin(patient_ids)]
        else:
            self.df = self.df_full.copy()

        self.patient_ids = self.df["PatientID"].tolist()
        self.labels = self.df["HPV Status"].astype(int).values

        # Process EHR
        num_cols = ["Age", "Gender"]
        num_data = self.df[num_cols].values
        self.scaler = scaler or StandardScaler()

        cat_cols = ["Tobacco Consumption", "Alcohol Consumption", "Performance Status", "M-stage"]
        df_cat = self.df[cat_cols].astype(str).fillna("Unknown")
        self.ohe = ohe or OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        # Apply transform or fit+transform
        if scaler is None:
            self.num = self.scaler.fit_transform(num_data)
        else:
            self.num = self.scaler.transform(num_data)

        if ohe is None:
            self.cat = self.ohe.fit_transform(df_cat)
        else:
            self.cat = self.ohe.transform(df_cat)

        self.clinical_feats = np.hstack([self.num, self.cat])

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]

        # locate ct/pet
        def find_path(mod):
            for d in self.img_dirs:
                path = os.path.join(d, f"{pid}__{mod}.nii.gz")
                if os.path.exists(path):
                    return path
            raise FileNotFoundError(f"{mod} for {pid} not found")

        ct_path = find_path("CT")
        pet_path = find_path("PT")
        data = self.transforms({"ct": ct_path, "pet": pet_path})
        x_img = torch.cat([data["ct"], data["pet"]], dim=0)
        x_clin = torch.tensor(self.clinical_feats[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x_img, x_clin, y


# 2) Transforms --------------------------------------------------
img_transforms = Compose([
    LoadImaged(keys=["ct","pet"]),
    EnsureChannelFirstd(keys=["ct", "pet"]),
    ScaleIntensityd(keys=["ct","pet"]),    # simple normalization
    Resized(keys=["ct", "pet"], spatial_size=(96, 96, 96)), 
    ToTensord(keys=["ct","pet"]),
])


# 3) Model -------------------------------------------------------
class MultiModalResNet(nn.Module):
    def __init__(self, clin_feat_dim, num_classes=2):
        super().__init__()
        # 4a) 3D ResNet18 backbone for CT+PET
        self.img_model = resnet18(
            spatial_dims=3,
            n_input_channels=2,
            num_classes=2,
        )
        # strip off its FC so we get a 512-d feature
        self.img_model.fc = nn.Identity()

        # 4b) MLP for clinical data
        self.clin_model = nn.Sequential(
            nn.Linear(clin_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 4c) fusion + classification
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_clin):
        f_img = self.img_model(x_img)      # → [B,512]
        f_clin = self.clin_model(x_clin)   # → [B,32]
        f = torch.cat([f_img, f_clin], dim=1)
        return self.classifier(f)


# 5) Training loop -----------------------------------------------

df_all = pd.read_csv("Task_3/HECKTOR_2025_Training_Task_3.csv")

# Step 2: fit scalers on the full available data
scaler = StandardScaler().fit(df_all[["Age", "Gender"]].values)
cat_data = df_all[["Tobacco Consumption", "Alcohol Consumption", "Performance Status", "M-stage"]].astype(str).fillna("Unknown").values
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(cat_data)

def run_crossval(csv_path, img_dirs, scaler, ohe, num_epochs=10, batch_size=4):

    os.makedirs("fold_logs_resnet", exist_ok=True)
    summary_path = "fold_logs_resnet/cv_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Fold", "Best_Val_Acc", "Test_Acc", "Test_F1", "Test_AUC"])

    df_all = pd.read_csv(csv_path)

    all_pids = df_all["PatientID"].values
    all_labels = df_all["HPV Status"].values


    # Hold out test set (20%)
    pids_trainval, pids_test, y_trainval, y_test = train_test_split(
        all_pids, all_labels, test_size=0.2, stratify=all_labels, random_state=42
    )

    # Test set dataloader
    test_ds = HecktorDataset(csv_file=csv_path, img_dirs=img_dirs, transforms=img_transforms, patient_ids=pids_test, scaler=scaler, ohe=ohe)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_val_accuracies = []
    test_auc_scores = []
    test_f1_scores = []
    test_acc = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(pids_trainval, y_trainval)):
        print(f"\nFold {fold + 1}/5")

        p_train = pids_trainval[train_idx]
        p_val   = pids_trainval[val_idx]

        train_ds = HecktorDataset(csv_path, img_dirs, img_transforms, patient_ids=p_train, scaler=scaler, ohe=ohe)
        val_ds   = HecktorDataset(csv_path, img_dirs, img_transforms, patient_ids=p_val, scaler=scaler, ohe=ohe)

        # Save patient IDs for each fold
        with open(f"fold_logs_resnet/fold{fold}_patients.txt", "w") as f:
            f.write("Train:\n" + "\n".join(p_train) + "\n\n")
            f.write("Val:\n" + "\n".join(p_val) + "\n\n")
            f.write("Test:\n" + "\n".join(pids_test) + "\n")

        
        print(f"  Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Init model
        model = MultiModalResNet(clin_feat_dim=train_ds.clinical_feats.shape[1], num_classes=2).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()

        best_acc = 0.0
        best_model_path = f"resnet_best_fold_{fold}.pt"
        val_acc_per_epoch = []

        for epoch in range(1, num_epochs + 1):
            model.train()
            for x_img, x_clin, y in train_loader:
                x_img, x_clin, y = x_img.cuda(), x_clin.cuda(), y.cuda()
                optimizer.zero_grad()
                logits = model(x_img, x_clin)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x_img, x_clin, y in val_loader:
                    x_img, x_clin, y = x_img.cuda(), x_clin.cuda(), y.cuda()
                    preds = model(x_img, x_clin).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            val_acc = correct / total
            print(f"  Epoch {epoch:02d} | Val Acc: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
            val_acc_per_epoch.append(val_acc)

        print(f"Best Val Acc (fold {fold}): {best_acc:.4f}")
        fold_val_accuracies.append(best_acc)
        np.savetxt(f"fold_logs_resnet/fold{fold}_val_acc.txt", val_acc_per_epoch, fmt="%.4f")
        # Evaluate on test set using best model
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        all_preds = []
        all_probs = []
        all_labels = []
        correct = total = 0
        with torch.no_grad():
            for x_img, x_clin, y in test_loader:
                x_img, x_clin, y = x_img.cuda(), x_clin.cuda(), y.cuda()
                logits = model(x_img, x_clin)
                probs = torch.softmax(logits, dim=1)[:, 1]  # probability of class 1
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        acc = correct / total

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        f1 = f1_score(all_labels, all_preds, average="macro")
        auc = roc_auc_score(all_labels, all_probs)

        print(f"Fold {fold} - F1-score: {f1:.4f}, AUC: {auc:.4f}, Accuracy: {acc:.4f}")

        test_auc_scores.append(auc)
        test_f1_scores.append(f1)
        test_acc.append(acc)
        # Log summary row
        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([fold, f"{best_acc:.4f}", f"{acc:.4f}", f"{f1:.4f}", f"{auc:.4f}"])

        print(f"Test Acc (fold {fold} best model): {acc:.4f} (AUC: {auc:.4f}, F1: {f1:.4f})")

    print("\n All fold best val accuracies:", fold_val_accuracies)
    print(f"Avg best val acc: {np.mean(fold_val_accuracies):.4f}")


run_crossval(
    csv_path=PATH_TO_EHR,
    img_dirs=[PATH_TO_TRAINING_IMAGES],
    scaler=scaler,
    ohe=ohe,
    num_epochs=10,
    batch_size=4
)
