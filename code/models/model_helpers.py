import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from monai.handlers import from_engine # type: ignore
from monai.transforms import MapTransform # type: ignore
from ignite.metrics import ConfusionMatrix, ROC_AUC

def load_train_val_data(cropped=True, suv=True, drop_na=False):
    crop = "cropped_" if cropped else ""
    suv = "_suv" if suv else ""
    df = pd.read_csv("data/labels.tsv", sep="\t", dtype={"pseudo_id": str})
    df = df.assign(pet=lambda df: df['pseudo_id'].map(lambda pseudo_id: f"data/{crop}nifti{suv}/{pseudo_id}_pet.nii.gz"))
    df = df.assign(ct=lambda df: df['pseudo_id'].map(lambda pseudo_id: f"data/{crop}nifti/{pseudo_id}_ct.nii.gz"))
    # remove label 2 cases
    df = df[df.label != 2]
    scaler = MinMaxScaler()
    psa_normalized = scaler.fit_transform(df[["psa"]])
    df["psa_norm"] = psa_normalized
    if drop_na:
        df = df.dropna()
    train_data = df[df["set"] == "train"].to_dict('records')
    val_data = df[df["set"] == "val"].to_dict('records')
    return train_data, val_data

def roc_transform(output):
    pred, label = from_engine(["pred","label"])(output)
    pred = [torch.softmax(p, 0)[1] for p in pred]
    return pred, label

def cm_transform(output):
    pred, label = from_engine(["pred", "label"])(output)
    pred = torch.stack(pred).to("cpu")
    label = torch.Tensor(label).to(torch.uint8)
    return pred, label

cm = ConfusionMatrix(2, output_transform=cm_transform)
tn = cm[0,0]
tp = cm[1,1]
fp = cm[0,1]
fn = cm[1,0]
tpr = tp/(tp+fn)
tnr = tn/(tn+fp)
bal_acc = (tpr+tnr)/2

def get_additional_metrics(prefix: str):
    return {
            f"{prefix}_cm": cm,
            f"{prefix}_auc": ROC_AUC(output_transform=roc_transform),
            f"{prefix}_tp": tp,
            f"{prefix}_tn": tn,
            f"{prefix}_fp": fp,
            f"{prefix}_fn": fn,
            f"{prefix}_tpr": tpr,
            f"{prefix}_tnr": tnr,
            f"{prefix}_bal_acc": bal_acc,
}

prepare_batch = lambda batch, device, _non_blocking: (batch["petct"].to(device), batch["label"].to(device))

class Repeatd(MapTransform):

    def __init__(
        self,
        keys,
        target_size,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys = True)
        self.target_size = target_size

    def __call__(self, data):

        d = dict(data)
        for key in d:
            if key in self.keys:
                d[key] = torch.Tensor([d[key]]).repeat(*self.target_size)
        return d