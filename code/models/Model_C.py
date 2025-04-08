#!/usr/bin/env python
# coding: utf-8

import monai
from monai.data import DataLoader, Dataset
from monai.transforms import LoadImaged, Compose, ScaleIntensityd, Resized, EnsureTyped, RandRotated, ConcatItemsd, ToTensord, CenterSpatialCropd
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import from_engine, ValidationHandler, StatsHandler, TensorBoardStatsHandler, CheckpointSaver

import torch

from ignite.metrics import Accuracy

from model_helpers import get_additional_metrics, prepare_batch, Repeatd, load_train_val_data

out_folder = "model_C"

train_data, val_data = load_train_val_data(drop_na=True)
print(f"Training: {len(train_data)}\nValidation: {len(val_data)}")

train_transforms = Compose(
    [
        LoadImaged(keys=["ct","pet"], ensure_channel_first=True),
        ScaleIntensityd(keys=["ct","pet"]),
        Resized(keys=["ct","pet"], spatial_size=(70, 70, 70)),
        CenterSpatialCropd(keys=["ct", "pet"], roi_size = (30, 40, 30)),
        Repeatd(keys=["psa_norm", "px"], target_size=(1, 30, 40, 30)),
        RandRotated(keys=["ct","pet"], prob=0.8, range_x=(-0.2,0.2), range_y=(-0.1,0.1), mode='bilinear'),                                                                                                              
        EnsureTyped(keys=["ct","pet", "psa_norm", "px"]),
        ConcatItemsd(keys=["ct", "pet", "psa_norm", "px"], name="petct", dim=0),
        ToTensord(keys=["petct", "ct", "pet"]),  
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["ct","pet"], ensure_channel_first=True),
        ScaleIntensityd(keys=["ct","pet"]),
        Resized(keys=["ct","pet"], spatial_size=(70, 70, 70)),
        CenterSpatialCropd(keys=["ct", "pet"], roi_size = (30, 40, 30)),
        Repeatd(keys=["psa_norm", "px"], target_size=(1, 30, 40, 30)),
        EnsureTyped(keys=["ct","pet", "psa_norm", "px"]),
        ConcatItemsd(keys=["ct", "pet", "psa_norm", "px"], name="petct", dim=0),
        ToTensord(keys=["petct", "ct", "pet"]),
    ]
) 

batchsize = 16

train_ds = Dataset(data=train_data, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=True, num_workers=1, pin_memory=torch.cuda.is_available())

val_ds = Dataset(data=val_data, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=batchsize, num_workers=1, pin_memory=torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=4, out_channels=2).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)

val_handlers = [
    StatsHandler(name="train_log", output_transform=lambda x: None),
    TensorBoardStatsHandler(log_dir=f"runs/{out_folder}", output_transform=lambda x: None),
    CheckpointSaver(save_dir=f"runs/{out_folder}", save_dict={"net": model}, save_key_metric=True),
]

evaluator = SupervisedEvaluator(
    device = device,
    val_data_loader = val_loader,
    network = model,
    prepare_batch = prepare_batch,
    key_val_metric = {"val_acc": Accuracy(output_transform = from_engine(["pred", "label"]))},
		additional_metrics = get_additional_metrics("val"),
    val_handlers = val_handlers,
    amp = False
)

train_handlers = [
    ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
    StatsHandler(name="train_log", tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
    TensorBoardStatsHandler(log_dir=f"runs/{out_folder}", tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
    CheckpointSaver(save_dir=f"runs/{out_folder}", save_dict={"net": model, "opt": optimizer}, save_interval=1, epoch_level=True),
]

trainer = SupervisedTrainer(
    device = device,
    max_epochs = 15,
    train_data_loader = train_loader,
    network = model,
    optimizer = optimizer,
    loss_function = loss_function,
    prepare_batch = prepare_batch,
    key_train_metric = {"train_acc": Accuracy(output_transform=from_engine(["pred", "label"]))},
		additional_metrics = get_additional_metrics("train"),
    train_handlers = train_handlers,
    amp = False
)

trainer.run()