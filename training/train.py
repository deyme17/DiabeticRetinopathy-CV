from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from multiprocessing import freeze_support
from datetime import datetime
import os
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import BaseLineCNN

from .utils import calculate_mean_std
from .early_stopping import EarlyStopping
from .train_pipeline import TrainPipeline
from .config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, 
    ES_PATIANCE, LRS_PATIANCE, LRS_PLATO_FACTOR,
    MANUAL_SEED, TRAIN_VAL_SPLIT, IMAGE_SIZE
)

def start_training() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])
    full_dataset = datasets.ImageFolder("data", transform=base_transform)
    num_classes = len(full_dataset.classes)

    # train/val/test split
    train_size = int(TRAIN_VAL_SPLIT[0] * len(full_dataset))
    val_size = int(TRAIN_VAL_SPLIT[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(MANUAL_SEED)
    )
    mean, std = calculate_mean_std(train_ds, batch_size=BATCH_SIZE)
    imgNorm_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    train_ds.dataset.transform = imgNorm_transform
    val_ds.dataset.transform = imgNorm_transform
    test_ds.dataset.transform = imgNorm_transform

    # dataloaders
    print("Load data...")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

    # handle class disbalance calculated class weights
    targets = [y for _, y in full_dataset.samples]
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(targets), y=targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # model
    model = BaseLineCNN(num_classes)

    # criterion, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        factor=LRS_PLATO_FACTOR, 
                                                        patience=LRS_PATIANCE)
    early_stopping = EarlyStopping(patience=ES_PATIANCE)

    #  main pipeline
    pipeline = TrainPipeline(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=NUM_EPOCHS,
        scheduler=scheduler,
        early_stopping=early_stopping
    )

    #  train
    print("Training starts...")
    best_model, metrics = pipeline.train()

    # save model & metrics
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    now = datetime.now()
    postfix = now.strftime('%d-%m_%H-%M')

    if best_model is not None:
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"saved_models/checkpoint_{postfix}.pth")
    if metrics is not None:
        with open(f"results/metrics_baseline({postfix}).json", "w") as f:
            json.dump(metrics, f, indent=4)

# main section
if __name__=='__main__':
    freeze_support()
    start_training()