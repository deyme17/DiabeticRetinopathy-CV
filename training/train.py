from datetime import datetime
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import BaseLineCNN

from .early_stopping import EarlyStopping
from .train_pipeline import TrainPipeline
from .config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, 
    ES_PATIANCE, LRS_PATIANCE, LRS_PLATO_FACTOR,
    MANUAL_SEED, TRAIN_VAL_SPLIT, IMAGE_SIZE
)

# transform
img_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])
# dataset
full_dataset = datasets.ImageFolder("data", transform=img_transform)
num_classes = len(full_dataset.classes)

# train/val/test split
train_size = int(TRAIN_VAL_SPLIT[0] * len(full_dataset))
val_size = int(TRAIN_VAL_SPLIT[1] * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_ds, val_ds, test_ds = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(MANUAL_SEED)
)

# apply val/test transforms
val_ds.dataset.transform = img_transform
test_ds.dataset.transform = img_transform

# dataloaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# model
model = BaseLineCNN(num_classes)

# criterion, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
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
best_model, metrics = pipeline.train()

# save model & metrics
if best_model is not None:
    now = datetime.now()
    postfix = now.strftime('%d-%m_%H-%M')
    torch.save(
        best_model.state_dict(), 
        f"saved_models/best_model_{postfix}.pth"
    )
if metrics is not None:
    with open(f"results/metrics_{postfix}.json", "w") as f:
        json.dump(metrics, f, indent=4)