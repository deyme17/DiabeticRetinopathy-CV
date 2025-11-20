from multiprocessing import freeze_support
from datetime import datetime
import os
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from models import BaseLineCNN

from training.data_processor import DataProcessor
from training.early_stopping import EarlyStopping
from training.train_pipeline import TrainPipeline
from training.config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, 
    ES_PATIANCE, LRS_PATIANCE, LRS_PLATO_FACTOR,
    MANUAL_SEED, TRAIN_VAL_TEST_SPLIT, IMAGE_SIZE
)

def start_training() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load and preprocess data
    processor = DataProcessor(
        data_path="data",
        image_size=IMAGE_SIZE,
        train_val_test_split=TRAIN_VAL_TEST_SPLIT,
        manual_seed=MANUAL_SEED
    )
    train_ds, val_ds, test_ds = processor.process(
        batch_size=BATCH_SIZE,
        use_augmentation=False,
    )
    # dataloaders
    print("Load data...")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

    # model
    model = BaseLineCNN(processor.num_classes)

    # criterion, optimizer, scheduler
    class_weights = processor.compute_class_weights(dataset=train_ds, device=device)
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