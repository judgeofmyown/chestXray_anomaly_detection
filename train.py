from chestXrayData import *
from training_script import *
from dotenv import load_dotenv
from torch import nn, optim
import torchvision
import os

load_dotenv()
train_df = os.getenv("TRAIN_DATA_PATH_csv")

BATCH_SIZE = 2
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
EPOCHS = 4

dist_imgs, df = preprocess_data(train_df)

dataset_01 = chestXDataset(df, dist_imgs, 'train')
train_loader, val_loader, test_loader = createLoaders(df, TRAIN_RATIO, VAL_RATIO, BATCH_SIZE)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=False
)
model.load_state_dict(torch.load("fasterrcnn_weights.pth"))
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

train(
    EPOCHS,
    train_loader,
    model,
    device,
    optimizer
)

torch.save(model.state_dict(), 'trained_fastrcnn_weights.pth')

