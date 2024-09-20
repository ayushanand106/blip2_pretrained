import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from transformers import AutoImageProcessor, AutoModelForImageClassification
# from datasets import load_dataset
import sys
# sys.path.append('/media/mlr_lab/325C37DE7879ABF2/prarabda/blip2/')
sys.path.append("/media/anil/ec5448df-0452-49b1-825b-d08ae2473211/Ashu/LAVIS/")
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer 
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import math
from datasets import load_dataset
import datasets
# datasets.config.DOWNLOADED_DATASETS_PATH = "/media/anil/ec5448df-0452-49b1-825b-d08ae2473211/Ashu/hf"

def convert_weights_to_float(model):
    for param in model.parameters():
        param.data = param.data.float()
    return model

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)  
        return self.fc(x)

def train_classification_head(model, head, dataloader, device, num_epochs=15, data_name="cifar"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(head.parameters(), lr=0.001)

    model.eval()  # BLIP2 model in eval mode
    head.train()
    best_loss = 1e9
    if data_name!="cifar":
        num_epochs = 10
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device).float() 
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                image_embeds, _ = model.forward_image(images)

            outputs = head(image_embeds)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        try:
            if running_loss/len(dataloader) < best_loss:
                torch.save({
                    "epoch" : epoch,
                    "head" : head.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    "loss" : loss.item()
                }, f"best_model_{data_name}.pth")
        except:
            print("torch save nahi ho raha")

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

def evaluate_image_classification(model, head, dataloader, device, num_classes=100, data_name=""):
    model.eval()
    head.eval()
    top1_accuracy = MulticlassAccuracy(num_classes=num_classes, top_k=1).to(device)
    top5_accuracy = MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device)

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device).float() 
            labels = labels.to(device)
            image_embeds, _ = model.forward_image(images)
            outputs = head(image_embeds)

            top1_accuracy.update(outputs, labels)
            top5_accuracy.update(outputs, labels)

    return top1_accuracy.compute().item(), top5_accuracy.compute().item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
 
    cifar100_train_loader = DataLoader(cifar100_train, batch_size=128, shuffle=True)
    cifar100_test_loader = DataLoader(cifar100_test, batch_size=128, shuffle=False)

    # imagenet_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    # imagenet_dataset = load_dataset("imagenet-1k", split="train", cache_dir="/media/anil/ec5448df-0452-49b1-825b-d08ae2473211/Ashu/hf")
    # imagenet_dataset = imagenet_dataset.with_transform(lambda examples: {"pixel_values": imagenet_processor(examples["image"]).pixel_values})
    # imagenet_loader = DataLoader(imagenet_dataset, batch_size=128, shuffle=False)

    blip2_model = Blip2Qformer()
    blip2_model = convert_weights_to_float(blip2_model)
    blip2_model = blip2_model.to(device)

    # Load checkpoint
    checkpoint_path = "/media/anil/ec5448df-0452-49b1-825b-d08ae2473211/Ashu/LAVIS/lavis/output/BLIP2/Pretrain_stage1/20240908162/checkpoint_9.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_state_dict = blip2_model.state_dict()
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            try:
                model_state_dict[name].copy_(param.float())  # Convert checkpoint weights to float32
            except Exception as e:
                print(f"Unable to load parameter {name}: {e}")
        else:
            print(f"Skipping parameter {name} as it's not in the model")
    
    blip2_model.load_state_dict(model_state_dict)

    # Create and train classification heads
    cifar_head = ClassificationHead(blip2_model.Qformer.config.hidden_size, 100).to(device)
    # imagenet_head = ClassificationHead(blip2_model.Qformer.config.hidden_size, 1000).to(device)

    print("Training CIFAR-100 classification head:")
    train_classification_head(blip2_model, cifar_head, cifar100_train_loader, device, data_name="cifar")

    print("\nEvaluating on CIFAR-100:")
    
    blip2_top1, blip2_top5 = evaluate_image_classification(blip2_model, cifar_head, cifar100_test_loader, device, num_classes=100, data_name="cifar")
    print(f"BLIP2 Model - Top-1 Accuracy: {blip2_top1:.4f}, Top-5 Accuracy: {blip2_top5:.4f}")

    # #imagenet
    # print("Training ImageNet classification head:")
    # train_classification_head(blip2_model, imagenet_head, imagenet_loader, device, data_name="imagenet")

    # print("\nEvaluating on ImageNet subset:")
    # blip2_top1, blip2_top5 = evaluate_image_classification(blip2_model, imagenet_head, imagenet_loader, device, num_classes=1000, data_name="imagenet")
    # print(f"BLIP2 Model - Top-1 Accuracy: {blip2_top1:.4f}, Top-5 Accuracy: {blip2_top5:.4f}")

if __name__ == "__main__":
    main()

