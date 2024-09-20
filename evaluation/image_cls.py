import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import sys
sys.path.append('/media/mlr_lab/325C37DE7879ABF2/prarabda/blip2/')
from lavis.models.blip2_models.blip2_qformer_new import Blip2Qformer 
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import math

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

def train_classification_head(model, head, dataloader, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(head.parameters(), lr=0.001)

    model.eval()  # BLIP2 model in eval mode
    head.train()

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

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

def evaluate_image_classification(model, head, dataloader, device, num_classes=100):
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
 
    cifar100_train_loader = DataLoader(cifar100_train, batch_size=64, shuffle=True)
    cifar100_test_loader = DataLoader(cifar100_test, batch_size=64, shuffle=False)

    # imagenet_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    # imagenet_dataset = load_dataset("imagenet-1k", split="validation[:1000]")
    # imagenet_dataset = imagenet_dataset.with_transform(lambda examples: {"pixel_values": imagenet_processor(examples["image"]).pixel_values})
    # imagenet_loader = DataLoader(imagenet_dataset, batch_size=64, shuffle=False)

    # base_model = torchvision.models.resnet50(pretrained=True).to(device)

    # Load BLIP2 model
    blip2_model = Blip2Qformer()
    blip2_model = convert_weights_to_float(blip2_model)
    blip2_model = blip2_model.to(device)

    # Load checkpoint
    # checkpoint_path = "/media/mlr_lab/325C37DE7879ABF2/prarabda/blip2/lavis/output/BLIP2/Pretrain_stage1/20240907161/checkpoint_9.pth"
    
    # Step 3: Initialize your custom model and load the checkpoint
    model = MyCustomModel()  # Replace this with your model class
    checkpoint = torch.load('path/to/your/checkpoint.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])  # Adjust the key based on how you saved the checkpoint
    model.eval()  # Set model to evaluation m       



    checkpoint_path = "/media/mlr_lab/325C37DE7879ABF2/prarabda/blip2/best_model.pth"
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

    print("Training CIFAR-100 classification head:")
    train_classification_head(blip2_model, cifar_head, cifar100_train_loader, device)

    print("\nEvaluating on CIFAR-100:")
    # base_top1, base_top5 = evaluate_image_classification(base_model, nn.Identity().to(device), cifar100_test_loader, device, num_classes=100)
    blip2_top1, blip2_top5 = evaluate_image_classification(blip2_model, cifar_head, cifar100_test_loader, device, num_classes=100)

    # print(f"Base Model - Top-1 Accuracy: {base_top1:.4f}, Top-5 Accuracy: {base_top5:.4f}")
    print(f"BLIP2 Model - Top-1 Accuracy: {blip2_top1:.4f}, Top-5 Accuracy: {blip2_top5:.4f}")

if __name__ == "__main__":
    main()