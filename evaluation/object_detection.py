# import torch
# import torchvision
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from torchmetrics.classification import MulticlassAccuracy
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# def evaluate_image_classification(model, dataloader, device):
#     model.eval()
#     top1_accuracy = MulticlassAccuracy(num_classes=100, top_k=1).to(device)
#     top5_accuracy = MulticlassAccuracy(num_classes=100, top_k=5).to(device)

#     with torch.no_grad():
#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             top1_accuracy.update(outputs, labels)
#             top5_accuracy.update(outputs, labels)

#     return top1_accuracy.compute().item(), top5_accuracy.compute().item()

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load CIFAR-100 dataset
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
#     cifar100_loader = DataLoader(cifar100_test, batch_size=64, shuffle=False)

#     # Load ImageNet subset from Hugging Face
#     imagenet_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
#     imagenet_dataset = load_dataset("imagenet-1k", split="validation[:1000]")
#     imagenet_dataset = imagenet_dataset.with_transform(lambda examples: {"pixel_values": imagenet_processor(examples["image"]).pixel_values})
#     imagenet_loader = DataLoader(imagenet_dataset, batch_size=64, shuffle=False)

#     # Load base model (e.g., ResNet-50)
#     base_model = torchvision.models.resnet50(pretrained=True).to(device)

#     # Load your custom encoder
#     custom_model = CustomEncoder().to(device)

#     # Evaluate on CIFAR-100
#     print("Evaluating on CIFAR-100:")
#     base_top1, base_top5 = evaluate_image_classification(base_model, cifar100_loader, device)
#     custom_top1, custom_top5 = evaluate_image_classification(custom_model, cifar100_loader, device)

#     print(f"Base Model - Top-1 Accuracy: {base_top1:.4f}, Top-5 Accuracy: {base_top5:.4f}")
#     print(f"Custom Model - Top-1 Accuracy: {custom_top1:.4f}, Top-5 Accuracy: {custom_top5:.4f}")

#     # Evaluate on ImageNet subset
#     print("\nEvaluating on ImageNet subset:")
#     base_top1, base_top5 = evaluate_image_classification(base_model, imagenet_loader, device)
#     custom_top1, custom_top5 = evaluate_image_classification(custom_model, imagenet_loader, device)

#     print(f"Base Model - Top-1 Accuracy: {base_top1:.4f}, Top-5 Accuracy: {base_top5:.4f}")
#     print(f"Custom Model - Top-1 Accuracy: {custom_top1:.4f}, Top-5 Accuracy: {custom_top5:.4f}")

# if __name__ == "__main__":
#     main()

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from transformers import AutoImageProcessor, AutoModelForImageClassification, Blip2Model, AutoProcessor, AutoModel
from datasets import load_dataset
import sys
sys.path.append('/media/mlr_lab/325C37DE7879ABF2/prarabda/blip2/')

# from lavis.models.blip2_models.blip2_qformer_new import Blip2Qformer 
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import math
from datasets import load_dataset
import datasets 

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

def train_classification_head(blip2_model, blip2_processor, dinov2_model, dinov2_processor, model, head, dataloader, device, num_epochs=15, data_name="cifar"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(head.parameters(), lr=0.001)

    model.eval()  # Hierarchical model in eval mode
    blip2_model.eval()  # Hierarchical model in eval mode
    dinov2_model.eval()  # Hierarchical model in eval mode
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
                blip2_inputs = blip2_processor(images=images, return_tensors="pt")
                blip2_inputs = {k: v.to(device) for k, v in blip2_inputs.items()}
                blip_image_embeds = blip2_model.get_image_features(**blip2_inputs).last_hidden_state

                dinov2_inputs = dinov2_processor(images=images, return_tensors="pt")
                dinov2_inputs = {k: v.to(device) for k, v in dinov2_inputs.items()}
                dino_image_embeds = dinov2_model(**dinov2_inputs).last_hidden_state

                output, reconstructed_image, loss_itc, loss_kl, loss_l2, loss_ssim = model(blip_image_embeds, dino_image_embeds, images)

                

            outputs = head(output)
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
            print("Error in saving model")

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
        transforms.ToTensor(),  # Converts to [0,1]
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Now in [-1,1]
    ])

    cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
 
    cifar100_train_loader = DataLoader(cifar100_train, batch_size=64, shuffle=True)
    cifar100_test_loader = DataLoader(cifar100_test, batch_size=64, shuffle=False)

    # imagenet_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    # imagenet_dataset = load_dataset("imagenet-1k", split="train", cache_dir="/media/mlr_lab/325C37DE7879ABF2/prarabda/HF_Datasets")
    # imagenet_dataset = imagenet_dataset.with_transform(lambda examples: {"pixel_values": imagenet_processor(examples["image"]).pixel_values})
    # imagenet_loader = DataLoader(imagenet_dataset, batch_size=64, shuffle=False)

    # blip2_model = Blip2Qformer()
    # /media/mlr_lab/325C37DE7879ABF2/prarabda/blip2/hireachial_fusion_mim.py
    # /media/mlr_lab/325C37DE7879ABF2/prarabda/blip2/evaluation/image_classification.py
    d_model = 768
    num_heads = 8
    num_layers = 3
    from hireachial_fusion_mim import HierarchicalAttentionFusion
    model = HierarchicalAttentionFusion(d_model, num_heads, num_layers)
    # model = model.to(device)
    model = convert_weights_to_float(model)
    model = model.to(device)

    # Initialize BLIP2 model and processor
    blip2_model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip2_model.eval()
    for param in blip2_model.parameters():
        param.requires_grad = False
    blip2_model = blip2_model.to(device)

    blip2_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    # Initialize DINOv2 model and processor
    dinov2_model = AutoModel.from_pretrained('facebook/dinov2-large')
    dinov2_model.eval()
    for param in dinov2_model.parameters():
        param.requires_grad = False
    dinov2_model = dinov2_model.to(device)

    dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')


    # Load checkpoint
    # checkpoint_path = "/media/mlr_lab/325C37DE7879ABF2/prarabda/blip2/lavis/output/BLIP2/Pretrain_stage1/20240907161/checkpoint_9.pth"
    checkpoint_path = "/media/mlr_lab/325C37DE7879ABF2/prarabda/blip2/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint)

    # model_state_dict = blip2_model.state_dict()
    # for name, param in checkpoint['model'].items():
    #     if name in model_state_dict:
    #         try:
    #             model_state_dict[name].copy_(param.float())  # Convert checkpoint weights to float32
    #         except Exception as e:
    #             print(f"Unable to load parameter {name}: {e}")
    #     else:
    #         print(f"Skipping parameter {name} as it's not in the model")
    
    # blip2_model.load_state_dict(model_state_dict)

    # Create and train classification heads
    cifar_head = ClassificationHead(blip2_model.qformer.config.hidden_size, 100).to(device)
    # imagenet_head = ClassificationHead(blip2_model.Qformer.config.hidden_size, 1000).to(device)

    print("Training CIFAR-100 classification head:")
    train_classification_head(blip2_model, blip2_processor, dinov2_model, dinov2_processor, model, cifar_head, cifar100_train_loader, device, data_name="cifar")

    print("\nEvaluating on CIFAR-100:")
    
    blip2_top1, blip2_top5 = evaluate_image_classification(blip2_model, cifar_head, cifar100_test_loader, device, num_classes=100, data_name="cifar")
    print(f"BLIP2 Model - Top-1 Accuracy: {blip2_top1:.4f}, Top-5 Accuracy: {blip2_top5:.4f}")

    #imagenet
    # print("Training ImageNet classification head:")
    # train_classification_head(blip2_model, imagenet_head, imagenet_loader, device, data_name="imagenet")

    # print("\nEvaluating on ImageNet subset:")
    # blip2_top1, blip2_top5 = evaluate_image_classification(blip2_model, imagenet_head, imagenet_loader, device, num_classes=1000, data_name="imagenet")
    # print(f"BLIP2 Model - Top-1 Accuracy: {blip2_top1:.4f}, Top-5 Accuracy: {blip2_top5:.4f}")

if __name__ == "__main__":
    main()
