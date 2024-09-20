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
        for images, labels in tqdm(dataloader,desc=f"epoch {epoch+1}"):
            images = images.to(device).float() 
            labels = labels.to(device)

            # optimizer.zero_grad()

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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item()) 

            with open("logs/evaluation_losses.txt", "a") as f:
                f.write(f"epoch {epoch+1} loss == {loss.item()}\n")
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
        with open("logs/evaluation_losses.txt", "a") as f:
            f.write(f"epoch {epoch+1} completed \n total loss == {running_loss/len(dataloader)}\n")

def evaluate_image_classification(blip2_model, blip2_processor, dinov2_model, dinov2_processor, model, head, dataloader, device, num_classes=100, data_name=""):
    model.eval()
    head.eval()
    top1_accuracy = MulticlassAccuracy(num_classes=num_classes, top_k=1).to(device)
    top5_accuracy = MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device)

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            blip2_inputs = blip2_processor(images=images, return_tensors="pt")
            blip2_inputs = {k: v.to(device) for k, v in blip2_inputs.items()}
            blip_image_embeds = blip2_model.get_image_features(**blip2_inputs).last_hidden_state

            dinov2_inputs = dinov2_processor(images=images, return_tensors="pt")
            dinov2_inputs = {k: v.to(device) for k, v in dinov2_inputs.items()}
            dino_image_embeds = dinov2_model(**dinov2_inputs).last_hidden_state

            output, _, _, _, _, _ = model(blip_image_embeds, dino_image_embeds, images)
            outputs = head(output)
            top1_accuracy.update(outputs, labels)
            top5_accuracy.update(outputs, labels)

    return top1_accuracy.compute().item(), top5_accuracy.compute().item()
def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Converts to [0,1]
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cifar100_train = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
 
    cifar100_train_loader = DataLoader(cifar100_train, batch_size=64, shuffle=True)
    cifar100_test_loader = DataLoader(cifar100_test, batch_size=64, shuffle=False)

    d_model = 768
    num_heads = 8
    num_layers = 3
    from hireachial_fusion_mim import HierarchicalAttentionFusion

    model = HierarchicalAttentionFusion(d_model, num_heads, num_layers)
    # model.count_parameters()
    # model = model.to(device)
    model = convert_weights_to_float(model)
    model = model.to(device)
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")

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

    checkpoint_path = "/media/mlr_lab/325C37DE7879ABF2/prarabda/blip2/best_model_r2r.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint)

    cifar_head = ClassificationHead(d_model, 100).to(device)
    total_params = count_parameters(cifar_head)
    print(f"Total trainable parameters: {total_params:,}")

    print("Training CIFAR-100 classification head:")
    train_classification_head(blip2_model, blip2_processor, dinov2_model, dinov2_processor, model, cifar_head, cifar100_train_loader, device, data_name="cifar")

    print("\nEvaluating on CIFAR-100:")
    
    model_top1, model_top5 = evaluate_image_classification(blip2_model, blip2_processor, dinov2_model, dinov2_processor, model, cifar_head, cifar100_test_loader, device, num_classes=100, data_name="cifar")
    print(f"BLIP2 Model - Top-1 Accuracy: {model_top1:.4f}, Top-5 Accuracy: {model_top5:.4f}")

if __name__ == "__main__":
    main()
