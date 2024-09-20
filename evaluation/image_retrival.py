import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            batch_features = model(images)
            features.append(batch_features.cpu().numpy())
            labels.extend(batch_labels.numpy())

    return np.concatenate(features), np.array(labels)

def compute_recall_at_k(query_features, gallery_features, query_labels, gallery_labels, k):
    similarity = cosine_similarity(query_features, gallery_features)
    
    correct = 0
    total = len(query_labels)

    for i, query_label in enumerate(query_labels):
        relevant_indices = np.where(gallery_labels == query_label)[0]
        top_k_indices = np.argsort(similarity[i])[-k:][::-1]
        if np.intersect1d(relevant_indices, top_k_indices).size > 0:
            correct += 1

    return correct / total

def compute_mAP(query_features, gallery_features, query_labels, gallery_labels):
    similarity = cosine_similarity(query_features, gallery_features)
    
    mAP = 0
    for i, query_label in enumerate(query_labels):
        relevant_indices = np.where(gallery_labels == query_label)[0]
        sorted_indices = np.argsort(similarity[i])[::-1]
        
        precision_sum = 0
        num_relevant = 0
        
        for j, index in enumerate(sorted_indices):
            if index in relevant_indices:
                num_relevant += 1
                precision_sum += num_relevant / (j + 1)
        
        if len(relevant_indices) > 0:
            mAP += precision_sum / len(relevant_indices)
    
    return mAP / len(query_labels)

def evaluate_image_retrieval(model, query_loader, gallery_loader, device):
    query_features, query_labels = extract_features(model, query_loader, device)
    gallery_features, gallery_labels = extract_features(model, gallery_loader, device)

    recall_at_1 = compute_recall_at_k(query_features, gallery_features, query_labels, gallery_labels, k=1)
    recall_at_5 = compute_recall_at_k(query_features, gallery_features, query_labels, gallery_labels, k=5)
    mAP = compute_mAP(query_features, gallery_features, query_labels, gallery_labels)

    return recall_at_1, recall_at_5, mAP

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load a subset of ImageNet for image retrieval
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    imagenet_dataset = torchvision.datasets.ImageNet(root='./data', split='val', transform=transform)
    
    # Use a subset of ImageNet for demonstration purposes
    subset_indices = torch.randperm(len(imagenet_dataset))[:10000]
    query_indices = subset_indices[:1000]
    gallery_indices = subset_indices[1000:]

    query_dataset = torch.utils.data.Subset(imagenet_dataset, query_indices)
    gallery_dataset = torch.utils.data.Subset(imagenet_dataset, gallery_indices)

    query_loader = DataLoader(query_dataset, batch_size=64, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=64, shuffle=False)

    # Load base model (e.g., ResNet-50)
    base_model = torchvision.models.resnet50(pretrained=True)
    base_model.fc = torch.nn.Identity()  # Remove the final classification layer
    base_model = base_model.to(device)

    # Load your custom encoder
    custom_model = CustomEncoder().to(device)

    # Evaluate on ImageNet subset
    print("Evaluating Image Retrieval on ImageNet subset:")
    base_recall_1, base_recall_5, base_mAP = evaluate_image_retrieval(base_model, query_loader, gallery_loader, device)
    custom_recall_1, custom_recall_5, custom_mAP = evaluate_image_retrieval(custom_model, query_loader, gallery_loader, device)

    print(f"Base Model - Recall@1: {base_recall_1:.4f}, Recall@5: {base_recall_5:.4f}, mAP: {base_mAP:.4f}")
    print(f"Custom Model - Recall@1: {custom_recall_1:.4f}, Recall@5: {custom_recall_5:.4f}, mAP: {custom_mAP:.4f}")

if __name__ == "__main__":
    main()