import torch
import torch.nn as nn
import torch
from PIL import Image
import requests
from transformers import AutoProcessor, Blip2Model
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F

from torchvision import transforms
from pytorch_msssim import ssim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

blip2_model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
# blip2_model = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xxl")

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl)
dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
dinov2_model = AutoModel.from_pretrained('facebook/dinov2-large')

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        return output

class AttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class ImageDecoder(nn.Module):
    def __init__(self, input_dim, output_channels=3):
        super(ImageDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, 14*14*512) 
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 14, 14)  # Reshape to [batch_size, 512, 14, 14]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.tanh(self.conv4(x))  # Output in range [-1, 1]
        return x

class HierarchicalAttentionFusion(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout=0.1, temp=0.07):
        super(HierarchicalAttentionFusion, self).__init__()
        self.num_layers = num_layers
        self.temp = temp
        
        self.blip2_projection = nn.Linear(1408, d_model)
        self.dinov2_projection = nn.Linear(1024, d_model)
        
        self.blip2_self_attention = nn.ModuleList([AttentionBlock(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.dinov2_self_attention = nn.ModuleList([AttentionBlock(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.cross_attention = nn.ModuleList([MultiHeadAttention(d_model, num_heads) for _ in range(num_layers)])
        self.fused_self_attention = nn.ModuleList([AttentionBlock(d_model, num_heads, dropout) for _ in range(num_layers)])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Image Decoder
        self.decoder = ImageDecoder(d_model)
        
    def kl_divergence(self, p, q):
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        return torch.sum(p * torch.log(p / q), dim=-1).mean()
        
    def forward(self, blip2_features, dinov2_features, original_image):
        blip2_features_proj = self.blip2_projection(blip2_features)
        dinov2_features_proj = self.dinov2_projection(dinov2_features)
        
        for i in range(self.num_layers):
            blip2_attended = self.blip2_self_attention[i](blip2_features_proj)
            dinov2_attended = self.dinov2_self_attention[i](dinov2_features_proj)
            
            fused_features = self.cross_attention[i](blip2_attended, dinov2_attended, dinov2_attended)
            
            fused_features = self.fused_self_attention[i](fused_features)
            
            blip2_features_proj = blip2_features_proj + blip2_attended
            dinov2_features_proj = dinov2_features_proj + dinov2_attended
        
        output = self.norm(fused_features)
        
        # ITC Loss calculation
        sim_b2d = torch.matmul(blip2_features_proj.mean(1), dinov2_features_proj.mean(1).t()) / self.temp
        sim_d2b = sim_b2d.t()
        
        bs = blip2_features_proj.size(0)
        targets = torch.arange(bs, dtype=torch.long, device=blip2_features_proj.device)
        
        loss_b2d = F.cross_entropy(sim_b2d, targets)
        loss_d2b = F.cross_entropy(sim_d2b, targets)
        loss_itc = (loss_b2d + loss_d2b) / 2
        
        # KL Divergence Loss calculation
        kl_loss_blip = self.kl_divergence(blip2_features_proj, output)
        kl_loss_dino = self.kl_divergence(dinov2_features_proj, output)
        loss_kl = (kl_loss_blip + kl_loss_dino) / 2
        
        # Image Reconstruction
        reconstructed_image = self.decoder(output.mean(dim=1))
        
        # Image Reconstruction Losses
        loss_l2 = F.mse_loss(reconstructed_image, original_image)
        loss_ssim = 1 - ssim(reconstructed_image, original_image, data_range=2.0, size_average=True) 
        
        return output, reconstructed_image, loss_itc, loss_kl, loss_l2, loss_ssim

class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None, mask_ratio=0.15):
        self.image_paths = glob.glob(os.path.join(image_folder, '*.*')) 
        self.transform = transform
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            original_image = self.transform(image)
        else:
            original_image = transforms.ToTensor()(image)

        # Apply random mask
        masked_image = apply_random_mask(original_image, self.mask_ratio)

        # Convert masked_image tensor back to [0,1] for PIL conversion
        masked_image_01 = (masked_image + 1) / 2  
        masked_image_pil = transforms.ToPILImage()(masked_image_01)

        return original_image, masked_image_pil

def apply_random_mask(image, mask_ratio=0.15):
    mask = torch.rand(image.shape) > mask_ratio
    return image * mask

def custom_collate(batch):
    original_images = torch.stack([item[0] for item in batch], dim=0)
    masked_images_pil = [item[1] for item in batch]
    return original_images, masked_images_pil

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == '__main__':
    # Set up device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Initialize BLIP2 model and processor
    blip2_model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip2_model.eval()
    for param in blip2_model.parameters():
        param.requires_grad = False
    blip2_model = blip2_model.to(device)

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    # Initialize DINOv2 model and processor
    dinov2_model = AutoModel.from_pretrained('facebook/dinov2-large')
    dinov2_model.eval()
    for param in dinov2_model.parameters():
        param.requires_grad = False
    dinov2_model = dinov2_model.to(device)

    dinov2_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')

    d_model = 768
    num_heads = 8
    num_layers = 3

    model = HierarchicalAttentionFusion(d_model, num_heads, num_layers)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    
    model = model.to(device)

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    image_folder = '/media/mlr_lab/325C37DE7879ABF2/AyushAnand/r2r'
    # image_folder = '/media/mlr_lab/325C37DE7879ABF2/prarabda/coco/train2014'
    mask_ratio = 0.15
    batch_size = 16
    num_workers = 4
    num_epochs = 10

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Converts to [0,1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Now in [-1,1]
    ])

    dataset = ImageDataset(image_folder=image_folder, transform=transform, mask_ratio=mask_ratio)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate 
    )
    best_loss = 1e9
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", unit='batch') as pbar:
            for i, (original_images, masked_images_pil) in enumerate(dataloader):
                original_images = original_images.to(device)
            
                blip2_inputs = processor(images=masked_images_pil, return_tensors="pt")
                blip2_inputs = {k: v.to(device) for k, v in blip2_inputs.items()}
                blip_image_embeds = blip2_model.get_image_features(**blip2_inputs).last_hidden_state

                dinov2_inputs = dinov2_processor(images=masked_images_pil, return_tensors="pt")
                dinov2_inputs = {k: v.to(device) for k, v in dinov2_inputs.items()}
                dino_image_embeds = dinov2_model(**dinov2_inputs).last_hidden_state

                output, reconstructed_image, loss_itc, loss_kl, loss_l2, loss_ssim = model(blip_image_embeds, dino_image_embeds, original_images)

                total_loss = loss_itc + loss_kl + loss_l2 + loss_ssim
                print(f"loss_itc: {loss_itc}, loss_kl: {loss_kl}, loss_l2: {loss_l2}, loss_ssim: {loss_ssim} ||||| total_loss: {total_loss}")

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': total_loss.item()})
                pbar.update(1)

                running_loss += total_loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'best_model_r2r.pth')
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")