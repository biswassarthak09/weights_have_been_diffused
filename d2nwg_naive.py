import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms

# Define hyperparameters
INPUT_DIM = None  # Will be determined later
LATENT_DIM = 128  # Tune this
BATCH_SIZE = 64  # Tune this
T_STEPS = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define the StrongCNN model
class StrongCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 5)  # Removed BatchNorm1d, ReLU, Dropout to match paper
        )

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.classifier(x)


# 2. Flatten + Pad Weights
@torch.no_grad()
def flatten_and_pad_weights(model, target_length=None):
    flat = torch.cat([p.flatten() for p in model.parameters()])
    if target_length is not None and flat.shape[0] < target_length:
        flat = F.pad(flat, (0, target_length - flat.shape[0]))
    elif target_length is not None and flat.shape[0] > target_length:
        flat = flat[:target_length]
    return flat

@torch.no_grad()
def unflatten_weights(model, flat_weights):
    pointer = 0
    new_state = {}
    for name, param in model.named_parameters():
        shape = param.shape
        size = param.numel()
        new_state[name] = flat_weights[pointer:pointer+size].reshape(shape)
        pointer += size
    model.load_state_dict(new_state, strict=False)
    return model


# Initialize a model to get the INPUT_DIM
temp_model = StrongCNN().to(device)
INPUT_DIM = len(flatten_and_pad_weights(temp_model))
del temp_model

# 3. Weight VAE
class WeightVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.LayerNorm(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.LayerNorm(512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LayerNorm(1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


# 4. Dataset Encoder (CNN Aggregation - Modified to take batch)
class SetEncoder(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.projection = nn.Sequential(
            nn.Linear(128, feat_dim), nn.LayerNorm(feat_dim), nn.ReLU()  # Replace BatchNorm1d with LayerNorm1d
        )

    def forward(self, x):  # x is now a batch of images
        feats = self.cnn(x).squeeze(-1).squeeze(-1)  # Shape: [BATCH_SIZE, 128]
        context = feats.mean(dim=0, keepdim=True)  # Aggregate across batch
        return self.projection(context)


# 5. Simplified DDPM in Latent Space
class SimpleDDPM(nn.Module):
    def __init__(self, latent_dim, context_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + context_dim + 1, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, zt, context, t):
        t_emb = t.float().view(-1, 1) / T_STEPS  # normalize and ensure shape [BATCH_SIZE, 1]
        context = context.expand(zt.size(0), -1)  # Broadcast context to match batch size
        x = torch.cat([zt, context, t_emb], dim=-1)
        return self.net(x)

# 6. Prepare Dataset (FashionMNIST, 5 classes)
transform = transforms.Compose([transforms.ToTensor()])
full_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
class_indices = [0, 1, 2, 3, 4]
subset_indices = [i for i, (_, label) in enumerate(full_dataset) if label in class_indices]
small_subset = Subset(full_dataset, subset_indices[:1000])
dataloader = DataLoader(small_subset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize components
vae = WeightVAE(INPUT_DIM, LATENT_DIM).to(device)
dataset_encoder = SetEncoder(feat_dim=32).to(device)
ddpm = SimpleDDPM(latent_dim=LATENT_DIM, context_dim=32).to(device)
# vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
# ddpm_optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-3)

# 7. VAE Training
pretrained_weights = []
model_zs = []
for _ in range(10):
    model = StrongCNN().to(device)
    x, _ = next(iter(dataloader))
    _ = model(x.to(device))
    wvec = flatten_and_pad_weights(model, target_length=INPUT_DIM).to(device)
    pretrained_weights.append(wvec)
    with torch.no_grad():
        mu, logvar = vae.encode(wvec.unsqueeze(0))
        z = vae.reparam(mu, logvar)
        model_zs.append(z.squeeze(0))

weight_tensor = torch.stack(pretrained_weights).to(device)

vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4, weight_decay=1e-5)  # Tune lr
scheduler = torch.optim.lr_scheduler.StepLR(vae_optimizer, step_size=10, gamma=0.5)

print("VAE Training")
num_epochs_vae = 100  # Tune this
for epoch in range(num_epochs_vae):
    print(f"Epoch {epoch+1}/{num_epochs_vae}")
    vae.train()
    total_loss = 0
    vae_batch_size = 5  # Tune this
    for i in range(0, weight_tensor.size(0), vae_batch_size):
        batch = weight_tensor[i:i + vae_batch_size]
        x_hat, mu, logvar = vae(batch)
        recon_loss = F.mse_loss(x_hat, batch)
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.1 * kld  # Tune KLD weight

        vae_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        vae_optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"VAE Epoch {epoch}: Loss = {total_loss / (weight_tensor.size(0) / vae_batch_size):.4f}")

# More sophisticated noise schedule
def cosine_noise_schedule(t, T):
    return torch.cos((t/T) * torch.pi/2).pow(2)

# 8. Train DDPM on VAE Latents
model_zs = torch.stack(model_zs).detach()

ddpm_optimizer = torch.optim.Adam(ddpm.parameters(), lr=2e-4, betas=(0.9, 0.999))  # Tune lr
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ddpm_optimizer, T_max=20)

print("DDPM Training")
num_epochs_ddpm = 100  # Tune this
for epoch in range(num_epochs_ddpm):
    print(f"Epoch {epoch+1}/{num_epochs_ddpm}")
    for x, _ in dataloader:
        ctx = dataset_encoder(x.to(device))
        z = model_zs[torch.randint(0, len(model_zs), (x.size(0),))].to(device)
        t = torch.randint(0, T_STEPS, (x.size(0),)).to(device)
        noise = torch.randn_like(z)
        alpha = cosine_noise_schedule(t.float(), T_STEPS).view(-1, 1).to(device)
        zt = alpha.sqrt() * z + (1 - alpha).sqrt() * noise  # Corrected noise schedule

        noise_pred = ddpm(zt, ctx, t)
        loss = F.mse_loss(noise_pred, noise)

        ddpm_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
        ddpm_optimizer.step()
    scheduler.step()
    print(f"DDPM Epoch {epoch}: Loss = {loss.item():.4f}")

# 9. Sampling
@torch.no_grad()
def sample_weights(context):
    zt = torch.randn(1, LATENT_DIM).to(device)
    for t in reversed(range(T_STEPS)):
        t_tensor = torch.tensor([[t]], dtype=torch.float).to(device)
        noise_pred = ddpm(zt, context, t_tensor)
        t_val = torch.tensor([t], dtype=torch.float32).to(device)
        alpha = torch.cos(t_val * torch.pi / 2 / T_STEPS)
        beta = 1 - alpha
        zt = (1 / alpha.sqrt()) * (zt - (beta / beta.sqrt()) * noise_pred)  # Simplified sampling step (assuming variance is identity)
    return vae.decode(zt)


# 10. Evaluation
model = StrongCNN().to(device)
dataset_encoder.eval()

# Get context from a batch (e.g., first batch from dataloader)
batch = next(iter(dataloader))[0].to(device)
context = dataset_encoder(batch)

gen_weights = sample_weights(context).squeeze(0)
model = unflatten_weights(model, gen_weights)
model.eval()

correct = 0
total = 0
eval_dataloader = DataLoader(small_subset, batch_size=32)
for x, y in eval_dataloader:
    x, y = x.to(device), y.to(device)
    outputs = model(x)
    _, preds = torch.max(outputs, 1)
    correct += (preds == y).sum().item()
    total += y.size(0)

print(f"Sampled Model Accuracy: {correct / total * 100:.2f}%")