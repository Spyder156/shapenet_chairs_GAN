import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import PointCloudDataset
from utils.models import Generator, Discriminator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_gan(epochs, dataloader, latent_dim, random_dim):
    adversarial_loss = nn.BCELoss()
    generator = Generator(random_dim, latent_dim).to(device)
    discriminator = Discriminator(latent_dim).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0004)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)

    writer = SummaryWriter(log_dir="../logs")  # TensorBoard Logging

    for epoch in range(epochs):
        for real_point_clouds in dataloader:
            real_point_clouds = real_point_clouds.view(-1, latent_dim).to(device)

            real_label = torch.ones(real_point_clouds.size(0), 1).to(device)
            fake_label = torch.zeros(real_point_clouds.size(0), 1).to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            z = torch.randn(real_point_clouds.size(0), latent_dim).to(device)
            fake_point_clouds = generator(z)
            real_loss = adversarial_loss(discriminator(real_point_clouds), real_label)
            fake_loss = adversarial_loss(discriminator(fake_point_clouds.detach()), fake_label)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            g_loss = adversarial_loss(discriminator(fake_point_clouds), real_label)
            g_loss.backward()
            optimizer_G.step()

        writer.add_scalar('Loss/Discriminator', d_loss.item(), epoch)
        writer.add_scalar('Loss/Generator', g_loss.item(), epoch)
        print(f"Epoch {epoch}/{epochs} - D Loss: {d_loss.item()} - G Loss: {g_loss.item()}")

    writer.close()
    torch.save(generator.state_dict(), "../models/generator_final.pth")
    torch.save(discriminator.state_dict(), "../models/discriminator_final.pth")

if __name__ == "__main__":
    pcd_folder = "../data/shapenet-chairs-pcd"
    dataset = PointCloudDataset(folder=pcd_folder)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    train_gan(epochs=100, dataloader=dataloader, latent_dim=100, random_dim=100)
