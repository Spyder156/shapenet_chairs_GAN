import torch
import numpy as np
from utils.models import Generator
from utils.visualize import visualize_point_cloud

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latent_dim = 100
generator = Generator(latent_dim, 3000).to(device)  # Output dim = 1000 points * 3
generator.load_state_dict(torch.load("../models/generator_final.pth"))
generator.eval()

z = torch.randn(1, latent_dim).to(device)
generated_point_cloud = generator(z).cpu().detach().numpy().reshape(1000, 3)

visualize_point_cloud(generated_point_cloud, title="Generated Point Cloud")
