import torch
import torch.nn as nn
import pennylane as qml
from app.core.config import settings
from loguru import logger

class QuantumBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = settings.FEATURE_COUNT
        self.seq_len = settings.SEQUENCE_LENGTH
        # WIR VERKLEINERN DAS GEHIRN (Engpass)
        self.embed_dim = 32  # Vorher 64 -> Zwingt zur Komprimierung
        
        # 1. Input Transformation
        self.input_proj = nn.Linear(self.features, self.embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_len, self.embed_dim))
        
        # 2. Transformer (Kleiner, aber tiefer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=4, batch_first=True, dim_feedforward=64)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3) # Mehr Layer, weniger Breite
        
        # 3. VAE Bottleneck (Extrem eng!)
        # Wir komprimieren alles auf nur 4 Zahlen! Das zwingt die AI, das Konzept "Gesund" zu lernen.
        self.latent_dim = 4 
        self.to_latent = nn.Linear(self.embed_dim * self.seq_len, self.latent_dim * 2)
        
        # 4. Decoder
        self.decoder_input = nn.Linear(self.latent_dim, self.embed_dim * self.seq_len)
        self.final_proj = nn.Linear(self.embed_dim, self.features)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if x.device != settings.DEVICE: x = x.to(settings.DEVICE)
        B, S, F = x.shape
        
        # Encoding
        x_emb = self.input_proj(x) + self.pos_encoder.to(x.device)[:, :S, :]
        x_trans = self.transformer(x_emb)
        flat = x_trans.reshape(B, -1)
        
        # Bottleneck
        latent_params = self.to_latent(flat)
        mu, logvar = torch.chunk(latent_params, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        
        # Decoding
        dec_in = self.decoder_input(z)
        dec_in = dec_in.reshape(B, S, self.embed_dim)
        reconstruction = self.final_proj(dec_in)
        
        return reconstruction, mu, logvar

    def get_loss(self, x, reconstruction, mu, logvar):
        if x.device != reconstruction.device: x = x.to(reconstruction.device)
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='mean')
        # Stärkerer KL-Loss zwingt zu Ordnung
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + (0.005 * kl_loss)
        return total_loss, recon_loss.item()