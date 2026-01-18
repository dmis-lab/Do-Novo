"""Tranformer models to handle mass spectra."""
from collections.abc import Callable
from typing import Optional, Tuple
import torch
import torch.nn as nn
from depthcharge.tokenizers import PeptideTokenizer
from do_novo.model.sinusoidal import PeakEncoder
from do_novo.model.decoder import PositionalEncoder


class SpectrumTransformerEncoder(torch.nn.Module):
    """A Transformer encoder for input mass spectra.

    Parameters
    ----------
    d_model : int, optional
        The latent dimensionality to represent peaks in the mass spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``d_model`` must be
        divisible by ``nhead``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    peak_encoder : PeakEncoder or bool, optional
        Sinusoidal encodings m/z and intensityvalues of each peak.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        peak_encoder: PeakEncoder | Callable | bool = True,
        k: int = 1,
    ) -> None:
        """Initialize a SpectrumEncoder."""
        super().__init__()
        self.k = k
        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, self.k, d_model))
        if callable(peak_encoder):
            self.peak_encoder = peak_encoder
        elif peak_encoder:
            self.peak_encoder = PeakEncoder(d_model)
        else:
            self.peak_encoder = torch.nn.Linear(2, d_model)

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False, # baek
        )

    def forward(
        self,
        spectra: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed a mass spectrum.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, d_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        zeros = ~spectra.sum(dim=2).bool()
        mask = [
            torch.tensor([[False]*self.k] * spectra.shape[0]).type_as(zeros),
            zeros,
        ]
        mask = torch.cat(mask, dim=1)
        peaks = self.peak_encoder(spectra)

        # Add the spectrum representation to each input:
        latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)

        peaks = torch.cat([latent_spectra, peaks], dim=1)
        return self.transformer_encoder(peaks, src_key_padding_mask=mask), mask

    @property
    def device(self) -> torch.device:
        """The current device for the model."""
        return next(self.parameters()).device
    


class _PeptideTransformer(torch.nn.Module):
    """A transformer base class for peptide sequences.

    Parameters
    ----------
    n_tokens : int or PeptideTokenizer
        The number of tokens used to tokenize peptide sequences.
    d_model : int
        The latent dimensionality to represent the amino acids in a peptide
        sequence.
    positional_encoder : PositionalEncoder or bool
        The positional encodings to use for the amino acid sequence. If
        ``True``, the default positional encoder is used. ``False`` disables
        positional encodings, typically only for ablation tests.
    x_charge : int
        The maximum precursor charge to embed.
    """

    def __init__(
        self,
        n_tokens: int | PeptideTokenizer,
        d_model: int,
        positional_encoder: PositionalEncoder | bool,
        max_charge: int,
    ) -> None:
        super().__init__()
        try:
            n_tokens = len(n_tokens)
        except TypeError:
            pass

        if callable(positional_encoder):
            self.positional_encoder = positional_encoder
        elif positional_encoder:
            self.positional_encoder = PositionalEncoder(d_model)
        else:
            self.positional_encoder = torch.nn.Identity()

        self.charge_encoder = torch.nn.Embedding(max_charge + 1, d_model)
        self.aa_encoder = torch.nn.Embedding(
            n_tokens + 1,
            d_model,
            padding_idx=0,
        )

    @property
    def device(self) -> torch.device:
        """The current device for the model."""
        return next(self.parameters()).device



class PeptideTransformerEncoder(_PeptideTransformer):
    """A transformer encoder for peptide sequences.

    Parameters
    ----------
    n_tokens : int or PeptideTokenizer
        The number of tokens used to tokenize peptide sequences.
    d_model : int
        The latent dimensionality to represent the amino acids in a peptide
        sequence.
    nhead : int, optional
        The number of attention heads in each layer. ``d_model`` must be
        divisible by ``nhead``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the Transformer
        layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    positional_encoder : PositionalEncoder or bool, optional
        The positional encodings to use for the amino acid sequence. If
        ``True``, the default positional encoder is used. ``False`` disables
        positional encodings, typically only for ablation tests.
    max_charge : int, optional
        The maximum charge state for peptide sequences.
    """

    def __init__(
        self,
        n_tokens: int | PeptideTokenizer,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        positional_encoder: PositionalEncoder | bool = True,
        max_charge: int = 5,
    ) -> None:
        """Initialize a PeptideEncoder."""
        super().__init__(
            n_tokens=n_tokens,
            d_model=d_model,
            positional_encoder=positional_encoder,
            max_charge=max_charge,
        )

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        charges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        tokens : torch.Tensor of size (batch_size, peptide_length)
            The integer tokens describing each peptide sequence, padded
            to the maximum peptide length in the batch with 0s.
        charges : torch.Tensor of size (batch_size,)
            The charge state of each peptide.

        Returns
        -------
        latent : torch.Tensor of shape (n_sequences, len_sequence, d_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        # Encode everything:``
        encoded = self.aa_encoder(tokens)
        charges = self.charge_encoder(charges)[:, None]
        encoded = torch.cat([charges, encoded], dim=1)

        # Create mask
        mask = ~encoded.sum(dim=2).bool()

        # Add positional encodings
        encoded = self.positional_encoder(encoded)

        # Run through the model:
        latent = self.transformer_encoder(encoded, src_key_padding_mask=mask)
        return latent, mask
    



class SpectrumTheoCrossAttention(nn.Module):
    """
    emb        : (B, L_spec, d_model)  # spectrum encoder output
    emb_mask   : (B, L_spec)           # True/1 = pad 위치 (standard key_padding_mask 스타일이면 True=pad)
    Theo_mz : (B, L_miss, d_model)  # Theo m/z embedding

    forward(emb, emb_mask, Theo_mz_enc) -> (emb', emb_mask)
    """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head

        # Q = emb (spectrum), K/V = Theo_mz_enc
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, emb, emb_mask, theo_mz_enc, theo_mask=None):
        # theo_mask: (B, L_miss), True=pad
        attn_out, attn_weights = self.cross_attn(
            query=emb,
            key=theo_mz_enc,
            value=theo_mz_enc,
            key_padding_mask=theo_mask,
        )
        x = self.norm1(emb + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x, emb_mask, attn_weights
    

class SelfAttnFocus(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln  = nn.LayerNorm(d_model)

    def forward(self, emb0, emb0_mask):
        key_padding_mask = emb0_mask.bool() if emb0_mask is not None else None
        attn_out, attn_w = self.mha(
            emb0, emb0, emb0,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,   # (B,H,L,L)
        )
        emb0_sa = self.ln(emb0 + attn_out)
        return emb0_sa, attn_w
    
    

class EmbAttnFocus(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0, pseudo_n=10):
        super().__init__()
        self.pseudo_n = pseudo_n
        
        self.mha = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln  = nn.LayerNorm(d_model)
        
        self.pseudo = nn.Parameter(torch.zeros(1, pseudo_n, d_model))

    def forward(self, emb0, emb0_mask, spectra):
        B = emb0.size(0)
        
        ms2_mask = (spectra[:, :, 3] != 2)   # (B, 826)
        front_mask = torch.ones((B, 1), dtype=torch.bool, device=ms2_mask.device)
        pseudo_unmask = torch.zeros((B, self.pseudo_n), dtype=torch.bool, device=ms2_mask.device)
        ms2_mask = torch.cat([front_mask, ms2_mask, pseudo_unmask], dim=1)
        
        emb0_kv = torch.cat([emb0, self.pseudo.expand(B, -1, -1).to(spectra.device)], dim=1)
        
        attn_out, attn_w = self.mha(
            emb0, emb0_kv, emb0_kv,
            key_padding_mask=ms2_mask,
            need_weights=True,
            average_attn_weights=False,   # (B,H,L,L)
        )
        emb0_ea = self.ln(emb0 + attn_out)
        
        
        aw = attn_w.mean(1)        
        ms2_sum = (aw * (~ms2_mask)[:, None, :].to(aw.dtype)).sum(dim=2) - aw[:, :, -self.pseudo_n:].sum(dim=2)
        pseudo_sum = aw[:, :, -self.pseudo_n:].sum(dim=2)

        return emb0_ea, attn_w, ms2_sum, pseudo_sum
    