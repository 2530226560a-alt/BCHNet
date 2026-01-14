r""" Enhanced Frequency Domain Filter with Adaptive Learning """

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedFrequencyFilter(nn.Module):
    """Enhanced Amplitude-Phase Masker with Learnable Frequency Bands"""

    def __init__(self, shape, reduction=8, num_bands=4):
        super().__init__()
        
        bsz, channel, h, w = shape[0], shape[1], shape[2], shape[3]
        self.channel = channel
        self.num_bands = num_bands
        
        # Learnable amplitude and phase masks for different frequency bands
        self.band_masks_amp = nn.ParameterList([
            nn.Parameter(torch.ones(1, channel, h, w)) for _ in range(num_bands)
        ])
        self.band_masks_phase = nn.ParameterList([
            nn.Parameter(torch.ones(1, channel, h, w)) for _ in range(num_bands)
        ])
        
        # Adaptive pooling and attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Dual-path attention for amplitude and phase
        self.amp_attn = nn.Sequential(
            nn.Linear(channel * 2, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
        self.phase_attn = nn.Sequential(
            nn.Linear(channel * 2, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
        # Band selection weights
        self.band_weights = nn.Parameter(torch.ones(num_bands) / num_bands)
        
    def get_frequency_bands(self, freq_domain, num_bands):
        """Divide frequency domain into multiple bands"""
        h, w = freq_domain.shape[-2:]
        center_h, center_w = h // 2, w // 2
        
        bands = []
        for i in range(num_bands):
            mask = torch.zeros_like(freq_domain)
            radius_inner = int((i / num_bands) * min(center_h, center_w))
            radius_outer = int(((i + 1) / num_bands) * min(center_h, center_w))
            
            y, x = torch.meshgrid(
                torch.arange(h, device=freq_domain.device),
                torch.arange(w, device=freq_domain.device),
                indexing='ij'
            )
            dist = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
            band_mask = ((dist >= radius_inner) & (dist < radius_outer)).float()
            
            bands.append(freq_domain * band_mask.unsqueeze(0).unsqueeze(0))
        
        return bands

    def forward(self, x):
        # Extract amplitude and phase information
        freq_domain = torch.fft.fftshift(torch.fft.fft2(x))
        amplitude = torch.abs(freq_domain)
        phase = torch.angle(freq_domain)
        
        # Multi-band processing
        adjusted_amplitude = torch.zeros_like(amplitude)
        adjusted_phase = torch.zeros_like(phase)
        
        band_weights_norm = F.softmax(self.band_weights, dim=0)
        
        for i in range(self.num_bands):
            # Apply learnable masks to each band
            mask_amp = torch.sigmoid(self.band_masks_amp[i])
            mask_phase = torch.sigmoid(self.band_masks_phase[i])
            
            adjusted_amplitude += band_weights_norm[i] * mask_amp * amplitude
            adjusted_phase += band_weights_norm[i] * mask_phase * phase
        
        # Reconstruct frequency domain
        adjusted_freq = torch.polar(adjusted_amplitude, adjusted_phase)
        adjusted_x = torch.fft.ifft2(torch.fft.ifftshift(adjusted_freq)).real
        
        # Dual-path channel attention
        # Amplitude path
        amp_avg = self.avg_pool(amplitude).view(x.size(0), self.channel)
        amp_max = self.max_pool(amplitude).view(x.size(0), self.channel)
        amp_combined = torch.cat([amp_avg, amp_max], dim=1)
        amp_weights = self.amp_attn(amp_combined).view(x.size(0), self.channel, 1, 1)
        
        # Phase path
        phase_avg = self.avg_pool(phase).view(x.size(0), self.channel)
        phase_max = self.max_pool(phase).view(x.size(0), self.channel)
        phase_combined = torch.cat([phase_avg, phase_max], dim=1)
        phase_weights = self.phase_attn(phase_combined).view(x.size(0), self.channel, 1, 1)
        
        # Combine both paths
        combined_weights = (amp_weights + phase_weights) * 0.5
        enhanced_x = adjusted_x * combined_weights.expand_as(x)
        
        return enhanced_x


class AdaptiveFrequencyFilter(nn.Module):
    """Adaptive Frequency Filter with Domain-specific Learning"""
    
    def __init__(self, shape, reduction=8):
        super().__init__()
        
        bsz, channel, h, w = shape[0], shape[1], shape[2], shape[3]
        self.channel = channel
        
        # Learnable frequency masks
        self.mask_amplitude = nn.Parameter(torch.ones(1, channel, h, w))
        self.mask_phase = nn.Parameter(torch.ones(1, channel, h, w))
        
        # Context-aware gating
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )
        
        # Frequency-spatial fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1)
        )
        
    def forward(self, x):
        # Frequency domain transformation
        freq_domain = torch.fft.fftshift(torch.fft.fft2(x))
        amplitude = torch.abs(freq_domain)
        phase = torch.angle(freq_domain)
        
        # Apply learnable masks
        mask_amplitude = torch.sigmoid(self.mask_amplitude)
        mask_phase = torch.sigmoid(self.mask_phase)
        
        adjusted_amplitude = mask_amplitude * amplitude
        adjusted_phase = mask_phase * phase
        
        # Reconstruct
        adjusted_freq = torch.polar(adjusted_amplitude, adjusted_phase)
        freq_enhanced = torch.fft.ifft2(torch.fft.ifftshift(adjusted_freq)).real
        
        # Context-aware gating
        gate = self.context_gate(x)
        freq_gated = freq_enhanced * gate
        
        # Frequency-spatial fusion
        combined = torch.cat([x, freq_gated], dim=1)
        output = self.fusion(combined)
        
        # Residual connection
        output = output + x
        
        return output
