import torch
import torch.nn as nn
#dfg model implementation
class DynamicFusionGraph(nn.Module):
    """
    Dynamic Fusion Graph (DFG) with modality efficacy computation.
    Models unimodal, bimodal, and trimodal interactions using dynamic graph structure.
    """
    def __init__(self, input_dim, hidden_dim):
        """
        Args:
            input_dim (int): Total input dimension from all modalities combined.
            hidden_dim (int): Dimension of the hidden representation after fusion.
        """
        super(DynamicFusionGraph, self).__init__()
        self.hidden_dim = hidden_dim

        # Gating layers for unimodal, bimodal, and trimodal interactions
        self.unimodal_gate = nn.Linear(hidden_dim, hidden_dim)
        self.bimodal_gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.trimodal_gate = nn.Linear(3 * hidden_dim, hidden_dim)

        # Node update layers for unimodal, bimodal, and trimodal representations
        self.unimodal_update = nn.Linear(hidden_dim, hidden_dim)
        self.bimodal_update = nn.Linear(2 * hidden_dim, hidden_dim)
        self.trimodal_update = nn.Linear(3 * hidden_dim, hidden_dim)

        # Efficacy computation layers (to dynamically compute the strength of the connections)
        self.efficacy_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # Output efficacy scores for unimodal, bimodal, trimodal
            nn.Sigmoid()  # To get values between 0 and 1 for the efficacy (as probabilities)
        )

    def forward(self, language, vision, acoustic):
        """
        Args:
            language (Tensor): Language features. Shape: [batch_size, hidden_dim].
            vision (Tensor): Vision features. Shape: [batch_size, hidden_dim].
            acoustic (Tensor): Acoustic features. Shape: [batch_size, hidden_dim].
        Returns:
            dict: Fused representations for unimodal, bimodal, trimodal interactions and memory updates.
        """
        # print("language",language.shape) #language torch.Size([32, 128])
        # print("vision",vision.shape) #vision torch.Size([32, 128])
        # print("acoustic",acoustic.shape) #acoustic torch.Size([32, 128])
        # Unimodal Fusion (combining all modalities for a unimodal interaction)
        unimodal_language = torch.tanh(self.unimodal_update(language))
        unimodal_vision = torch.tanh(self.unimodal_update(vision))
        unimodal_acoustic = torch.tanh(self.unimodal_update(acoustic))
        unimodal_language_gated = unimodal_language * torch.sigmoid(self.unimodal_gate(language))
        unimodal_vision_gated = unimodal_vision * torch.sigmoid(self.unimodal_gate(vision))
        unimodal_acoustic_gated = unimodal_acoustic * torch.sigmoid(self.unimodal_gate(acoustic))

        unimodal_gated = (unimodal_language_gated + unimodal_vision_gated + unimodal_acoustic_gated) / 3

        # Bimodal Fusion (pairwise combination of modalities)
        bimodal_lv = torch.tanh(self.bimodal_update(torch.cat([language, vision], dim=-1)))  # Language + Vision  #bimodal_lv= torch.Size([32, 128])
        bimodal_la = torch.tanh(self.bimodal_update(torch.cat([language, acoustic], dim=-1)))  # Language + Acoustic #bimodal_la= torch.Size([32, 128])
        bimodal_va = torch.tanh(self.bimodal_update(torch.cat([vision, acoustic], dim=-1)))  # Vision + Acoustic  #bimodal_va= torch.Size([32, 128])

        bimodal_lv_fusion = bimodal_lv * torch.sigmoid(self.bimodal_gate(torch.cat([language, vision], dim=-1)))
        bimodal_la_fusion = bimodal_la * torch.sigmoid(self.bimodal_gate(torch.cat([language, acoustic], dim=-1)))
        bimodal_va_fusion = bimodal_va * torch.sigmoid(self.bimodal_gate(torch.cat([vision, acoustic], dim=-1)))

        bimodal_gated = (bimodal_lv_fusion + bimodal_la_fusion + bimodal_va_fusion) / 3

        # Trimodal Fusion (all three modalities combined)
        trimodal_input = torch.cat([language, vision, acoustic], dim=-1) #trimodal_input= torch.Size([32, 384])
        trimodal = torch.tanh(self.trimodal_update(trimodal_input)) #trimodal= torch.Size([32, 128])
        trimodal_gated = torch.sigmoid(self.trimodal_gate(trimodal_input)) * trimodal #trimodal_gated= torch.Size([32, 128])

        # Efficacy computation
        # Calculate efficacy values based on unimodal, bimodal, trimodal representations
        efficacy_input = torch.cat([unimodal_gated, bimodal_gated, trimodal_gated], dim=-1) #efficacy_input= torch.Size([32, 384])  
        efficacy_scores = self.efficacy_network(efficacy_input)  # Output efficacy values for each type #efficacy_scores= torch.Size([32, 3])

        # Dynamic fusion based on efficacy
        unimodal_fusion = unimodal_gated * efficacy_scores[:, 0].view(-1, 1) #unimodal_fusion torch.Size([32, 128])
        bimodal_fusion = bimodal_gated * efficacy_scores[:, 1].view(-1, 1) #bimodal_fusion torch.Size([32, 128])
        trimodal_fusion = trimodal_gated * efficacy_scores[:, 2].view(-1, 1) #trimodal_fusion torch.Size([32, 128])

        # Final fusion representation
        final_representation = unimodal_fusion + bimodal_fusion + trimodal_fusion #final_representation= torch.Size([32, 128])
        
        # Return the final fusion representation and memory updates
        return {
            "final_representation": final_representation,
            "unimodal": unimodal_gated,
            "bimodal": bimodal_gated,
            "trimodal": trimodal_gated,
            "efficacy_scores": efficacy_scores
        }
