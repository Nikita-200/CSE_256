import torch
import torch.nn as nn
from dfg import DynamicFusionGraph
torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection globally

def init_lstm_weights(lstm_layer):
    for name, param in lstm_layer.named_parameters():
        if 'weight_ih' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'weight_hh' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)
            n = param.size(0) // 4
            param.data[n:2 * n].fill_(1)  # Initialize forget gate bias to 1

class GraphMemoryFusionNetwork(nn.Module):
    """
    Graph Memory Fusion Network (Graph-MFN) with LSTMs, DFG, and Multi-view Gated Memory.
    """
    def __init__(self, input_dims, output_dim, hidden_dim=128, lstm_layers=1):
        """
        Args:
            input_dims (list): List of input dimensions for [language, vision, acoustic].
            output_dim (int): Number of output classes (for emotion classification).
            hidden_dim (int): Dimension of the hidden representation for graph memory.
            lstm_layers (int): Number of LSTM layers for sequential modeling.
        """
        super(GraphMemoryFusionNetwork, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        # LSTM layers for each modality
        self.language_lstm = nn.LSTM(input_dims[0], hidden_dim, lstm_layers, batch_first=True)  # self.language_lstm LSTM(300, 128, batch_first=True)
        self.vision_lstm = nn.LSTM(input_dims[1], hidden_dim, lstm_layers, batch_first=True) # self.vision_lstm LSTM(746, 128, batch_first=True)
        self.acoustic_lstm = nn.LSTM(input_dims[2], hidden_dim, lstm_layers, batch_first=True) # self.acoustic_lstm LSTM(74, 128, batch_first=True)
        # Initialize weights
        init_lstm_weights(self.language_lstm)
        init_lstm_weights(self.vision_lstm)
        init_lstm_weights(self.acoustic_lstm)

        # Delta Networks for memory tracking across two consecutive timestamps
        self.Dl = nn.Linear( hidden_dim, hidden_dim)  # Language Delta Network
        self.Dv = nn.Linear(hidden_dim, hidden_dim)  # Vision Delta Network
        self.Da = nn.Linear(hidden_dim, hidden_dim)  # Acoustic Delta Network

        # Graph Memory Fusion (DFG)
        self.graph_memory = DynamicFusionGraph(hidden_dim, hidden_dim)

        # Multi-view Gated Memory Networks
        self.Du = nn.Linear(hidden_dim, hidden_dim)  # Update memory transformation network
        self.D1 = nn.Linear(hidden_dim, hidden_dim)  # Retain gate network
        self.D2 = nn.Linear(hidden_dim, hidden_dim)  # Update gate network

        # Sentiment prediction head (regression)
        self.sentiment_head = nn.Linear(hidden_dim, 1)    # Linear transformation from 128 to 1

        # Emotion prediction head (classification for 6 emotions)
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, 6),
            nn.ReLU())

        self.language_norm = nn.LayerNorm(hidden_dim)
        self.vision_norm = nn.LayerNorm(hidden_dim)
        self.acoustic_norm = nn.LayerNorm(hidden_dim)

        # Add Dropout (optional)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, language, vision, acoustic):
        """
        Args:
            language (Tensor): Language input. Shape: [batch_size, seq_len, input_dims[0]]
            vision (Tensor): Vision input. Shape: [batch_size, seq_len, input_dims[1]]
            acoustic (Tensor): Acoustic input. Shape: [batch_size, seq_len, input_dims[2]]
        Returns:
            sentiment_output (Tensor): Predicted sentiment score. Shape: [batch_size]
            emotion_outputs (Tensor): Predicted emotion scores. Shape: [batch_size, 6]
        """
        #print("language",language.shape)  #language torch.Size([32, 50, 300])
        #print("vision",vision.shape)   #vision torch.Size([32, 50, 746])
        #print("acoustic",acoustic.shape)   #acoustic torch.Size([32, 50, 74])

        if torch.any(torch.isnan(language)) or torch.any(torch.isinf(language)):
            language[torch.isnan(language)] = 0.0
            language[torch.isinf(language)] = 0.0
        if torch.any(torch.isnan(vision)) or torch.any(torch.isinf(vision)):
            vision[torch.isnan(vision)] = 0.0
            vision[torch.isinf(vision)] = 0.0
        if torch.any(torch.isnan(acoustic)) or torch.any(torch.isinf(acoustic)):
            acoustic[torch.isnan(acoustic)] = 0.0
            acoustic[torch.isinf(acoustic)] = 0.0

        # Process each modality with LSTM
        language_lstm_out, _ = self.language_lstm(language)  # [batch_size, seq_len, hidden_dim] #language_lstm_out torch.Size([32, 50, 128]) 
        vision_lstm_out, _ = self.vision_lstm(vision)        # [batch_size, seq_len, hidden_dim] #vision_lstm_out torch.Size([32, 50, 128])     
        acoustic_lstm_out, _ = self.acoustic_lstm(acoustic)  # [batch_size, seq_len, hidden_dim] #acoustic_lstm_out torch.Size([32, 50, 128])

        language_lstm_out = self.language_norm(language_lstm_out)
        vision_lstm_out = self.vision_norm(vision_lstm_out)
        acoustic_lstm_out = self.acoustic_norm(acoustic_lstm_out)

        language_emb = self.dropout(language_lstm_out[:, -1, :]) # [batch_size, hidden_dim] #language_emb torch.Size([32, 128])
        vision_emb = self.dropout(vision_lstm_out[:, -1, :])  # [batch_size, hidden_dim]  #vision_emb torch.Size([32, 128])
        acoustic_emb = self.dropout(acoustic_lstm_out[:, -1, :]) # [batch_size, hidden_dim] #acoustic_emb torch.Size([32, 128])

        Dl_output = torch.tanh(self.Dl(language_emb))  # Memory updates for language - #Dl_output torch.Size([32, 128])   
        Dv_output = torch.tanh(self.Dv(vision_emb))  # Memory updates for vision - #Dv_output torch.Size([32, 128])
        Da_output = torch.tanh(self.Da(acoustic_emb))  # Memory updates for acoustic - #Da_output torch.Size([32, 128])

        # Graph Memory Fusion (DFG)
        fused_representations = self.graph_memory(Dl_output, Dv_output, Da_output)["final_representation"] #fused_representations torch.Size([32, 128])

        # Multi-view Gated Memory update
        Tt = fused_representations  # Output vertex from DFG
        ut = self.Du(Tt)  # Proposed memory update #ut torch.Size([32, 128])
 
        retain_gate = torch.sigmoid(self.D1(ut))  # Retain gate  #retain_gate torch.Size([32, 128])
        update_gate = torch.sigmoid(self.D2(ut))  # Update gate #update_gate torch.Size([32, 128])

        # Applying gates to update memory
        updated_memory = retain_gate * Tt + update_gate * ut #updated_memory torch.Size([32, 128])
        # Final representation for prediction
        final_representation = updated_memory  # Using the final memory for sentiment/emotion prediction

        sentiment_output = self.sentiment_head(final_representation)   # [batch_size] #sentiment_output torch.Size([32])
        emotion_outputs = self.emotion_head(final_representation)              # [batch_size, 6] #emotion_outputs torch.Size([32, 6])

        return sentiment_output, emotion_outputs
 