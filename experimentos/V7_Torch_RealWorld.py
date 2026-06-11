import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import json
from time import time

# Forçando determinismo
torch.manual_seed(42)
np.random.seed(42)

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class V7_Torch_MoE(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, top_k=3, diversity_weight=0.001, balance_weight=0.1):
        super().__init__()
        self.n_experts = len(hidden_sizes)
        self.output_dim = output_dim
        self.top_k = top_k
        self.diversity_weight = diversity_weight
        self.balance_weight = balance_weight
        
        # Experts Heterogêneos
        self.experts = nn.ModuleList([Expert(input_dim, h, output_dim) for h in hidden_sizes])
        
        # Contextual Gate (3 layers)
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.gate_head = nn.Linear(64, self.n_experts)
        
        self.register_buffer('expert_memory', torch.ones(self.n_experts, output_dim))
        self.ema = 0.9
        
    def forward(self, x, temperature=1.0, y_true=None):
        B = x.shape[0]
        
        # Forward dos experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # [B, out_dim]
        expert_outputs = torch.stack(expert_outputs, dim=1) # [B, n_experts, out_dim]
        
        # Contextual Gate
        g2 = self.gate_net(x)
        logits = self.gate_head(g2) # [B, n_experts]
        gate_probs = F.softmax(logits / temperature, dim=-1) # [B, n_experts]
        
        # Residual Top-K Routing
        topk_probs, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        gate_mask = torch.zeros_like(gate_probs)
        
        # Construindo o mask iterativamente por causa do residual
        for b in range(B):
            idx = topk_indices[b]
            gate_mask[b, idx[0]] = 1.0 # Top-1 absoluto
            for k in range(1, self.top_k):
                gate_mask[b, idx[k]] = 0.5 # Auxiliares residuais
            gate_mask[b] /= gate_mask[b].sum() # Normalizar
            
        gate = gate_mask # Usar mask direto sem gradiente pro gate forward
        gate = gate.unsqueeze(-1) # [B, n_experts, 1]
        
        # Mistura final
        out = (gate * expert_outputs).sum(dim=1) # [B, out_dim]
        
        # ---- LOSSES PERSONALIZADOS ----
        loss_div = 0.0
        loss_balance = 0.0
        target_probs = None
        
        if self.training and y_true is not None:
            # Soft Cosine Diversity
            if self.diversity_weight > 0:
                for i in range(self.n_experts):
                    for j in range(self.n_experts):
                        if i != j:
                            w_i = self.experts[i].net[2].weight # [out_dim, hidden_dim]
                            w_j = self.experts[j].net[2].weight # [out_dim, hidden_dim]
                            min_h = min(w_i.shape[1], w_j.shape[1])
                            w_i_sub = w_i[:, :min_h]
                            w_j_sub = w_j[:, :min_h]
                            cos_sim = F.cosine_similarity(w_i_sub.flatten().unsqueeze(0), w_j_sub.flatten().unsqueeze(0))
                            loss_div += cos_sim.mean()
                            
            # Atualizar memória com predições locais
            with torch.no_grad():
                preds = expert_outputs.argmax(dim=-1) # [B, n_experts]
                correct = (preds == y_true.unsqueeze(-1)).float() # [B, n_experts]
                
                for i in range(self.n_experts):
                    for c in range(self.output_dim):
                        mask = (y_true == c)
                        if mask.sum() > 0:
                            self.expert_memory[i, c] = self.ema * self.expert_memory[i, c] + (1 - self.ema) * correct[mask, i].mean()

                memory_prior = self.expert_memory[:, y_true].T # [B, n_experts]
                memory_prior = memory_prior / (memory_prior.sum(dim=1, keepdim=True) + 1e-9)
                
                sum_correct = correct.sum(dim=1, keepdim=True)
                local_target = torch.where(sum_correct > 0, correct / sum_correct, torch.ones_like(correct)/self.n_experts)
                
                target_probs = 0.5 * local_target + 0.5 * memory_prior
                
            # Gate Loss: KL Divergence
            loss_gate = F.kl_div(F.log_softmax(logits/temperature, dim=-1), target_probs, reduction='batchmean')
            
            # Load Balance Loss
            usage = gate_probs.mean(dim=0)
            loss_balance = ((usage - 1.0/self.n_experts)**2).sum()
            
            # Task Loss (Cross Entropy)
            loss_task = F.cross_entropy(out, y_true)
            
            total_loss = loss_task + self.diversity_weight * loss_div + loss_gate + self.balance_weight * loss_balance
            
            return out, total_loss, gate_probs, g2, expert_outputs
        
        return out, None, gate_probs, g2, expert_outputs

def gini_index(x):
    x = x.flatten()
    diffsum = 0
    for i in x:
        for j in x:
            diffsum += abs(i - j)
    return diffsum / (2 * len(x)**2 * x.mean() + 1e-9)

def compute_eri(expert_outputs):
    preds = expert_outputs.argmax(dim=-1).cpu().numpy()
    n = preds.shape[1]
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            sim_matrix[i, j] = np.mean(preds[:, i] == preds[:, j])
    return (np.sum(sim_matrix)) / (n * (n - 1) + 1e-9)

def train_cifar():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    print("Baixando/Carregando CIFAR-10...")
    os.makedirs('./data', exist_ok=True)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Subsample para treinar mais rápido e avaliar pruning
    if not torch.cuda.is_available():
        print("CUDA não detectado. Usando subset de 10000 imagens para CPU.")
        trainset = torch.utils.data.Subset(trainset, range(10000))
        testset = torch.utils.data.Subset(testset, range(2000))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    hidden_sizes = [64]*2 + [128]*2 + [256]*2 + [512]*2 # 8 Experts para scale
    model = V7_Torch_MoE(input_dim=32*32*3, hidden_sizes=hidden_sizes, output_dim=10, top_k=3)
    model = model.to(device)

    optimizer_experts = optim.Adam(model.experts.parameters(), lr=0.001)
    optimizer_gate = optim.Adam(list(model.gate_net.parameters()) + list(model.gate_head.parameters()), lr=0.001)

    epochs = 20
    prev_gate_probs = None
    
    print("\nIniciando Treinamento V7 (PyTorch) no CIFAR-10...")
    for epoch in range(epochs):
        model.train()
        temperature = max(0.7, 3.0 * (0.85 ** epoch))
        
        t0 = time()
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.view(inputs.size(0), -1).to(device)
            labels = labels.to(device)
            
            optimizer_experts.zero_grad()
            optimizer_gate.zero_grad()
            
            out, loss, gate_probs, _, _ = model(inputs, temperature, labels)
            
            loss.backward()
            optimizer_experts.step()
            optimizer_gate.step()
            
            running_loss += loss.item()
            
        model.eval()
        correct = 0
        total = 0
        all_gate_probs = []
        all_expert_outs = []
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                inputs = inputs.view(inputs.size(0), -1).to(device)
                labels = labels.to(device)
                
                out, _, gate_probs, g2, expert_outs = model(inputs, temperature=0.7)
                
                _, predicted = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_gate_probs.append(gate_probs.cpu())
                all_expert_outs.append(expert_outs.cpu())
                
        val_acc = correct / total
        all_gate_probs = torch.cat(all_gate_probs, dim=0).numpy()
        all_expert_outs = torch.cat(all_expert_outs, dim=0)
        
        usage = all_gate_probs.mean(axis=0)
        gini = gini_index(usage)
        eri = compute_eri(all_expert_outs)
        
        rs = np.mean(np.abs(all_gate_probs - prev_gate_probs)) if prev_gate_probs is not None else 1.0
        prev_gate_probs = all_gate_probs.copy()
        
        print(f"Epoch {epoch+1:02d} | Val ACC: {val_acc:.4f} | ERI: {eri:.4f} | RS: {rs:.4f} | Gini: {gini:.4f} | Time: {time()-t0:.1f}s")
        
    print("\nUso Final dos Experts (Pruning):")
    print(np.round(usage, 3))
    
    print("\nGerando Visualizações Científicas SOTA...")
    os.makedirs('graficos', exist_ok=True)
    
    # Heatmap de uso
    plt.figure(figsize=(10, 6))
    sns.heatmap(all_gate_probs[:100], cmap='viridis', cbar=True)
    plt.title('Roteamento Contextual por Imagem (Amostra do CIFAR-10)')
    plt.xlabel('Especialistas')
    plt.ylabel('Imagens')
    plt.tight_layout()
    plt.savefig('graficos/v7_cifar_heatmap.png')
    plt.close()
    
    print("\nV7 CIFAR-10 Finalizado com sucesso. Próxima Etapa: Paper!")

if __name__ == "__main__":
    train_cifar()
