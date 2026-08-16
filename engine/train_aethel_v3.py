import os
import torch
import time
import tiktoken
from datasets import load_dataset
from safetensors.torch import save_file

# Importamos la arquitectura V3 auditada y de bajo nivel
from aethel_model import AethelModel, AethelConfig

print("===================================================================")
print(" 🚀 ENTRENAMIENTO AETHEL V3 (FRONTERA MULTILINGÜE)")
print("===================================================================\n")

# 1. CONFIGURACIÓN OPTIMIZADA PARA KAGGLE / COLAB
config = AethelConfig()
config.vocab_size = 100288  # tiktoken cl100k_base + tokens especiales alineado a 32
config.dim = 512            
config.n_layers = 8         
config.n_heads = 8          
config.n_kv_heads = 4       # GQA Activado
config.n_experts = 4        
config.active_experts = 2   
config.max_seq_len = 256    

batch_size = 4
grad_accum_steps = 16
learning_rate = 3e-4
max_iters = 15000

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[*] Dispositivo de cómputo: {device.upper()}")

# 2. CARGA DE DATOS MULTILINGÜES (Tu pipeline intacto)
print("[*] Cargando Tokenizador Multilingüe cl100k_base...")
tokenizer = tiktoken.get_encoding("cl100k_base")

corpus_textos = [
    "Definición: La inteligencia artificial (IA) es la simulación de procesos de inteligencia humana por parte de máquinas.",
    "Dictionary: Artificial Intelligence - The capability of a computer system to emulate human intelligence and logic.",
    # (El resto de tu corpus va aquí...)
]

token_buffer = []
for t in corpus_textos:
    token_buffer.extend(tokenizer.encode(t))
    token_buffer.append(tokenizer.eot_token)

# Intentar cargar wikipedia (streaming)
try:
    print("[*] Descargando fuentes multilingües de Wikipedia...")
    ds_es = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)
    iter_es = iter(ds_es)
    while len(token_buffer) < 1_500_000:
        art_es = next(iter_es)
        token_buffer.extend(tokenizer.encode(art_es['text'][:2000]))
        token_buffer.append(tokenizer.eot_token)
except Exception as e:
    print(f"[!] Aviso: No se pudo cargar Wikipedia, usando corpus base. {e}")

data_tensor = torch.tensor(token_buffer, dtype=torch.long)
print(f"[+] Corpus preparado. Total tokens en memoria: {len(data_tensor):,}")

def get_batch():
    ix = torch.randint(len(data_tensor) - config.max_seq_len, (batch_size,))
    x = torch.stack([data_tensor[i:i+config.max_seq_len] for i in ix])
    y = torch.stack([data_tensor[i+1:i+config.max_seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# 3. ENTRENAMIENTO CON ARQUITECTURA V3
torch.cuda.empty_cache()
model = AethelModel(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

print(f"\n[*] INICIANDO ENTRENAMIENTO V3 EN {device.upper()}")
start_time = time.time()

for iter_num in range(max_iters):
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    aux_loss_accum = 0.0
    
    for _ in range(grad_accum_steps):
        xb, yb = get_batch()
        
        if device == 'cuda':
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits, aux_loss, _ = model(xb)
                B, T, C = logits.shape
                ce_loss = torch.nn.functional.cross_entropy(logits.view(B*T, C), yb.view(B*T))
                total_loss = (ce_loss + 0.01 * aux_loss) / grad_accum_steps
            scaler.scale(total_loss).backward()
        else:
            logits, aux_loss, _ = model(xb)
            B, T, C = logits.shape
            ce_loss = torch.nn.functional.cross_entropy(logits.view(B*T, C), yb.view(B*T))
            total_loss = (ce_loss + 0.01 * aux_loss) / grad_accum_steps
            total_loss.backward()
        
        loss_accum += ce_loss.item() / grad_accum_steps
        aux_loss_accum += aux_loss.item() / grad_accum_steps
        
    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    if iter_num % 100 == 0:
        elapsed = (time.time() - start_time) / 60
        print(f"Paso {iter_num:5d}/{max_iters} | CE Loss: {loss_accum:.4f} | MoE Aux: {aux_loss_accum:.4f} | Tiempo: {elapsed:.2f} mins")

print("\n[*] Entrenamiento Completado.")

# 4. EXPORTAR PESOS A SAFETENSORS (Filtrando tensores duplicados para evitar RuntimeError)
try:
    state_dict_limpio = {}
    for k, v in model.state_dict().items():
        if k == 'tok_embeddings.weight':
            continue
        state_dict_limpio[k] = v.contiguous()
    save_file(state_dict_limpio, 'aethel_v3_pesos.safetensors')
    print("✅ Pesos exportados exitosamente a 'aethel_v3_pesos.safetensors'")
except Exception as e:
    print(f"[!] Error al exportar: {e}")
