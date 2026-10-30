#!/usr/bin/env python3
import os
import struct
import numpy as np

def load_latest_weights(filepath="weights/ofuda_latest.bin"):
    if not os.path.exists(filepath):
        print(f"[Warning] No checkpoint found at {filepath}. Using random weights.")
        return np.random.normal(0, 0.02, (32, 64))
    
    with open(filepath, "rb") as f:
        nr, nc = struct.unpack("qq", f.read(16))
        data = f.read()
        weights = np.frombuffer(data, dtype=np.float32).reshape(nr, nc)
    print(f"[Loaded] Checkpoint weights loaded from {filepath}")
    return weights

def char_to_idx(c):
    if 'a' <= c <= 'z': return ord(c) - ord('a')
    if 'A' <= c <= 'Z': return ord(c) - ord('A') + 26
    if '0' <= c <= '9': return ord(c) - ord('0') + 52
    if c == ' ': return 62
    if c == '.': return 63
    return 0

def idx_to_char(idx):
    if idx < 26: return chr(ord('a') + idx)
    if idx < 52: return chr(ord('A') + (idx - 26))
    if idx < 62: return chr(ord('0') + (idx - 52))
    if idx == 62: return ' '
    if idx == 63: return '.'
    return '?'

def layer_norm(X):
    mean = np.mean(X, axis=1, keepdims=True)
    var = np.var(X, axis=1, keepdims=True)
    return (X - mean) / np.sqrt(var + 1e-5)

def causal_attention(Q, K, V):
    seq_len = Q.shape[0]
    d_k = Q.shape[1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
    scores += mask
    
    max_val = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - max_val)
    attn = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.dot(attn, V)

def sample_token(logits_row):
    max_val = np.max(logits_row)
    exp_logits = np.exp(logits_row - max_val)
    probs = exp_logits / np.sum(exp_logits)
    return int(np.argmax(probs))

def main():
    print("=== Ofuda Interactive Chat ===")
    print("Type your prompt and press Enter. Type 'exit' to quit.\n")
    
    # Initialize basic dummy weights for projection layers or load Wout checkpoint
    np.random.seed(42)
    W_emb = np.random.normal(0, 0.02, (64, 32))
    Wq = np.random.normal(0, 0.02, (32, 32))
    Wk = np.random.normal(0, 0.02, (32, 32))
    Wv = np.random.normal(0, 0.02, (32, 32))
    Wo = np.random.normal(0, 0.02, (32, 32))
    W1 = np.random.normal(0, 0.02, (32, 128))
    W2 = np.random.normal(0, 0.02, (128, 32))
    
    # Load trained output projection weights if available
    Wout = load_latest_weights("weights/ofuda_latest.bin")

    while True:
        try:
            prompt = input("Prompt> ")
            if prompt.lower() in ["exit", "quit"]:
                break
            if not prompt.strip():
                continue
            
            tokens = [char_to_idx(c) for c in prompt]
            max_new_tokens = 30
            
            print("Ofuda> ", end="", flush=True)
            for _ in range(max_new_tokens):
                seq_len = len(tokens)
                X = np.array([W_emb[t] for t in tokens])
                X = layer_norm(X)
                
                Q = np.dot(X, Wq)
                K = np.dot(X, Wk)
                V = np.dot(X, Wv)
                
                attn_out = causal_attention(Q, K, V)
                attn_proj = np.dot(attn_out, Wo)
                
                X2 = layer_norm(X + attn_proj)
                
                H = np.maximum(0, np.dot(X2, W1)) # ReLU
                ffn_out = np.dot(H, W2)
                
                X3 = layer_norm(X2 + ffn_out)
                
                logits = np.dot(X3, Wout)
                next_token = sample_token(logits[-1])
                
                tokens.append(next_token)
                print(idx_to_char(next_token), end="", flush=True)
            print("\n")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
