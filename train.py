#!/usr/bin/env python3
import os
import struct
import numpy as np

def char_to_idx(c):
    if 'a' <= c <= 'z': return ord(c) - ord('a')
    if 'A' <= c <= 'Z': return ord(c) - ord('A') + 26
    if '0' <= c <= '9': return ord(c) - ord('0') + 52
    if c == ' ': return 62
    if c == '.': return 63
    return 0

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

def save_weights(filepath, Wout):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        nr, nc = Wout.shape
        f.write(struct.pack("qq", nr, nc))
        f.write(Wout.astype(np.float32).tobytes())
    print(f"[Checkpoint] Weights saved to {filepath}")

def main():
    print("=== Ofuda Python Trainer ===")
    os.makedirs("weights", exist_ok=True)
    
    corpus_path = "training_data/corpus.txt"
    if os.path.exists(corpus_path):
        with open(corpus_path, "r") as f:
            corpus = f.read()
    else:
        corpus = "hello world. ofuda learns c++ transformers."

    split_idx = int(len(corpus) * 0.8)
    train_text = corpus[:split_idx]
    val_text = corpus[split_idx:] if split_idx < len(corpus) else train_text

    VOCAB_SIZE, D_MODEL, D_FF = 64, 32, 128
    np.random.seed(42)
    
    W_emb = np.random.normal(0, 0.02, (VOCAB_SIZE, D_MODEL))
    Wq = np.random.normal(0, 0.02, (D_MODEL, D_MODEL))
    Wk = np.random.normal(0, 0.02, (D_MODEL, D_MODEL))
    Wv = np.random.normal(0, 0.02, (D_MODEL, D_MODEL))
    Wo = np.random.normal(0, 0.02, (D_MODEL, D_MODEL))
    W1 = np.random.normal(0, 0.02, (D_MODEL, D_FF))
    W2 = np.random.normal(0, 0.02, (D_FF, D_MODEL))
    Wout = np.random.normal(0, 0.02, (D_MODEL, VOCAB_SIZE))

    learning_rate = 0.005
    lr_decay = 0.95
    epochs = 10

    train_tokens = [char_to_idx(c) for c in train_text]
    val_tokens = [char_to_idx(c) for c in val_text]

    for epoch in range(epochs):
        total_train_loss = 0.0
        for step in range(len(train_tokens) - 1):
            seq_len = step + 1
            X = np.array([W_emb[train_tokens[i]] for i in range(seq_len)])
            X = layer_norm(X)
            Q = np.dot(X, Wq)
            K = np.dot(X, Wk)
            V = np.dot(X, Wv)
            attn_out = causal_attention(Q, K, V)
            attn_proj = np.dot(attn_out, Wo)
            X2 = layer_norm(X + attn_proj)
            H = np.maximum(0, np.dot(X2, W1))
            ffn_out = np.dot(H, W2)
            X3 = layer_norm(X2 + ffn_out)
            
            logits = np.dot(X3, Wout)
            last_logit = logits[-1]
            target = train_tokens[step + 1]
            max_l = np.max(last_logit)
            probs = np.exp(last_logit - max_l) / np.sum(np.exp(last_logit - max_l))
            
            loss = -np.log(max(probs[target], 1e-15))
            total_train_loss += loss
            
            error = probs.copy()
            error[target] -= 1.0
            for j in range(VOCAB_SIZE):
                for d in range(D_MODEL):
                    Wout[d, j] -= learning_rate * error[j] * X3[-1, d]

        avg_train_loss = total_train_loss / max(1, len(train_tokens) - 1)
        
        total_val_loss = 0.0
        for step in range(len(val_tokens) - 1):
            seq_len = step + 1
            X = np.array([W_emb[val_tokens[i]] for i in range(seq_len)])
            X = layer_norm(X)
            Q = np.dot(X, Wq)
            K = np.dot(X, Wk)
            V = np.dot(X, Wv)
            attn_out = causal_attention(Q, K, V)
            attn_proj = np.dot(attn_out, Wo)
            X2 = layer_norm(X + attn_proj)
            H = np.maximum(0, np.dot(X2, W1))
            ffn_out = np.dot(H, W2)
            X3 = layer_norm(X2 + ffn_out)
            logits = np.dot(X3, Wout)
            last_logit = logits[-1]
            target = val_tokens[step + 1]
            max_l = np.max(last_logit)
            probs = np.exp(last_logit - max_l) / np.sum(np.exp(last_logit - max_l))
            total_val_loss += -np.log(max(probs[target], 1e-15))

        avg_val_loss = total_val_loss / max(1, len(val_tokens) - 1)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {learning_rate:.5f}")
        
        save_weights("weights/ofuda_latest.bin", Wout)
        save_weights(f"weights/ofuda_epoch_{epoch+1}.bin", Wout)
        
        learning_rate *= lr_decay

    print("\n[Training Complete] Weights saved successfully.")

if __name__ == "__main__":
    main()
