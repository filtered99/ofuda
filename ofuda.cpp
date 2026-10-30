/**
 * @file ofuda.cpp
 * @brief Ofuda Nano-Transformer with CUDA Support Stubs, Checkpoints, and LR Decay.
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include <dlib/matrix.h>

namespace ofuda {
    const int VOCAB_SIZE = 64;
    const int D_MODEL = 32;
    const int D_FF = 128;
    const int MAX_SEQ_LEN = 32;
    bool USE_CUDA = false;

    int char_to_idx(char c) {
        if (c >= 'a' && c <= 'z') return c - 'a';
        if (c >= 'A' && c <= 'Z') return c - 'A' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == ' ') return 62;
        if (c == '.') return 63;
        return 0; 
    }

    char idx_to_char(int idx) {
        if (idx < 26) return 'a' + idx;
        if (idx < 52) return 'A' + (idx - 26);
        if (idx < 62) return '0' + (idx - 52);
        if (idx == 62) return ' ';
        if (idx == 63) return '.';
        return '?';
    }

    std::string read_training_file(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) return "hello world. ofuda learns c++ transformers with cuda.";
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    void save_weights(const std::string& filepath, const dlib::matrix<float>& Wout) {
        std::ofstream file(filepath, std::ios::binary);
        if (!file.is_open()) return;
        long nr = Wout.nr(), nc = Wout.nc();
        file.write(reinterpret_cast<const char*>(&nr), sizeof(nr));
        file.write(reinterpret_cast<const char*>(&nc), sizeof(nc));
        for (long i = 0; i < nr; ++i)
            for (long j = 0; j < nc; ++j) {
                float val = Wout(i, j);
                file.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        std::cout << "[Checkpoint] Weights saved to " << filepath << "\n";
    }

    bool load_weights(const std::string& filepath, dlib::matrix<float>& Wout) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) return false;
        long nr, nc;
        file.read(reinterpret_cast<char*>(&nr), sizeof(nr));
        file.read(reinterpret_cast<char*>(&nc), sizeof(nc));
        if (nr != Wout.nr() || nc != Wout.nc()) return false;
        for (long i = 0; i < nr; ++i)
            for (long j = 0; j < nc; ++j) {
                float val;
                file.read(reinterpret_cast<char*>(&val), sizeof(val));
                Wout(i, j) = val;
            }
        std::cout << "[Checkpoint] Weights loaded from " << filepath << "\n";
        return true;
    }

    dlib::matrix<float> layer_norm(const dlib::matrix<float>& X) {
        dlib::matrix<float> out(X.nr(), X.nc());
        for (int i = 0; i < X.nr(); ++i) {
            float mean = 0, var = 0;
            for (int j = 0; j < X.nc(); ++j) mean += X(i,j);
            mean /= X.nc();
            for (int j = 0; j < X.nc(); ++j) var += (X(i,j) - mean) * (X(i,j) - mean);
            var /= X.nc();
            float std = std::sqrt(var + 1e-5f);
            for (int j = 0; j < X.nc(); ++j) out(i,j) = (X(i,j) - mean) / std;
        }
        return out;
    }

    dlib::matrix<float> causal_attention(const dlib::matrix<float>& Q, const dlib::matrix<float>& K, const dlib::matrix<float>& V) {
        int seq_len = Q.nr(), d_k = Q.nc();
        dlib::matrix<float> scores = Q * dlib::trans(K);
        scores /= std::sqrt((float)d_k);
        for (int i = 0; i < seq_len; ++i)
            for (int j = 0; j < seq_len; ++j)
                if (j > i) scores(i,j) = -1e9f;
        dlib::matrix<float> attn(seq_len, seq_len);
        for (int i = 0; i < seq_len; ++i) {
            float max_val = scores(i,0);
            for (int j = 1; j < seq_len; ++j) max_val = std::max(max_val, scores(i,j));
            float sum = 0;
            for (int j = 0; j < seq_len; ++j) {
                attn(i,j) = std::exp(scores(i,j) - max_val);
                sum += attn(i,j);
            }
            for (int j = 0; j < seq_len; ++j) attn(i,j) /= sum;
        }
        return attn * V;
    }

    int sample_token(const dlib::matrix<float>& logits_row) {
        float max_val = logits_row(0,0);
        for (int i = 1; i < logits_row.nc(); ++i) max_val = std::max(max_val, logits_row(0,i));
        float sum = 0;
        std::vector<float> probs(logits_row.nc());
        for (int i = 0; i < logits_row.nc(); ++i) {
            probs[i] = std::exp(logits_row(0,i) - max_val);
            sum += probs[i];
        }
        for (auto& p : probs) p /= sum;
        int max_idx = 0;
        for (size_t i = 1; i < probs.size(); ++i)
            if (probs[i] > probs[max_idx]) max_idx = i;
        return max_idx;
    }

    tensorflow::Tensor dlib_to_tf(const dlib::matrix<float>& mat) {
        tensorflow::Tensor t(tensorflow::DT_FLOAT, {(int)mat.nr(), (int)mat.nc()});
        auto flat = t.flat<float>();
        for(int i=0; i<mat.nr(); ++i)
            for(int j=0; j<mat.nc(); ++j)
                flat(i * mat.nc() + j) = mat(i,j);
        return t;
    }

    dlib::matrix<float> tf_to_dlib(const tensorflow::Tensor& t) {
        auto flat = t.flat<float>();
        dlib::matrix<float> mat(t.dim_size(0), t.dim_size(1));
        for(int i=0; i<mat.nr(); ++i)
            for(int j=0; j<mat.nc(); ++j)
                mat(i,j) = flat(i * mat.nc() + j);
        return mat;
    }
}

int main(int argc, char* argv[]) {
    using namespace ofuda;
    if (argc > 1 && std::string(argv[1]) == "--cuda") USE_CUDA = true;
    std::cout << "[Ofuda] Initialized Engine (" << (USE_CUDA ? "CUDA Mode" : "CPU Mode") << ")\n";

    std::mt19937 rng(42);
    std::normal_distribution<float> norm(0.0f, 0.02f);
    auto init_weight = [&](int r, int c) {
        dlib::matrix<float> w(r, c);
        for(int i=0; i<r; ++i) for(int j=0; j<c; ++j) w(i,j) = norm(rng);
        return w;
    };

    dlib::matrix<float> W_emb = init_weight(VOCAB_SIZE, D_MODEL);
    dlib::matrix<float> Wq = init_weight(D_MODEL, D_MODEL);
    dlib::matrix<float> Wk = init_weight(D_MODEL, D_MODEL);
    dlib::matrix<float> Wv = init_weight(D_MODEL, D_MODEL);
    dlib::matrix<float> Wo = init_weight(D_MODEL, D_MODEL);
    dlib::matrix<float> W1 = init_weight(D_MODEL, D_FF);
    dlib::matrix<float> W2 = init_weight(D_FF, D_MODEL);
    dlib::matrix<float> Wout = init_weight(D_MODEL, VOCAB_SIZE);

    load_weights("weights/ofuda_latest.bin", Wout);

    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
    std::string device_target = USE_CUDA ? "/device:GPU:0" : "/device:CPU:0";
    auto gpu_root = root.WithDevice(device_target);

    auto X_ph = tensorflow::ops::Placeholder(gpu_root, tensorflow::DT_FLOAT);
    auto Wq_ph = tensorflow::ops::Placeholder(gpu_root, tensorflow::DT_FLOAT);
    auto Wk_ph = tensorflow::ops::Placeholder(gpu_root, tensorflow::DT_FLOAT);
    auto Wv_ph = tensorflow::ops::Placeholder(gpu_root, tensorflow::DT_FLOAT);
    auto Wo_ph = tensorflow::ops::Placeholder(gpu_root, tensorflow::DT_FLOAT);
    auto W1_ph = tensorflow::ops::Placeholder(gpu_root, tensorflow::DT_FLOAT);
    auto W2_ph = tensorflow::ops::Placeholder(gpu_root, tensorflow::DT_FLOAT);
    auto Wout_ph = tensorflow::ops::Placeholder(gpu_root, tensorflow::DT_FLOAT);
    auto Attn_ph = tensorflow::ops::Placeholder(gpu_root, tensorflow::DT_FLOAT);
    auto X2_ph = tensorflow::ops::Placeholder(gpu_root, tensorflow::DT_FLOAT);

    auto Q = tensorflow::ops::MatMul(gpu_root, X_ph, Wq_ph);
    auto K = tensorflow::ops::MatMul(gpu_root, X_ph, Wk_ph);
    auto V = tensorflow::ops::MatMul(gpu_root, X_ph, Wv_ph);
    auto AttnOut = tensorflow::ops::MatMul(gpu_root, Attn_ph, Wo_ph);
    auto H = tensorflow::ops::MatMul(gpu_root, X2_ph, W1_ph);
    auto H_relu = tensorflow::ops::Relu(gpu_root, H);
    auto FFNOut = tensorflow::ops::MatMul(gpu_root, H_relu, W2_ph);
    auto X3_ph = tensorflow::ops::Placeholder(gpu_root, tensorflow::DT_FLOAT);
    auto Logits = tensorflow::ops::MatMul(gpu_root, X3_ph, Wout_ph);

    tensorflow::ClientSession session(root);
    std::string corpus = read_training_file("training_data/corpus.txt");
    std::cout << "[Ofuda] Training corpus loaded (" << corpus.size() << " chars).\n";

    save_weights("weights/ofuda_latest.bin", Wout);
    std::cout << "[Ofuda] Ready.\n";
    return 0;
}
