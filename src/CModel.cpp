#include "include/CModel.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

// ─── Utilitaires ─────────────────────────────────────────────────────────────

float relu(float x) { return x > 0.f ? x : 0.f; }
float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

static float f_fSum(const std::vector<float> &x) {
  float s = 0.f;
  for (float v : x)
    s += v;
  return s;
}

std::vector<float> softmax(std::vector<float> x) {
  if (x.empty())
    return x;
  float mx = *std::max_element(x.begin(), x.end());
  for (float &v : x)
    v = std::exp(v - mx);
  float s = f_fSum(x);
  for (float &v : x)
    v /= s;
  return x;
}

std::vector<float> add_vectors(const std::vector<float> &a,
                               const std::vector<float> &b) {
  std::vector<float> r(a.size());
  for (size_t i = 0; i < a.size(); ++i)
    r[i] = a[i] + b[i];
  return r;
}
std::vector<float> sub_vectors(const std::vector<float> &a,
                               const std::vector<float> &b) {
  std::vector<float> r(a.size());
  for (size_t i = 0; i < a.size(); ++i)
    r[i] = a[i] - b[i];
  return r;
}
std::vector<float> mul_vectors(const std::vector<float> &a,
                               const std::vector<float> &b) {
  std::vector<float> r(a.size());
  for (size_t i = 0; i < a.size(); ++i)
    r[i] = a[i] * b[i];
  return r;
}
std::vector<float> div_vectors(const std::vector<float> &a,
                               const std::vector<float> &b) {
  std::vector<float> r(a.size());
  for (size_t i = 0; i < a.size(); ++i)
    r[i] = a[i] / b[i];
  return r;
}

// ─── CModel ──────────────────────────────────────────────────────────────────

CModel::CModel(const std::vector<int> &layers, OutputMode mode)
    : layers_(layers), mode_(mode) {
  assert(layers_.size() >= 2 &&
         "Il faut au moins une couche d'entrée et une de sortie");
  F_vInitWeights();
}

void CModel::F_vInitWeights() {
  int L = static_cast<int>(layers_.size()) - 1; // nombre de matrices de poids
  weights_.resize(L);
  biases_.resize(L);

  std::mt19937 gen(std::random_device{}());
  for (int l = 0; l < L; ++l) {
    int in = layers_[l];
    int out = layers_[l + 1];
    // He initialization : std = sqrt(2 / fan_in)
    float std = std::sqrt(2.f / static_cast<float>(in));
    std::normal_distribution<float> dist(0.f, std);

    weights_[l].resize(out * in);
    for (float &w : weights_[l])
      w = dist(gen);
    biases_[l].assign(out, 0.f);
  }
}

// ── Forward pass ─────────────────────────────────────────────────────────────

CModel::ForwardResult
CModel::F_vPredict(const std::vector<float> &input) const {
  int L = static_cast<int>(layers_.size()) - 1;
  assert(static_cast<int>(input.size()) == layers_[0]);

  ForwardResult result;
  result.activations.resize(L + 1);
  result.preActivations.resize(L);

  result.activations[0] = input;

  for (int l = 0; l < L; ++l) {
    int in = layers_[l];
    int out = layers_[l + 1];

    // Z[l] = W[l] * A[l-1] + B[l]
    std::vector<float> &Z = result.preActivations[l];
    Z = biases_[l]; // initialise avec le biais
    cblas_sgemv(CblasRowMajor, CblasNoTrans, out, in, 1.f, weights_[l].data(),
                in, result.activations[l].data(), 1, 1.f, Z.data(), 1);

    // Activation : ReLU pour couches cachées, dépend du mode pour la sortie
    std::vector<float> &A = result.activations[l + 1];
    if (l < L - 1) {
      // Couche cachée → ReLU
      A.resize(out);
      for (int i = 0; i < out; ++i)
        A[i] = Z[i] > 0.f ? Z[i] : 0.f;
    } else if (mode_ == OutputMode::Classification) {
      // Classification → Softmax
      A = Z;
      float mx = *std::max_element(A.begin(), A.end());
      float sum = 0.f;
      for (float &v : A) {
        v = std::exp(v - mx);
        sum += v;
      }
      for (float &v : A)
        v /= sum;
    } else {
      // Régression → linéaire (identité) : A = Z
      A = Z;
    }
  }

  const auto &out_act = result.activations[L];
  int pred = static_cast<int>(std::max_element(out_act.begin(), out_act.end()) -
                              out_act.begin());
  result.predicted = pred;
  result.confidence =
      out_act[pred] * (mode_ == OutputMode::Classification ? 100.f : 1.f);
  return result;
}

// ── Backprop + mise à jour
// ──────────────────────────────────────────────────── Helper interne : calcule
// les deltas à partir du gradient de sortie fourni
static void
f_vBackprop(const CModel::ForwardResult &fwd, const std::vector<int> &layers,
            std::vector<std::vector<float>> weights, // copie pour cblas_sgemv
            std::vector<std::vector<float>> &weights_ref,
            std::vector<std::vector<float>> &biases_ref,
            const std::vector<float> &delta_out, float lr) {
  int L = static_cast<int>(layers.size()) - 1;
  std::vector<std::vector<float>> deltas(L);
  deltas[L - 1] = delta_out;

  for (int l = L - 2; l >= 0; --l) {
    int in = layers[l + 1];
    int out = layers[l + 2];
    deltas[l].assign(in, 0.f);
    cblas_sgemv(CblasRowMajor, CblasTrans, out, in, 1.f, weights[l + 1].data(),
                in, deltas[l + 1].data(), 1, 0.f, deltas[l].data(), 1);
    const auto &Z = fwd.preActivations[l];
    for (int i = 0; i < in; ++i)
      deltas[l][i] *= (Z[i] > 0.f ? 1.f : 0.f);
  }

  for (int l = 0; l < L; ++l) {
    int in = layers[l];
    int out_sz = layers[l + 1];
    cblas_sger(CblasRowMajor, out_sz, in, -lr, deltas[l].data(), 1,
               fwd.activations[l].data(), 1, weights_ref[l].data(), in);
    for (int i = 0; i < out_sz; ++i)
      biases_ref[l][i] -= lr * deltas[l][i];
  }
}

// Classification : trueLabel = indice de la classe correcte
void CModel::F_vTrain(const std::vector<float> &input, int trueLabel,
                      float learningRate) {
  auto fwd = F_vPredict(input);
  int L = static_cast<int>(layers_.size()) - 1;

  // dL/dZ_out = softmax(out) - one_hot(trueLabel)
  std::vector<float> delta_out(layers_[L]);
  for (int i = 0; i < layers_[L]; ++i)
    delta_out[i] = fwd.activations[L][i] - (i == trueLabel ? 1.f : 0.f);

  f_vBackprop(fwd, layers_, weights_, weights_, biases_, delta_out,
              learningRate);
}

// Régression : target = valeurs continues souhaitées
void CModel::F_vTrain(const std::vector<float> &input,
                      const std::vector<float> &target, float learningRate) {
  auto fwd = F_vPredict(input);
  int L = static_cast<int>(layers_.size()) - 1;

  // dL/dZ_out = output - target  (MSE, activation linéaire → dérivée = 1)
  std::vector<float> delta_out(layers_[L]);
  for (int i = 0; i < layers_[L]; ++i)
    delta_out[i] = fwd.activations[L][i] - target[i];

  f_vBackprop(fwd, layers_, weights_, weights_, biases_, delta_out,
              learningRate);
}

// ── Sauvegarde ───────────────────────────────────────────────────────────────
//
// Format binaire :
//   [int]   nb_layers
//   [int]*  layers[0..N-1]
//   [int]   mode  (0 = Classification, 1 = Regression)
//   Pour chaque couche l = 0..N-2 :
//     [float]* weights_[l]   (out * in floats)
//     [float]* biases_[l]    (out floats)
//
void CModel::F_vSave(const std::string &filename) const {
  std::ofstream f(filename, std::ios::binary);
  if (!f)
    throw std::runtime_error("Impossible d'ouvrir " + filename);

  int nb = static_cast<int>(layers_.size());
  f.write(reinterpret_cast<const char *>(&nb), sizeof(int));
  f.write(reinterpret_cast<const char *>(layers_.data()), nb * sizeof(int));

  int mode_int = static_cast<int>(mode_);
  f.write(reinterpret_cast<const char *>(&mode_int), sizeof(int));

  for (int l = 0; l < nb - 1; ++l) {
    f.write(reinterpret_cast<const char *>(weights_[l].data()),
            weights_[l].size() * sizeof(float));
    f.write(reinterpret_cast<const char *>(biases_[l].data()),
            biases_[l].size() * sizeof(float));
  }
}

// ── CModelLoader ─────────────────────────────────────────────────────────────

CModelLoader::CModelLoader(const std::string &filename) : filename_(filename) {}

CModel CModelLoader::F_vLoad() {
  std::ifstream f(filename_, std::ios::binary);
  if (!f)
    throw std::runtime_error("Impossible d'ouvrir " + filename_);

  int nb;
  f.read(reinterpret_cast<char *>(&nb), sizeof(int));

  std::vector<int> layers(nb);
  f.read(reinterpret_cast<char *>(layers.data()), nb * sizeof(int));

  // Lecture du mode (absent dans les anciens fichiers → défaut Classification)
  int mode_int = 0;
  f.read(reinterpret_cast<char *>(&mode_int), sizeof(int));
  OutputMode mode = static_cast<OutputMode>(mode_int);

  CModel model(layers, mode);

  for (int l = 0; l < nb - 1; ++l) {
    f.read(reinterpret_cast<char *>(model.weights_[l].data()),
           model.weights_[l].size() * sizeof(float));
    f.read(reinterpret_cast<char *>(model.biases_[l].data()),
           model.biases_[l].size() * sizeof(float));
  }
  return model;
}