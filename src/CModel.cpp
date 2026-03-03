#include "include/CModel.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

std::vector<std::string> split(const std::string &l_sString,
                               const std::string &l_sDelimiter) {
  std::vector<std::string> l_lTokens;
  std::string l_sToken;
  size_t l_iPos = 0;
  while ((l_iPos = l_sString.find(l_sDelimiter)) != std::string::npos) {
    l_sToken = l_sString.substr(0, l_iPos);
    l_lTokens.push_back(l_sToken);
  }
  l_lTokens.push_back(l_sString);
  return l_lTokens;
}

float relu(float x) { return x > 0 ? x : 0; }
float sigmoid(float x) { return 1 / (1 + std::exp(-x)); }
float sum(const std::vector<float> &x) {
  float sum = 0;
  for (const float &val : x) {
    sum += val;
  }
  return sum;
}

std::vector<float> softmax(std::vector<float> x) {
  if (x.empty())
    return x;
  float val_max = *std::max_element(x.begin(), x.end());
  std::vector<float> exps;
  for (float &val : x) {
    exps.push_back(std::exp(val - val_max));
  }
  float sum_exps = sum(exps);
  for (float &val : exps) {
    val /= sum_exps;
  }
  return exps;
}

std::vector<float> add_vectors(const std::vector<float> &a,
                               const std::vector<float> &b) {
  std::vector<float> result;
  for (size_t i = 0; i < a.size(); i++) {
    result.push_back(a[i] + b[i]);
  }
  return result;
}

std::vector<float> sub_vectors(const std::vector<float> &a,
                               const std::vector<float> &b) {
  std::vector<float> result;
  for (size_t i = 0; i < a.size(); i++) {
    result.push_back(a[i] - b[i]);
  }
  return result;
}

std::vector<float> mul_vectors(const std::vector<float> &a,
                               const std::vector<float> &b) {
  std::vector<float> result;
  for (size_t i = 0; i < a.size(); i++) {
    result.push_back(a[i] * b[i]);
  }
  return result;
}

std::vector<float> div_vectors(const std::vector<float> &a,
                               const std::vector<float> &b) {
  std::vector<float> result;
  for (size_t i = 0; i < a.size(); i++) {
    result.push_back(a[i] / b[i]);
  }
  return result;
}

std::vector<float>
product_matrix_vector(const std::vector<std::vector<float>> &a,
                      const std::vector<float> &b) {
  std::vector<float> result;
  for (size_t i = 0; i < a.size(); i++) {
    std::vector<float> l_lVal;
    for (size_t j = 0; j < b.size(); j++) {
      l_lVal.push_back(a[i][j] * b[j]);
    }
    result.push_back(sum(l_lVal));
  }
  return result;
}

static void f_vSoftmaxInplace(std::vector<float> &x) {
  float l_fMax = *std::max_element(x.begin(), x.end());
  float l_fSum = 0.f;
  for (float &v : x) {
    v = std::exp(v - l_fMax);
    l_fSum += v;
  }
  for (float &v : x)
    v /= l_fSum;
}

CModel::CModel(int l_iNbC1, int l_iNbC2, int l_iInputSize)
    : g_iNbC1(l_iNbC1), g_iNbC2(l_iNbC2), g_vW1_flat(l_iNbC1 * l_iInputSize),
      G_iInputSize(l_iInputSize), g_vW2_flat(l_iNbC2 * l_iNbC1),
      g_vB1(l_iNbC1, 0.f), g_vB2(l_iNbC2, 0.f), g_vZ1(l_iNbC1), g_vA1(l_iNbC1),
      g_vZ2(l_iNbC2), g_vA2(l_iNbC2) {
  std::mt19937 l_gen(std::random_device{}());
  float l_fStd1 = std::sqrt(2.f / G_iInputSize); // He init
  float l_fStd2 = std::sqrt(2.f / l_iNbC1);
  std::normal_distribution<float> l_dist1(0.f, l_fStd1);
  std::normal_distribution<float> l_dist2(0.f, l_fStd2);
  for (float &w : g_vW1_flat)
    w = l_dist1(l_gen);
  for (float &w : g_vW2_flat)
    w = l_dist2(l_gen);
}

CModel::CModel() {}

std::tuple<float, float, std::vector<float>, std::vector<float>,
           std::vector<float>>
CModel::F_vPredict(const std::vector<float> &l_vInput) {
  assert(static_cast<int>(l_vInput.size()) == G_iInputSize &&
         "Taille input != G_iInputSize !");

  // Z1 = W1 * input + B1
  g_vZ1 = g_vB1;
  cblas_sgemv(CblasRowMajor, CblasNoTrans, g_iNbC1, G_iInputSize, 1.f,
              g_vW1_flat.data(), G_iInputSize, l_vInput.data(), 1, 1.f,
              g_vZ1.data(), 1);

  // A1 = ReLU(Z1)
  for (int i = 0; i < g_iNbC1; ++i)
    g_vA1[i] = g_vZ1[i] > 0.f ? g_vZ1[i] : 0.f;

  // Z2 = W2 * A1 + B2
  g_vZ2 = g_vB2;
  cblas_sgemv(CblasRowMajor, CblasNoTrans, g_iNbC2, g_iNbC1, 1.f,
              g_vW2_flat.data(), g_iNbC1, g_vA1.data(), 1, 1.f, g_vZ2.data(),
              1);

  // A2 = softmax(Z2)
  g_vA2 = g_vZ2;
  f_vSoftmaxInplace(g_vA2);

  int l_iMaxIdx = std::max_element(g_vA2.begin(), g_vA2.end()) - g_vA2.begin();
  return {static_cast<float>(l_iMaxIdx), g_vA2[l_iMaxIdx] * 100.f, g_vZ1, g_vA1,
          g_vA2};
}

void CModel::F_vTrain(const std::vector<float> &l_vInput, int l_iTrueLabel,
                      float l_fLearningRate) {
  auto [l_fPred, l_fConf, l_vZ1, l_vA1, l_vA2] = F_vPredict(l_vInput);

  // dL/dZ2 = softmax - one_hot
  std::vector<float> l_vDZ2(g_iNbC2);
  for (int i = 0; i < g_iNbC2; ++i)
    l_vDZ2[i] = l_vA2[i] - (i == l_iTrueLabel ? 1.f : 0.f);

  // dL/dA1 = W2^T * dZ2
  std::vector<float> l_vDA1(g_iNbC1, 0.f);
  cblas_sgemv(CblasRowMajor, CblasTrans, g_iNbC2, g_iNbC1, 1.f,
              g_vW2_flat.data(), g_iNbC1, l_vDZ2.data(), 1, 0.f, l_vDA1.data(),
              1);

  // dL/dZ1 = dA1 * ReLU'(Z1)
  std::vector<float> l_vDZ1(g_iNbC1);
  for (int i = 0; i < g_iNbC1; ++i)
    l_vDZ1[i] = l_vA1[i] > 0.f ? l_vDA1[i] : 0.f;

  // Maj W2 -= lr * dZ2 ⊗ A1
  cblas_sger(CblasRowMajor, g_iNbC2, g_iNbC1, -l_fLearningRate, l_vDZ2.data(),
             1, l_vA1.data(), 1, g_vW2_flat.data(), g_iNbC1);

  // Maj B2
  for (int i = 0; i < g_iNbC2; ++i)
    g_vB2[i] -= l_fLearningRate * l_vDZ2[i];

  // Maj W1 -= lr * dZ1 ⊗ input
  cblas_sger(CblasRowMajor, g_iNbC1, G_iInputSize, -l_fLearningRate,
             l_vDZ1.data(), 1, l_vInput.data(), 1, g_vW1_flat.data(),
             G_iInputSize);

  // Maj B1
  for (int i = 0; i < g_iNbC1; ++i)
    g_vB1[i] -= l_fLearningRate * l_vDZ1[i];
}

void CModel::F_vSave(const std::string &l_sFileName) {
  std::ofstream l_file(l_sFileName, std::ios::binary);
  if (!l_file)
    throw std::runtime_error("Impossible d'ouvrir " + l_sFileName);
  l_file.write(reinterpret_cast<const char *>(&g_iNbC1), sizeof(int));
  l_file.write(reinterpret_cast<const char *>(&g_iNbC2), sizeof(int));
  l_file.write(reinterpret_cast<const char *>(&G_iInputSize), sizeof(int));
  l_file.write(reinterpret_cast<const char *>(g_vW1_flat.data()),
               g_vW1_flat.size() * sizeof(float));
  l_file.write(reinterpret_cast<const char *>(g_vB1.data()),
               g_vB1.size() * sizeof(float));
  l_file.write(reinterpret_cast<const char *>(g_vW2_flat.data()),
               g_vW2_flat.size() * sizeof(float));
  l_file.write(reinterpret_cast<const char *>(g_vB2.data()),
               g_vB2.size() * sizeof(float));
}

CModelLoader::CModelLoader(const std::string &l_sFileName)
    : g_sFileName(l_sFileName) {}

CModel CModelLoader::F_vLoad() {
  std::ifstream l_file(g_sFileName, std::ios::binary);
  if (!l_file)
    throw std::runtime_error("Impossible d'ouvrir " + g_sFileName);
  int l_iNbC1, l_iNbC2, l_iInputSize;
  l_file.read(reinterpret_cast<char *>(&l_iNbC1), sizeof(int));
  l_file.read(reinterpret_cast<char *>(&l_iNbC2), sizeof(int));
  l_file.read(reinterpret_cast<char *>(&l_iInputSize), sizeof(int));

  CModel l_cModel(l_iNbC1, l_iNbC2, l_iInputSize);
  l_file.read(reinterpret_cast<char *>(l_cModel.g_vW1_flat.data()),
              l_cModel.g_vW1_flat.size() * sizeof(float));
  l_file.read(reinterpret_cast<char *>(l_cModel.g_vB1.data()),
              l_cModel.g_vB1.size() * sizeof(float));
  l_file.read(reinterpret_cast<char *>(l_cModel.g_vW2_flat.data()),
              l_cModel.g_vW2_flat.size() * sizeof(float));
  l_file.read(reinterpret_cast<char *>(l_cModel.g_vB2.data()),
              l_cModel.g_vB2.size() * sizeof(float));
  return l_cModel;
}