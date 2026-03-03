#ifndef CMODEL_HPP
#define CMODEL_HPP

#include <cblas.h>
#include <string>
#include <tuple>
#include <vector>

float relu(float x);
float sigmoid(float x);
std::vector<float> softmax(std::vector<float> x);
std::vector<float> add_vectors(const std::vector<float> &a,
                               const std::vector<float> &b);
std::vector<float> sub_vectors(const std::vector<float> &a,
                               const std::vector<float> &b);
std::vector<float> mul_vectors(const std::vector<float> &a,
                               const std::vector<float> &b);
std::vector<float> div_vectors(const std::vector<float> &a,
                               const std::vector<float> &b);

class CModel {
public:
  CModel(int l_iNbC1, int l_iNbC2, int l_iInputSize);
  CModel();
  ~CModel() = default;

  std::tuple<float, float, std::vector<float>, std::vector<float>,
             std::vector<float>>
  F_vPredict(const std::vector<float> &l_vInput);

  void F_vTrain(const std::vector<float> &l_vInput, int l_iTrueLabel,
                float l_fLearningRate);
  void F_vSave(const std::string &l_sFileName);

  int g_iNbC1 = 0;
  int g_iNbC2 = 0;
  std::vector<float> g_vW1_flat; // [NbC1 x 4096]
  std::vector<float> g_vW2_flat; // [NbC2 x NbC1]
  std::vector<float> g_vB1, g_vB2;

private:
  // buffers réutilisables
  mutable std::vector<float> g_vZ1, g_vA1, g_vZ2, g_vA2;
  int G_iInputSize;
};

class CModelLoader {
public:
  explicit CModelLoader(const std::string &l_sFileName);
  CModel F_vLoad();

private:
  std::string g_sFileName;
};

#endif