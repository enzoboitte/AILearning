#ifndef CENTRAINEUR_HPP
#define CENTRAINEUR_HPP

#include "CModel.hpp"
#include <fstream>
#include <iostream>
#include <json/json.h>
#include <random>
#include <string>
#include <vector>

class CEntraineur {
public:
  CEntraineur(const std::string &l_sDatasetPath);
  ~CEntraineur();

  void F_vLancerEntrainement();
  float F_fGetTauxApprentissage();
  void F_vUpdateTauxApprentissage(float l_fLoss);

private:
  void F_vChargerImagePGM(const std::string &l_sChemin,
                          std::vector<float> &l_lPixels);
  void F_vChargerDataset(const std::string &l_sJsonPath);
  void F_vAugmenter(const std::vector<float> &l_lSrc,
                    std::vector<float> &l_lDst);

  std::vector<std::pair<std::vector<float>, int>> g_lDonneesEntrainement;

  // Paramètres globaux
  float g_fTauxApprentissage = 0.005f;
  float g_fLastMinLoss = 100.0f;
  int g_iNbEpochsSinceLastMinLoss = 0;

  size_t g_iNbEpochs = 0;
  size_t g_iSaveStep = 50;
  std::mt19937 g_genRandom;
  std::uniform_int_distribution<> g_distShuffle;
};

#endif
