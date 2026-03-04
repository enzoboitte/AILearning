#ifndef CENTRAINEUR_HPP
#define CENTRAINEUR_HPP

#include <json/json.h>
#include <random>
#include <string>
#include <vector>

class CEntraineur {
public:
  // layers = { inputSize, hidden1, ..., outputSize }
  // Si layers est vide, une topologie par défaut est choisie.
  CEntraineur(const std::string &l_sDatasetPath,
              const std::vector<int> &layers = {});
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

  // Topologie du réseau (stockée pour créer le modèle après chargement du
  // dataset)
  std::vector<int> g_vLayers;

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
