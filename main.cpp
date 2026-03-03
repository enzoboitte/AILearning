#include "src/include/CEntraineur.hpp"
#include <iostream>

int main() {
  try {
    CEntraineur entraineur("entrainement/dataset.json");
    entraineur.F_vLancerEntrainement();
  } catch (const std::exception &e) {
    std::cerr << "Erreur : " << e.what() << std::endl;
    return 1;
  }
  return 0;
}