package dk.alexandra.fresco.ml.svm;

import java.util.List;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.value.SInt;

public class SVMModelClosed {
  private final List<List<DRes<SInt>>> supportVectors;
  private final List<DRes<SInt>> bias;

  public SVMModelClosed(List<List<DRes<SInt>>> supportVectors, List<DRes<SInt>> bias) {
    this.supportVectors = supportVectors;
    this.bias = bias;

    if (supportVectors.size() != bias.size()) {
      throw new IllegalArgumentException("The amount of bias and support vectors is not the same");
    }

    int size = supportVectors.get(0).size();
    for (List<DRes<SInt>> currentVector : supportVectors) {
      if (size != currentVector.size()) {
        throw new IllegalArgumentException(
            "The amount of featues is not the same for all support vectors");
      }
    }
  }

  public List<List<DRes<SInt>>> getSupportVectors() {
    return supportVectors;
  }

  public List<DRes<SInt>> getBias() {
    return bias;
  }

  public int getNumFeatures() {
    return supportVectors.get(0).size();
  }

  public int getNumSupportVectors() {
    return supportVectors.size();
  }
}
