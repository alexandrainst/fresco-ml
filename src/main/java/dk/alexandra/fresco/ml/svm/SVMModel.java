package dk.alexandra.fresco.ml.svm;

import java.math.BigInteger;
import java.util.List;

public class SVMModel {
  private final List<List<BigInteger>> supportVectors;
  private final List<BigInteger> bias;

  public SVMModel(List<List<BigInteger>> supportVectors, List<BigInteger> bias) {
    this.supportVectors = supportVectors;
    this.bias = bias;

    if (supportVectors.size() != bias.size()) {
      throw new IllegalArgumentException("The amount of bias and support vectors is not the same");
    }

    int size = supportVectors.get(0).size();
    for (List<BigInteger> currentVector : supportVectors) {
      if (size != currentVector.size()) {
        throw new IllegalArgumentException(
            "The amount of featues is not the same for all support vectors");
      }
    }
  }

  public List<List<BigInteger>> getSupportVectors() {
    return supportVectors;
  }

  public List<BigInteger> getBias() {
    return bias;
  }

  public int getNumFeatures() {
    return supportVectors.get(0).size();
  }

  public int getNumSupportVectors() {
    return supportVectors.size();
  }

}
