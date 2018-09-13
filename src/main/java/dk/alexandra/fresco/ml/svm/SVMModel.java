package dk.alexandra.fresco.ml.svm;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

public class SVMModel {
  private final List<List<BigInteger>> supportVectors;
  private final List<BigInteger> bias;
  private final int precision;

  public SVMModel(List<List<Double>> doubleSupportVectors, List<Double> doubleBias, int precision) {
    if (doubleSupportVectors.size() != doubleBias.size()) {
      throw new IllegalArgumentException("The amount of bias and support vectors is not the same");
    }

    int size = doubleSupportVectors.get(0).size();
    for (List<Double> currentVector : doubleSupportVectors) {
      if (size != currentVector.size()) {
        throw new IllegalArgumentException(
            "The amount of featues is not the same for all support vectors");
      }
    }
    this.precision = precision;

    List<BigInteger> bigBias = new ArrayList<>(doubleBias.size());
    for (Double currentDouble : doubleBias) {
      // We must multiply the bias with 'precision' again since the inner products have been shifted
      // by precision twice as they consist of the sum of the product of shifted values
      bigBias.add(convertToBigInteger(currentDouble).multiply(BigInteger.valueOf(
          precision)));
    }

    List<List<BigInteger>> bigSupportvectors = new ArrayList<>(doubleSupportVectors.size());
    for (List<Double> currentVector : doubleSupportVectors) {
      List<BigInteger> currentBigVector = new ArrayList<>(currentVector.size());
      for (Double currentDouble : currentVector) {
        currentBigVector.add(convertToBigInteger(currentDouble));
      }
      bigSupportvectors.add(currentBigVector);
    }
    this.supportVectors = bigSupportvectors;
    this.bias = bigBias;
  }

  // public SVMModel(List<List<BigInteger>> supportVectors, List<BigInteger> bias) {
  // this.supportVectors = supportVectors;
  // this.bias = bias;
  //
  // if (supportVectors.size() != bias.size()) {
  // throw new IllegalArgumentException("The amount of bias and support vectors is not the same");
  // }
  //
  // int size = supportVectors.get(0).size();
  // for (List<BigInteger> currentVector : supportVectors) {
  // if (size != currentVector.size()) {
  // throw new IllegalArgumentException(
  // "The amount of featues is not the same for all support vectors");
  // }
  // }
  // }

  public BigInteger convertToBigInteger(Double input) {
    // We use BigDecimal to avoid loss of precision when converting to integer
    BigDecimal currentBigDouble = new BigDecimal(input);
    // "Shift" to become an "integer"
    currentBigDouble = currentBigDouble.multiply(new BigDecimal(precision));
    // Round down to integer
    return currentBigDouble.toBigInteger();
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
