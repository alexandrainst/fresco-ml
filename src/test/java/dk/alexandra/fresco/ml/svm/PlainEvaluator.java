package dk.alexandra.fresco.ml.svm;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

public class PlainEvaluator {

  private final SVMModel model;

  public PlainEvaluator(SVMModel model) {
    this.model = model;
  }

  BigInteger evaluate(List<BigInteger> inputVector) {
    List<BigInteger> partialResults = new ArrayList<>(model.getNumSupportVectors());
    for (int i = 0; i < model.getNumSupportVectors(); i++) {
      BigInteger currentProduct = innerProduct(inputVector, model.getSupportVectors().get(i));
      currentProduct.add(model.getBias().get(i));
      partialResults.add(currentProduct);
    }

    return argMax(partialResults);
  }

  private BigInteger innerProduct(List<BigInteger> first, List<BigInteger> second) {
    BigInteger partialResult = BigInteger.ZERO;
    for (int i = 0; i < first.size(); i++) {
      partialResult = partialResult.add(first.get(i).multiply(second.get(i)));
    }
    return partialResult;
  }

  private BigInteger argMax(List<BigInteger> list) {
    BigInteger currentMax = list.get(0);
    for (BigInteger currentVal : list) {
      if (currentVal.compareTo(currentMax) == 1) {
        currentMax = currentVal;
      }
    }
    return currentMax;
  }
}
