package dk.alexandra.fresco.ml.svm;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class PlainEvaluator {
  private final SVMModel model;

  public PlainEvaluator(SVMModel model) {
    this.model = model;
  }

  public int evaluate(List<Double> inputVector) {
    List<BigInteger> list = inputVector.stream().map(val -> model.convertToBigInteger(val)).collect(
        Collectors.toList());
    List<BigInteger> partialResults = new ArrayList<>(model.getNumSupportVectors());
    for (int i = 0; i < model.getNumSupportVectors(); i++) {
      BigInteger currentProduct = innerProduct(list, model.getSupportVectors().get(i));
      currentProduct.add(model.getBias().get(i));
      partialResults.add(currentProduct);
    }
    if (partialResults.size() > 1) {
      BigInteger maxValue = Collections.max(partialResults);
      return partialResults.indexOf(maxValue);
    } else {
      if (partialResults.get(0).compareTo(BigInteger.ZERO) > 0) {
        return 1;
      } else {
        return 0;
      }
    }
  }

  private BigInteger innerProduct(List<BigInteger> first, List<BigInteger> second) {
    BigInteger partialResult = BigInteger.ZERO;
    for (int i = 0; i < first.size(); i++) {
      partialResult = partialResult.add(first.get(i).multiply(second.get(i)));
    }
    return partialResult;
  }

  public static List<Double> transform(List<Double> input, int nc, double gamma) {
    List<List<Double>> gaussian = getGaussianRandomness(input.size(), nc, gamma);
    List<Double> uniform = getUniformRandomness(nc);
    List<Double> res = new ArrayList<>(nc);
    for (int i = 0; i < nc; i++) {
      double current = 0.0;
      for (int j = 0; j < input.size(); j++) {
        current += input.get(j) * gaussian.get(i).get(j);
      }
      // Add offset
      current += uniform.get(i);
      current = Math.cos(current);
      current *= Math.sqrt(2.0) / Math.sqrt(nc);
      res.add(current);
    }
    return res;
  }

  private static List<List<Double>> getGaussianRandomness(int shape, int nc, double gamma) {
    Random rand = new Random();
    List<List<Double>> res = new ArrayList<>(nc);
    for (int i = 0; i < nc; i++) {
      List<Double> currentList = new ArrayList<>(shape);
      for (int j = 0; j < shape; j++) {
        currentList.add(rand.nextGaussian() * Math.sqrt(2.0 * gamma));
      }
      res.add(currentList);
    }
    return res;
  }

  private static List<Double> getUniformRandomness(int nc) {
    Random rand = new Random();
    List<Double> uniformValues = new ArrayList<>(nc);
    for (int i = 0; i < nc; i++) {
      Double decimal = rand.nextDouble() * 2.0 * Math.PI;
      uniformValues.add(decimal);
    }
    return uniformValues;
  }
}
