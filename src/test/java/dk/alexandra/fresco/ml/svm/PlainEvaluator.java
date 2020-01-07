package dk.alexandra.fresco.ml.svm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.csv.CSVRecord;

public class PlainEvaluator {
  private final List<List<Double>> supportVectors;
  private final List<Double> bias;

  public PlainEvaluator(List<List<Double>> supportVectors, List<Double> bias) {
    this.supportVectors = supportVectors;
    this.bias = bias;
  }

  public int evaluate(List<Double> inputVector) {
    List<Double> partialResults = new ArrayList<>(supportVectors.size());
    for (int i = 0; i < supportVectors.size(); i++) {
      double currentProduct = innerProduct(inputVector, supportVectors.get(i));
      currentProduct += bias.get(i);
      partialResults.add(currentProduct);
    }
    if (partialResults.size() > 1) {
      double maxValue = Collections.max(partialResults);
      return partialResults.indexOf(maxValue);
    } else {
      if (partialResults.get(0) > 0.0) {
        return 1;
      } else {
        return 0;
      }
    }
  }

  private double innerProduct(List<Double> first, List<Double> second) {
    double partialResult = 0.0;
    for (int i = 0; i < first.size(); i++) {
      partialResult += first.get(i) * second.get(i);
    }
    return partialResult;
  }

  public static List<List<Double>> CSVListToDouble(List<CSVRecord> records) {
    List<List<Double>> res = new ArrayList<List<Double>>(records.size());
    for (int i = 0; i < records.size(); i++) {
      res.add(CSVToDouble(records.get(i)));
    }
    return res;
  }

  public static List<Double> CSVToDouble(CSVRecord record) {
    List<Double> res = new ArrayList<>();
    for (int i = 0; i < record.size(); i++) {
      res.add(new Double(record.get(i)));
    }
    return res;
  }

  /*** The outcommented code below is only needed if we use EPIC SVMs ***/
  // public static List<Double> transform(List<Double> input, int nc, double gamma) {
  // List<List<Double>> gaussian = getGaussianRandomness(input.size(), nc, gamma);
  // List<Double> uniform = getUniformRandomness(nc);
  // List<Double> res = new ArrayList<>(nc);
  // for (int i = 0; i < nc; i++) {
  // double current = 0.0;
  // for (int j = 0; j < input.size(); j++) {
  // current += input.get(j) * gaussian.get(i).get(j);
  // }
  // // Add offset
  // current += uniform.get(i);
  // current = Math.cos(current);
  // current *= Math.sqrt(2.0) / Math.sqrt(nc);
  // res.add(current);
  // }
  // return res;
  // }
  //
  // private static List<List<Double>> getGaussianRandomness(int shape, int nc, double gamma) {
  // Random rand = new Random();
  // List<List<Double>> res = new ArrayList<>(nc);
  // for (int i = 0; i < nc; i++) {
  // List<Double> currentList = new ArrayList<>(shape);
  // for (int j = 0; j < shape; j++) {
  // currentList.add(rand.nextGaussian() * Math.sqrt(2.0 * gamma));
  // }
  // res.add(currentList);
  // }
  // return res;
  // }
  //
  // private static List<Double> getUniformRandomness(int nc) {
  // Random rand = new Random();
  // List<Double> uniformValues = new ArrayList<>(nc);
  // for (int i = 0; i < nc; i++) {
  // Double decimal = rand.nextDouble() * 2.0 * Math.PI;
  // uniformValues.add(decimal);
  // }
  // return uniformValues;
  // }
}
