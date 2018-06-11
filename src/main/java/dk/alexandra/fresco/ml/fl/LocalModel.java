package dk.alexandra.fresco.ml.fl;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.ndarray.INDArray;

class LocalModel {

  List<INDArray> weights;
  List<INDArray> biases;
  int examples;

  LocalModel(List<INDArray> weights, List<INDArray> biases, int examples) {
    this.weights = weights;
    this.biases = biases;
    this.examples = examples;
  }

  List<INDArray> getWeights() {
    return weights;
  }

  List<INDArray> getBiases() {
    return biases;
  }

  int getExamples() {
    return examples;
  }

  static LocalModel addModels(LocalModel a, LocalModel b) {
    if (a.weights.size() != b.weights.size()) {
      throw new IllegalArgumentException("Weights of local models do not match size");
    }
    if (a.biases.size() != b.biases.size()) {
      throw new IllegalArgumentException("Biases of local models do not match size");
    }
    List<INDArray> sumweights = IntStream.range(0, a.weights.size())
        .mapToObj(i -> a.weights.get(i).add(b.weights.get(i)))
        .collect(Collectors.toList());
    List<INDArray> sumbiases = IntStream.range(0, a.biases.size())
        .mapToObj(i -> a.biases.get(i).add(b.biases.get(i)))
        .collect(Collectors.toList());
    return new LocalModel(sumweights, sumbiases, a.examples + b.examples);
  }

  static LocalModel divModel(LocalModel model, Number divisor) {
    List<INDArray> divweights = model.weights.stream()
        .map(w -> w.div(divisor))
        .collect(Collectors.toList());
    List<INDArray> divbiases = model.biases.stream()
        .map(b -> b.div(divisor))
        .collect(Collectors.toList());
    return new LocalModel(divweights, divbiases, model.examples);
  }

}