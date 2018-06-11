package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.util.Pair;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class LocalFederatedTrainer implements FederatedTrainer {

  private LocalFederatedTrainingServer server;

  public LocalFederatedTrainer(LocalFederatedTrainingServer server) {
    this.server = server;
  }

  @Override
  public void fit(MultiLayerNetwork model, DataSetIterator trainingData) {
    model.fit(trainingData);
    List<INDArray> weights = extractWeigths(model);
    List<INDArray> biases = extractBiases(model);
    int examples = trainingData.numExamples();
    weights = weights.stream().map(w -> w.mul(examples)).collect(Collectors.toList());
    biases = biases.stream().map(b -> b.mul(examples)).collect(Collectors.toList());
    Pair<List<INDArray>, List<INDArray>> globalModel =
        server.nextRound().addLocalModel(weights, biases, examples).getGobalModel();
    updateModel(model, globalModel);
  }

  private void updateModel(MultiLayerNetwork model,
      Pair<List<INDArray>, List<INDArray>> globalModel) {
    int numLayers = model.getnLayers();
    List<INDArray> weights = globalModel.getFirst();
    List<INDArray> biases = globalModel.getSecond();
    for (int i = 0; i < numLayers; i++) {
      model.setParam(i + "_W", weights.get(i));
      model.setParam(i + "_b", biases.get(i));
    }
  }

  private List<INDArray> extractWeigths(MultiLayerNetwork model) {
    int numLayers = model.getnLayers();
    List<INDArray> weights = new ArrayList<>(numLayers);
    for (int i = 0; i < numLayers; i++) {
      weights.add(model.paramTable().get(i + "_W"));
    }
    return weights;
  }

  private List<INDArray> extractBiases(MultiLayerNetwork model) {
    int numLayers = model.getnLayers();
    List<INDArray> biases = new ArrayList<>(numLayers);
    for (int i = 0; i < numLayers; i++) {
      biases.add(model.paramTable().get(i + "_b"));
    }
    return biases;
  }

}
