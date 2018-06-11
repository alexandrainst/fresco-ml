package dk.alexandra.fresco.ml.fl;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class ClientFlTrainerImpl implements ClientFlTrainer {

  private final ClientFlProtocolHandler server;
  private final MultiLayerNetwork model;
  private final DataSetIterator trainingData;
  private final int epochs;

  public ClientFlTrainerImpl(MultiLayerNetwork model, DataSetIterator trainingData, int epochs,
      ClientFlProtocolHandler server) {
    this.model = model;
    this.trainingData = trainingData;
    this.server = server;
    this.epochs = epochs;
  }

  @Override
  public void fitLocalModel() {
    trainingData.reset();
    for (int i = 0; i < epochs; i++) {
      model.fit(trainingData);
    }
    List<INDArray> weights = extractWeigths(model);
    List<INDArray> biases = extractBiases(model);
    int examples = trainingData.numExamples();
    weights = weights.stream().map(w -> w.mul(examples)).collect(Collectors.toList());
    biases = biases.stream().map(b -> b.mul(examples)).collect(Collectors.toList());
    server.submitLocalModel(new LocalModel(weights, biases, examples));
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

  @Override
  public void updateGlobalModel() {
    LocalModel globalModel = server.getGlobalModel();
    int numLayers = model.getnLayers();
    model.getLayer(1).setParams(null);
    for (int i = 0; i < numLayers; i++) {
      model.setParam(i + "_W", globalModel.getWeights().get(i));
      model.setParam(i + "_b", globalModel.getBiases().get(i));
    }
  }

  @Override
  public MultiLayerNetwork getModel() {
    return model;
  }

}
