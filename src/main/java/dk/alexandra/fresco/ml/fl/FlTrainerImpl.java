package dk.alexandra.fresco.ml.fl;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Basic implementation of a trainer for Federated Learning.
 */
public final class FlTrainerImpl implements FlTrainer {

  private final ClientFlHandler flHandler;
  private final MultiLayerNetwork model;
  private final DataSetIterator trainingData;
  private final int epochs;

  /**
   * Constructs a new trainer.
   *
   * @param model the model to train (should be initialized)
   * @param trainingData the local training data to fit to
   * @param epochs the number of epochs to use when training locally
   * @param flHandler the handler used for interacting with the other parties
   */
  public FlTrainerImpl(MultiLayerNetwork model, DataSetIterator trainingData, int epochs,
      ClientFlHandler flHandler) {
    this.model = model;
    this.trainingData = trainingData;
    this.flHandler = flHandler;
    this.epochs = epochs;
  }

  @Override
  public void fitLocalModel() {
    for (int i = 0; i < epochs; i++) {
      model.fit(trainingData);
    }
    flHandler.submitLocalModel(new FlatModel(model.params(), trainingData.numExamples()));
  }

  @Override
  public void updateGlobalModel() {
    FlatModel globalModel = flHandler.getAveragedModel();
    model.setParameters(globalModel.getParams());
  }

  @Override
  public MultiLayerNetwork getModel() {
    return model;
  }

}
