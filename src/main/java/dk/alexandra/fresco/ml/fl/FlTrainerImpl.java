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
  private final int numExamples;

  /**
   * Constructs a new trainer.
   *
   * @param flHandler
   *          the handler used for interacting with the other parties
   * @param model
   *          the model to train (should be initialized)
   * @param trainingData
   *          the local training data to fit to
   * @param epochs
   *          the number of epochs to use when training locally
   * @param numExamples
   *          the number of examples represented in the <code>trainingData</code>
   */
  public FlTrainerImpl(ClientFlHandler flHandler, MultiLayerNetwork model,
      DataSetIterator trainingData, int epochs, int numExamples) {
    this.model = model;
    this.trainingData = trainingData;
    this.numExamples = numExamples;
    this.flHandler = flHandler;
    this.epochs = epochs;
  }

  @Override
  public void fitLocalModel() {
    model.fit(trainingData, epochs);
  }

  @Override
  public void updateGlobalModel() {
    flHandler.submitLocalModel(new FlatModel(model.params(), numExamples));
    FlatModel globalModel = flHandler.getAveragedModel();
    model.setParameters(globalModel.getParams());
  }

  @Override
  public MultiLayerNetwork getModel() {
    return model;
  }

  public DataSetIterator getTrainingData() {
    return trainingData;
  }

}
