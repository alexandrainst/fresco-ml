package dk.alexandra.fresco.ml.fl;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * Interface towards a machine learning application for training using federated learning.
 */
public interface FlTrainer {

  /**
   * Fits a local model to the local training data.
   */
  void fitLocalModel();

  /**
   * Updates the local model from the global model.
   */
  void updateGlobalModel();

  /**
   * Gets the model being trained by this trainer.
   *
   * @return the model being trained
   */
  MultiLayerNetwork getModel();

}
