package dk.alexandra.fresco.ml.fl;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

/**
 * Interface for federated training.
 */
public interface ClientFlTrainer {

  void fitLocalModel();

  void updateGlobalModel();

  MultiLayerNetwork getModel();

}
