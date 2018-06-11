package dk.alexandra.fresco.ml.fl;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Interface for federated training
 */
public interface FederatedTrainer {

  void fit(MultiLayerNetwork model, DataSetIterator trainingData);

}
