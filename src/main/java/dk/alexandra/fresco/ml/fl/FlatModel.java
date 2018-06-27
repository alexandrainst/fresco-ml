package dk.alexandra.fresco.ml.fl;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A flat representation of a locally trained model simply holding the parameters and the number of
 * examples the model was trained on. I.e., the only things needed for the averaging.
 */
class FlatModel {

  private final INDArray params;
  private final int examples;

  /**
   * A basic constructor for flat local model parameters.
   * @param params the parameters
   * @param examples the number of examples the model is trained on
   */
  FlatModel(INDArray params, int examples) {
    this.params = params;
    this.examples = examples;
  }

  /**
   * Get the model parameters.
   * @return the parameters
   */
  INDArray getParams() {
    return params;
  }

  /**
   * Get the number of examples these model parameters are trained on.
   * @return the number of examples
   */
  int getExamples() {
    return examples;
  }

}