package dk.alexandra.fresco.ml.nn;

import dk.alexandra.fresco.lib.collections.Matrix;
import dk.alexandra.fresco.ml.nn.ActivationFunctions.Type;

public class FullyConnectedLayerParameters<T> {

  private Matrix<T> weights;
  private Type activation;
  private Matrix<T> bias;

  public FullyConnectedLayerParameters(Matrix<T> weights, Matrix<T> bias, Type activationFunction) {

    if (bias.getWidth() != 1) {
      throw new IllegalArgumentException(
          "Bias must be a column vector. Has width " + bias.getWidth() + " != 1.");
    }

    if (weights.getHeight() != bias.getHeight()) {
      throw new IllegalArgumentException("Height of weight matrix (" + weights.getHeight()
          + ") must be equal to height of bias vector (" + bias.getHeight() + ")");
    }

    this.weights = weights;
    this.bias = bias;
    this.activation = activationFunction;
  }

  public Matrix<T> getWeights() {
    return weights;
  }

  public Type getActivation() {
    return activation;
  }

  public Matrix<T> getBias() {
    return bias;
  }

  public int getInputs() {
    return weights.getWidth();
  }

  public int getOutputs() {
    return bias.getHeight();
  }
}
