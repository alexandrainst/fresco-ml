package dk.alexandra.fresco.ml.fl.demo;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * A convenience class used to hold default parameters and ease configuration for the federated
 * MNIST tests.
 *
 */
public class MnistTestContext {

  public static final double DEFAULT_LEARNING_RATE = 0.05;
  public static final double DEFAULT_REGULARIZATION = 0.005;
  public static final int DEFAULT_LAYER_SIZE = 200;
  public static final int DEFAULT_LOCAL_EXAMPLES = 600;
  public static final int DEFAULT_SEED = 123;
  public static final int DEFAULT_LOCAL_EPOCHS = 10;
  public static final int DEFAULT_GLOBAL_EPOCHS = 10;
  public static final int DEFAULT_BATCHSIZE = 10;
  public static final int NUM_CLASSES = 10; // 10 digits
  public static final int NUM_MNIST_FEATURES = 28 * 28; // 28 x 28 pixel images

  private int batchSize = DEFAULT_BATCHSIZE;
  private int globalEpochs = DEFAULT_GLOBAL_EPOCHS;
  private int localEpochs = DEFAULT_LOCAL_EPOCHS;
  private int seed = DEFAULT_SEED;
  private int localExamples = DEFAULT_LOCAL_EXAMPLES;
  private int layerSize = DEFAULT_LAYER_SIZE;
  private double regularization = DEFAULT_REGULARIZATION;
  private double learningRate = DEFAULT_LEARNING_RATE;
  private MultiLayerConfiguration conf;

  public static MnistTestContextBuilder builder() {
    return new MnistTestContextBuilder();
  }

  private MnistTestContext() {

  }

  private void setBatchSize(int batchSize) {
    this.batchSize = batchSize;
  }

  private void setGlobalEpochs(int globalEpochs) {
    this.globalEpochs = globalEpochs;
  }

  private void setLocalEpochs(int localEpochs) {
    this.localEpochs = localEpochs;
  }

  private void setSeed(int seed) {
    this.seed = seed;
  }

  private void setLocalExamples(int localExamples) {
    this.localExamples = localExamples;
  }

  private void setLayerSize(int layerSize) {
    this.layerSize = layerSize;
  }

  private void setRegularization(double regularization) {
    this.regularization = regularization;
  }

  private void setLearningRate(double learningRate) {
    this.learningRate = learningRate;
  }

  /**
   * Gets the batch size to be used in local training. Defaults to
   * {@link MnistTestContext#DEFAULT_BATCHSIZE}.
   *
   * @return the batchSize
   */
  public int getBatchSize() {
    return batchSize;
  }

  /**
   * Get the number of global epochs to use. I.e., the number of times to perform the averaging of
   * models. Defaults to {@link MnistTestContext#DEFAULT_GLOBAL_EPOCHS}.
   *
   * @return the global epochs
   */
  public int getGlobalEpochs() {
    return globalEpochs;
  }

  /**
   * Get the number of local epochs. I.e., the number of epochs to use when training locally.
   * Defaults to {@link MnistTestContext#DEFAULT_LOCAL_EPOCHS}.
   *
   * @return the number of local epochs
   */
  public int getLocalEpochs() {
    return localEpochs;
  }

  /**
   * Get a number to be used through out as randomness seed for reproducibility. Defaults to
   * {@link MnistTestContext#DEFAULT_SEED}.
   *
   * @return a randomness seed
   */
  public int getSeed() {
    return seed;
  }

  /**
   * The number of examples this party should use when training locally. Defaults to
   * {@link MnistTestContext#DEFAULT_LOCAL_EXAMPLES}.
   *
   * @return the number of local examples to use
   */
  public int getLocalExamples() {
    return localExamples;
  }

  /**
   * The size of each of the two dense layers of the model. Defaults to
   * {@link MnistTestContext#DEFAULT_LAYER_SIZE}.
   *
   * @return size of dense layers
   */
  public int getLayerSize() {
    return layerSize;
  }

  /**
   * The regularization parameter. Defaults to {@link MnistTestContext#DEFAULT_REGULARIZATION}.
   *
   * @return regularization parameter
   */
  public double getRegularization() {
    return regularization;
  }

  /**
   * The learning rate used in training the model. Defaults to
   * {@link MnistTestContext#DEFAULT_LEARNING_RATE}.
   *
   *
   * @return learning rate
   */
  public double getLearningRate() {
    return learningRate;
  }

  /**
   * The Neural Network configuration to be used in the test.
   *
   * <p>
   * The configuration is for a network with two dense layers of equal size configured using the
   * parameters of this context. The network uses SGD, RELU activation for the dense layers and
   * SOFTMAX for the output layer.
   * </p>
   *
   * @return the Neural Network configuration
   */
  public MultiLayerConfiguration getConf() {
    return conf;
  }

  private void createConf() {
    this.conf = new NeuralNetConfiguration.Builder().seed(seed).activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER).updater(new Sgd(learningRate))
        .l2(learningRate * regularization).list()
        .layer(0, new DenseLayer.Builder().nIn(NUM_MNIST_FEATURES).nOut(layerSize).build())
        .layer(1, new DenseLayer.Builder().nIn(layerSize).nOut(layerSize).build())
        .layer(2,
            new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX).nIn(layerSize).nOut(NUM_CLASSES).build())
        .pretrain(false).backprop(true) // use backpropagation to adjust weights
        .build();
  }

  /**
   * A builder for the MnistTestContext using the defaults when no parameter is explicitly given.
   */
  public static class MnistTestContextBuilder {

    private MnistTestContext context;

    public MnistTestContextBuilder() {
      this.context = new MnistTestContext();
    }

    public MnistTestContextBuilder batchSize(int batchSize) {
      this.context.setBatchSize(batchSize);
      return this;
    }

    public MnistTestContextBuilder globalEpochs(int globalEpochs) {
      this.context.setGlobalEpochs(globalEpochs);
      return this;
    }

    public MnistTestContextBuilder localEpochs(int localEpochs) {
      this.context.setLocalEpochs(localEpochs);
      return this;
    }

    public MnistTestContextBuilder seed(int seed) {
      this.context.setSeed(seed);
      return this;
    }

    public MnistTestContextBuilder localExamples(int localExamples) {
      this.context.setLocalExamples(localExamples);
      return this;
    }

    public MnistTestContextBuilder layerSize(int layerSize) {
      this.context.setLayerSize(layerSize);
      return this;
    }

    public MnistTestContextBuilder regularization(double regularization) {
      this.context.setRegularization(regularization);
      return this;
    }

    public MnistTestContextBuilder learningRate(double learningRate) {
      this.context.setLearningRate(learningRate);
      return this;
    }

    public MnistTestContext build() {
      context.createConf();
      return context;
    }
  }
}