package dk.alexandra.fresco.ml.fl;

import java.io.IOException;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistFederatedLearningTest {

  private static Logger log = LoggerFactory.getLogger(MnistFederatedLearningTest.class);

  @Test
  public void test() throws IOException {
  //number of rows and columns in the input pictures
    final int numRows = 28;
    final int numColumns = 28;
    int outputNum = 10; // number of output classes
    int batchSize = 128; // batch size for each epoch
    int rngSeed = 123; // random number seed for reproducibility
    int numEpochs = 10; // number of epochs to perform
    double rate = 0.0015; // learning rate

    //Get the DataSetIterators:
    DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,
        3000,
        false,
        true,
        true,
        666);
    DataSetIterator mnistTrain2 = new MnistDataSetIterator(batchSize,
        3000,
        false,
        true,
        true,
        777);
    DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);


    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(rngSeed) //include a random seed for reproducibility
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .updater(new Sgd(rate))
        //.l2(rate * 0.005) // regularize learning model
        .list()
        .layer(0, new DenseLayer.Builder() //create the first input layer.
                .nIn(numRows * numColumns)
                .nOut(500)
                .build())
        .layer(1, new DenseLayer.Builder() //create the second input layer
                .nIn(500)
                .nOut(100)
                .build())
        .layer(2, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                .activation(Activation.SOFTMAX)
                .nIn(100)
                .nOut(outputNum)
                .build())
        .pretrain(false).backprop(true) //use backpropagation to adjust weights
        .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    model.setListeners(new ScoreIterationListener(batchSize));  //print the score with every iteration
    MultiLayerNetwork model2 = new MultiLayerNetwork(conf);
    model2.init();
    ClientFlProtocolHandler server = new ClientLocalFlHandler();
    ClientFlTrainerImpl trainer = new ClientFlTrainerImpl(model, mnistTrain, 10, server);
    ClientFlTrainerImpl trainer2 = new ClientFlTrainerImpl(model2, mnistTrain2, 10, server);
    log.info("Train model....");
    for( int i=0; i< numEpochs; i++ ){
      log.info("Epoch " + i);
      //model.fit(mnistTrain);
      trainer.fitLocalModel();
      trainer2.fitLocalModel();
      trainer.updateGlobalModel();
      trainer2.updateGlobalModel();
    }
    model = trainer.getModel();
    model2 = trainer2.getModel();
    /*
    log.info("Evaluate model 1....");
    Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
    while(mnistTest.hasNext()){
        DataSet next = mnistTest.next();
        INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
        eval.eval(next.getLabels(), output); //check the prediction against the true class
    }
    */

    mnistTest.reset();
    log.info("Evaluate model 2....");
    Evaluation eval2 = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
    while(mnistTest.hasNext()){
        DataSet next = mnistTest.next();
        INDArray output = model2.output(next.getFeatureMatrix()); //get the networks prediction
        eval2.eval(next.getLabels(), output); //check the prediction against the true class
    }

    log.info(eval2.stats());
    //log.info(eval2.stats());
    log.info("****************Example finished********************");
  }

}
