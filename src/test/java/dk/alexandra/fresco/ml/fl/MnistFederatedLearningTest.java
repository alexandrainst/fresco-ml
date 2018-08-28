package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.ml.fl.demo.MnistTestContext;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Functional test for plaintext fully local federated learning (i.e., just the averaging of
 * models).
 *
 * <p>
 * This is can be seen as a sanity check to compliment the MPC based tests. Also, useful for
 * comparing performance of the trained models.
 * </p>
 */
public class MnistFederatedLearningTest {

  private static Logger log = LoggerFactory.getLogger(MnistFederatedLearningTest.class);

  @Test
  public void test() throws IOException {
    final int parties = 3;
    MnistTestContext context = MnistTestContext.builder().build();
    List<DataSetIterator> iterators = new ArrayList<>(parties);
    List<FlTrainer> trainers = new ArrayList<>(parties);
    ClientFlHandler flHandler = new LocalFlHandler();
    for (int i = 0; i < parties; i++) {
      DataSetIterator train = new MnistDataSetIterator(context.getBatchSize(),
          context.getLocalExamples(), false, true, true, i);
      iterators.add(train);
      MultiLayerNetwork model = new MultiLayerNetwork(context.getConf());
      model.init();
      FlTrainerImpl trainer = new FlTrainerImpl(flHandler, model, train, context.getLocalEpochs(),
          context.getLocalExamples());
      trainers.add(trainer);
    }
    log.info("Train model....");
    for (int i = 0; i < context.getGlobalEpochs(); i++) {
      log.info("Epoch " + i);
      trainers.stream().parallel().forEach(FlTrainer::fitLocalModel);
      trainers.stream().forEach(FlTrainer::updateGlobalModel);
    }
    DataSetIterator mnistTest = new MnistDataSetIterator(context.getBatchSize(), false,
        context.getSeed());

    log.info("Evaluate model ....");
    Evaluation evaluator = trainers.get(0).getModel().evaluate(mnistTest);
    log.info(evaluator.stats());
  }

}
