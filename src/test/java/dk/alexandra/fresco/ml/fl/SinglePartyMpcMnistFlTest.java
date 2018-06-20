package dk.alexandra.fresco.ml.fl;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This test tests a run of the MPC based Fl training with just a single party.
 * <p>
 * This can be considered a sanity check.
 * </p>
 */
public class SinglePartyMpcMnistFlTest {

  private static Logger log = LoggerFactory.getLogger(SinglePartyMpcMnistFlTest.class);
  private MnistTestContext context;
  private SinglePartyTestSetup setup;

  /**
   * Sets up the test including setting some parameters for the test and setting up the MPC parts.
   * @throws Exception if anything goes badly
   */
  @Before
  public void setUp() throws Exception {
    this.context = MnistTestContext.builder().localExamples(5000).build();
    this.setup = new SinglePartyTestSetup();
  }

  @After
  public void tearDown() throws Exception {
    this.setup.close();
  }

  @Test
  public void test() throws IOException {
    FlTrainer trainer = buildFlTrainer(context.getConf());
    trainModel(trainer);
    testModel(trainer);
  }

  private void testModel(FlTrainer trainer) throws IOException {
    DataSetIterator mnistTest =
        new MnistDataSetIterator(context.getBatchSize(), false, context.getSeed());
    Evaluation eval = new Evaluation(MnistTestContext.NUM_CLASSES);
    while (mnistTest.hasNext()) {
      DataSet next = mnistTest.next();
      INDArray output = trainer.getModel().output(next.getFeatureMatrix());
      eval.eval(next.getLabels(), output);
    }
    log.info(eval.stats());
    assertTrue(eval.accuracy() > 0.8);
    assertTrue(eval.precision() > 0.8);
    assertTrue(eval.recall() > 0.8);
    assertTrue(eval.f1() > 0.8);
  }

  private void trainModel(FlTrainer trainer) {
    log.info("Train model....");
    for (int i = 0; i < context.getGlobalEpochs(); i++) {
      log.info("Epoch " + i);
      trainer.fitLocalModel();
      trainer.updateGlobalModel();
    }
  }

  private FlTrainer buildFlTrainer(MultiLayerConfiguration conf) throws IOException {
    ClientFlHandler flHandler =
        new DirectMpcFlHandler<>(setup.getSce(), setup.getRp(), setup.getNet());
    DataSetIterator train =
        new MnistDataSetIterator(context.getBatchSize(), context.getLocalExamples());
    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    FlTrainer trainer = new FlTrainerImpl(model, train, context.getLocalEpochs(), flHandler);
    return trainer;
  }

}
