package dk.alexandra.fresco.ml.fl;

import static org.junit.Assert.assertTrue;

import dk.alexandra.fresco.framework.builder.numeric.NumericResourcePool;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.ml.fl.demo.MnistTestContext;
import dk.alexandra.fresco.ml.fl.demo.TestSetup;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * An abstract functional test of MPC based federated training of a model for the MNIST data set.
 *
 * @param <ResourcePoolT>
 *          the resource pool type used in the computation.
 */
public abstract class MpcMnistFlTest<ResourcePoolT extends NumericResourcePool> {

  private static final int NUM_PARTIES = 3;
  private Map<Integer, TestSetup<ResourcePoolT, ProtocolBuilderNumeric>> setups;
  private MnistTestContext mnistContext;

  @Before
  public void setUp() throws Exception {
    setups = getSetups(NUM_PARTIES);
    mnistContext = MnistTestContext.builder().build();
  }

  public abstract Map<Integer, TestSetup<ResourcePoolT, ProtocolBuilderNumeric>> getSetups(
      int numParties);

  /**
   * Closes the setups used for the test.
   *
   * @throws Exception
   *           if the closing courses exceptions
   */
  @After
  public void tearDown() throws Exception {
    for (TestSetup<?, ?> setup : setups.values()) {
      setup.close();
    }
  }

  @Test
  public void test() throws InterruptedException, ExecutionException {
    ExecutorService es = Executors.newFixedThreadPool(NUM_PARTIES);
    Map<Integer, Future<?>> futures = new HashMap<>(NUM_PARTIES);
    for (int i = 1; i <= NUM_PARTIES; i++) {
      Future<?> f = es.submit(new MnistClient<>(setups.get(i), mnistContext));
      futures.put(i, f);
    }
    for (Future<?> f : futures.values()) {
      f.get();
    }
  }

  /**
   * Callable representing a party in the test.
   *
   * @param <ResourcePoolT>
   *          the resource pool type of the computation
   */
  private static class MnistClient<ResourcePoolT extends NumericResourcePool>
      implements Callable<Object> {

    private TestSetup<ResourcePoolT, ProtocolBuilderNumeric> setup;
    private MnistTestContext context;

    public MnistClient(TestSetup<ResourcePoolT, ProtocolBuilderNumeric> setup,
        MnistTestContext context) {
      this.setup = setup;
      this.context = context;
    }

    @Override
    public Object call() throws Exception {
      FlTrainer trainer = buildFlTrainer(context, setup.getRp().getMyId());
      for (int i = 0; i < context.getGlobalEpochs(); i++) {
        trainer.fitLocalModel();
        trainer.updateGlobalModel();
      }
      DataSetIterator mnistTest = new MnistDataSetIterator(context.getBatchSize(), false,
          context.getSeed());
      Evaluation eval = trainer.getModel().evaluate(mnistTest);
      System.out.println(eval.stats());
      assertTrue(eval.accuracy() > 0.8);
      assertTrue(eval.precision() > 0.8);
      assertTrue(eval.recall() > 0.8);
      assertTrue(eval.f1() > 0.8);
      return null;
    }

    private FlTrainer buildFlTrainer(MnistTestContext context, int id) throws IOException {
      ClientFlHandler flHandler = new DirectMpcFlHandler<>(setup.getSce(), setup.getRp(),
          setup.getNet());
      DataSetIterator trainSet = new MnistDataSetIterator(context.getBatchSize(),
          context.getLocalExamples(), false, true, true, id);
      MultiLayerNetwork model = new MultiLayerNetwork(context.getConf());
      model.init();
      FlTrainer trainer = new FlTrainerImpl(flHandler, model, trainSet, context.getLocalEpochs(),
          context.getLocalExamples());
      return trainer;
    }

  }

}
