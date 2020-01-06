package dk.alexandra.fresco.ml;

import dk.alexandra.fresco.framework.sce.evaluator.EvaluationStrategy;
import dk.alexandra.fresco.suite.dummy.arithmetic.AbstractDummyArithmeticTest;
import org.junit.Test;

public class TestDummyArithmeticProtocolSuite extends AbstractDummyArithmeticTest {

  @Test
  public void test_NN_1_layer() throws Exception {
    runTest(new NNTests.TestNN1layer<>(), EvaluationStrategy.SEQUENTIAL, 2);
  }

  @Test
  public void test_NN_2_layers() throws Exception {
    runTest(new NNTests.TestNN2layer<>(), EvaluationStrategy.SEQUENTIAL, 2);
  }

  @Test
  public void test_Federated_Learning() throws Exception {
    runTest(new NNTests.TestFederatedLearning<>(), EvaluationStrategy.SEQUENTIAL,
        2);
  }
  
  @Test
  public void test_logistic_regression_prediction() throws Exception {
    runTest(new LRTests.TestLogRegPrediction<>(), EvaluationStrategy.SEQUENTIAL, 2);
  }

  @Test
  public void test_logistic_regression_sgd_single_epoch() throws Exception {
    runTest(new LRTests.TestLogRegSGDSingleEpoch<>(), EvaluationStrategy.SEQUENTIAL, 2);
  }

}
