package dk.alexandra.fresco.ml.svm;

import dk.alexandra.fresco.framework.sce.evaluator.EvaluationStrategy;
import dk.alexandra.fresco.suite.spdz.AbstractSpdzTest;
import dk.alexandra.fresco.suite.spdz.configuration.PreprocessingStrategy;
import org.junit.Test;

public class TestSVMComputationsSpdz extends AbstractSpdzTest {

  @Test
  public void testEvaluateSVM() {
    runTest(new SVMComputationTests.TestEvaluateSVM<>(), EvaluationStrategy.SEQUENTIAL_BATCHED,
        PreprocessingStrategy.DUMMY, 2, false, 128, 64, 16);
  }
}
