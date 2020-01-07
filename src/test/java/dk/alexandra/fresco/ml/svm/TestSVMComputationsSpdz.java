package dk.alexandra.fresco.ml.svm;

import dk.alexandra.fresco.suite.spdz.AbstractSpdzTest;
import dk.alexandra.fresco.suite.spdz.configuration.PreprocessingStrategy;
import org.junit.Test;

public class TestSVMComputationsSpdz extends AbstractSpdzTest {

  @Test
  public void testEvaluateSVM() {
    runTest(new SVMComputationTests.TestEvaluateSVM<>(),
        PreprocessingStrategy.DUMMY, 2, 128, 64, 16);
  }
}
