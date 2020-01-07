package dk.alexandra.fresco.ml.svm;

import org.junit.Test;

import dk.alexandra.fresco.suite.dummy.arithmetic.AbstractDummyArithmeticTest;

public class TestSVMComputationsDummy extends AbstractDummyArithmeticTest {
  @Test
  public void testEvaluateSVM() {
    runTest(new SVMComputationTests.TestEvaluateSVM<>(),
        new TestParameters().numParties(2));
  }
}
