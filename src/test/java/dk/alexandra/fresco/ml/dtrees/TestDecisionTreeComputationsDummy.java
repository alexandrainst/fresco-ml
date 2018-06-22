package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.suite.dummy.arithmetic.AbstractDummyArithmeticTest;
import org.junit.Test;

public class TestDecisionTreeComputationsDummy extends AbstractDummyArithmeticTest {

  @Test
  public void testInputDecisionTree() {
    runTest(new DecisionTreeComputationTests.TestInputDecisionTree<>(), new TestParameters().numParties(2));
  }

  @Test
  public void testEvaluateDecisionTree() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTree<>(), new TestParameters().numParties(2));
  }

}
