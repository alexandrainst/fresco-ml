package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.suite.dummy.arithmetic.AbstractDummyArithmeticTest;
import org.junit.Test;

public class TestDecisionTreeComputationsDummy extends AbstractDummyArithmeticTest {

  @Test
  public void testEvaluateDecisionTree() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTree<>(),
        new TestParameters().numParties(2));
  }

  @Test
  public void testEvaluateDecisionTreeFour() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTreeFour<>(), new TestParameters()
        .numParties(2));
  }

  @Test
  public void testEvaluateDecisionTreeFive() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTreeFive<>(), new TestParameters()
        .numParties(2));
  }

  @Test
  public void testEvaluateDecisionTreeSix() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTreeSix<>(), new TestParameters()
        .numParties(2));
  }

  @Test
  public void testEvaluateDecisionTreeAdvancedModel() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTreeAdvancedModel<>(),
        new TestParameters()
        .numParties(2));
  }
}
