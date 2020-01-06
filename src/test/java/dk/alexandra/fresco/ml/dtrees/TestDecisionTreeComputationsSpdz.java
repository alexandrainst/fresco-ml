package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.framework.sce.evaluator.EvaluationStrategy;
import dk.alexandra.fresco.suite.spdz.AbstractSpdzTest;
import dk.alexandra.fresco.suite.spdz.configuration.PreprocessingStrategy;
import org.junit.Test;

public class TestDecisionTreeComputationsSpdz extends AbstractSpdzTest {

  @Test
  public void testEvaluateDecisionTree() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTree<>(),
        PreprocessingStrategy.DUMMY, 2);
  }

  @Test
  public void testEvaluateDecisionTreeFour() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTreeFour<>(),
        PreprocessingStrategy.DUMMY, 2);
  }

  @Test
  public void testEvaluateDecisionTreeFive() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTreeFive<>(),
        PreprocessingStrategy.DUMMY, 2);
  }

  @Test
  public void testEvaluateDecisionTreeSix() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTreeSix<>(),
        PreprocessingStrategy.DUMMY, 2);
  }

}
