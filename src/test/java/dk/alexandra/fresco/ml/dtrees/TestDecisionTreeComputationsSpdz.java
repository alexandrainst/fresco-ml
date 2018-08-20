package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.framework.sce.evaluator.EvaluationStrategy;
import dk.alexandra.fresco.suite.spdz.AbstractSpdzTest;
import dk.alexandra.fresco.suite.spdz.configuration.PreprocessingStrategy;
import org.junit.Test;

public class TestDecisionTreeComputationsSpdz extends AbstractSpdzTest {

  @Test
  public void testInputDecisionTree() {
    runTest(new DecisionTreeComputationTests.TestInputDecisionTree<>(),
        EvaluationStrategy.SEQUENTIAL_BATCHED,
        PreprocessingStrategy.DUMMY, 2);
  }

  @Test
  public void testEvaluateDecisionTree() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTree<>(),
        EvaluationStrategy.SEQUENTIAL_BATCHED,
        PreprocessingStrategy.DUMMY, 2);
  }

  @Test
  public void testEvaluateDecisionTreeFour() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTreeFour<>(),
        EvaluationStrategy.SEQUENTIAL_BATCHED,
        PreprocessingStrategy.DUMMY, 2);
  }

  @Test
  public void testEvaluateDecisionTreeFive() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTreeFive<>(),
        EvaluationStrategy.SEQUENTIAL_BATCHED,
        PreprocessingStrategy.DUMMY, 2);
  }

  @Test
  public void testEvaluateDecisionTreeSix() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTreeSix<>(),
        EvaluationStrategy.SEQUENTIAL_BATCHED,
        PreprocessingStrategy.DUMMY, 2);
  }

}
