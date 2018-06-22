package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.ml.dtrees.utils.ModelLoader;
import java.io.IOException;
import java.math.BigInteger;
import org.junit.Assert;
import org.junit.Test;

public class TestPlainEvaluator {

  @Test
  public void testPlainEvaluator() throws IOException {
    ModelLoader loader = new ModelLoader();
    DecisionTreeModel treeModel = loader
        .modelFromFile(loader.getFile("dtrees/models/test-model-1.csv"));
    PlainEvaluator evaluator = new PlainEvaluator(treeModel);
    Assert.assertEquals(BigInteger.valueOf(1),
        evaluator.evaluate(TestUtils.toBitIntegers(new int[]{11, 3, 5, 7})));
    Assert.assertEquals(BigInteger.valueOf(2),
        evaluator.evaluate(TestUtils.toBitIntegers(new int[]{11, 0, 5, 7})));
    Assert.assertEquals(BigInteger.valueOf(3),
        evaluator.evaluate(TestUtils.toBitIntegers(new int[]{5, 0, 5, 12})));
    Assert.assertEquals(BigInteger.valueOf(4),
        evaluator.evaluate(TestUtils.toBitIntegers(new int[]{4, 0, 5, 7})));
  }

  @Test
  public void testPlainEvaluatorDepthFour() throws IOException {
    ModelLoader loader = new ModelLoader();
    DecisionTreeModel treeModel = loader
        .modelFromFile(loader.getFile("dtrees/models/test-model-2.csv"));
    PlainEvaluator evaluator = new PlainEvaluator(treeModel);
    Assert.assertEquals(BigInteger.valueOf(1),
        evaluator.evaluate(TestUtils.toBitIntegers(new int[]{1, 1, 0, 1, 0, 0, 0})));
    Assert.assertEquals(BigInteger.valueOf(2),
        evaluator.evaluate(TestUtils.toBitIntegers(new int[]{1, 1, 0, 0, 0, 0, 0})));
    Assert.assertEquals(BigInteger.valueOf(3),
        evaluator.evaluate(TestUtils.toBitIntegers(new int[]{1, 0, 0, 0, 1, 0, 0})));
    Assert.assertEquals(BigInteger.valueOf(4),
        evaluator.evaluate(TestUtils.toBitIntegers(new int[]{1, 0, 0, 0, 0, 0, 0})));
    Assert.assertEquals(BigInteger.valueOf(5),
        evaluator.evaluate(TestUtils.toBitIntegers(new int[]{0, 0, 1, 0, 0, 1, 0})));
    Assert.assertEquals(BigInteger.valueOf(6),
        evaluator.evaluate(TestUtils.toBitIntegers(new int[]{0, 0, 1, 0, 0, 0, 0})));
    Assert.assertEquals(BigInteger.valueOf(7),
        evaluator.evaluate(TestUtils.toBitIntegers(new int[]{0, 0, 0, 0, 0, 0, 1})));
    Assert.assertEquals(BigInteger.valueOf(8),
        evaluator.evaluate(TestUtils.toBitIntegers(new int[]{0, 0, 0, 0, 0, 0, 0})));
  }

}
