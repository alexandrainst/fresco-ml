package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.ml.dtrees.utils.ModelLoader;
import java.io.IOException;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Assert;
import org.junit.Test;

public class TestDecisionTrees {

  @Test
  public void testPlainEvaluator() throws IOException {
    ModelLoader loader = new ModelLoader();
    DecisionTreeModel treeModel = loader
        .modelFromFile(loader.getFile("dtrees/models/test-model-1.csv"));
    PlainEvaluator evaluator = new PlainEvaluator(treeModel);
    Assert.assertEquals(BigInteger.valueOf(1),
        evaluator.evaluate(toBitIntegers(new int[]{11, 3, 5, 7})));
    Assert.assertEquals(BigInteger.valueOf(2),
        evaluator.evaluate(toBitIntegers(new int[]{11, 0, 5, 7})));
    Assert.assertEquals(BigInteger.valueOf(3),
        evaluator.evaluate(toBitIntegers(new int[]{5, 0, 5, 12})));
    Assert.assertEquals(BigInteger.valueOf(4),
        evaluator.evaluate(toBitIntegers(new int[]{4, 0, 5, 7})));
  }

  @Test
  public void testPlainEvaluatorDepthFour() throws IOException {
    ModelLoader loader = new ModelLoader();
    DecisionTreeModel treeModel = loader
        .modelFromFile(loader.getFile("dtrees/models/test-model-2.csv"));
    PlainEvaluator evaluator = new PlainEvaluator(treeModel);
    Assert.assertEquals(BigInteger.valueOf(1),
        evaluator.evaluate(toBitIntegers(new int[]{1, 1, 0, 1, 0, 0, 0})));
    Assert.assertEquals(BigInteger.valueOf(2),
        evaluator.evaluate(toBitIntegers(new int[]{1, 1, 0, 0, 0, 0, 0})));
    Assert.assertEquals(BigInteger.valueOf(3),
        evaluator.evaluate(toBitIntegers(new int[]{1, 0, 0, 0, 1, 0, 0})));
    Assert.assertEquals(BigInteger.valueOf(4),
        evaluator.evaluate(toBitIntegers(new int[]{1, 0, 0, 0, 0, 0, 0})));
    Assert.assertEquals(BigInteger.valueOf(5),
        evaluator.evaluate(toBitIntegers(new int[]{0, 0, 1, 0, 0, 1, 0})));
    Assert.assertEquals(BigInteger.valueOf(6),
        evaluator.evaluate(toBitIntegers(new int[]{0, 0, 1, 0, 0, 0, 0})));
    Assert.assertEquals(BigInteger.valueOf(7),
        evaluator.evaluate(toBitIntegers(new int[]{0, 0, 0, 0, 0, 0, 1})));
    Assert.assertEquals(BigInteger.valueOf(8),
        evaluator.evaluate(toBitIntegers(new int[]{0, 0, 0, 0, 0, 0, 0})));
  }

  private static List<BigInteger> toBitIntegers(int[] indexes) {
    return Arrays.stream(indexes).mapToObj(BigInteger::valueOf)
        .collect(Collectors.toCollection(() -> new ArrayList<>(indexes.length)));
  }

}
