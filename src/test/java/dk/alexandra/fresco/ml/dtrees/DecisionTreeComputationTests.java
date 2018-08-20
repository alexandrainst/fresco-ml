package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.framework.Application;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.TestThreadRunner.TestThread;
import dk.alexandra.fresco.framework.TestThreadRunner.TestThreadFactory;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.sce.resources.ResourcePool;
import dk.alexandra.fresco.framework.util.ExceptionConverter;
import dk.alexandra.fresco.framework.value.SInt;
import dk.alexandra.fresco.ml.dtrees.utils.ModelLoader;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.List;
import org.junit.Assert;

public class DecisionTreeComputationTests {

  private static Application<BigInteger, ProtocolBuilderNumeric> constructApp(
      DecisionTreeModel treeModel,
      List<BigInteger> featureVectorOpen) {
    int treeInputPartyId = 1;
    int featureInputPartyId = 2;
    int featureVectorSize = featureVectorOpen.size();

    return root -> {
      DRes<DecisionTreeModelClosed> closedModelD;
      if (root.getBasicNumericContext().getMyId() == treeInputPartyId) {
        InputDecisionTree f = new InputDecisionTree(treeModel, featureVectorSize, treeInputPartyId);
        closedModelD = root.seq(f);
      } else {
        InputDecisionTreeAsReceiver f = new InputDecisionTreeAsReceiver(treeModel.getDepth(),
            featureVectorSize, treeInputPartyId);
        closedModelD = root.seq(f);
      }

      DRes<List<DRes<SInt>>> featureVectorD = root.collections().closeList(featureVectorOpen,
          featureInputPartyId);
      return root.seq(seq -> {
        DecisionTreeModelClosed model = closedModelD.out();
        List<DRes<SInt>> features = featureVectorD.out();
        DRes<SInt> category = seq.seq(new EvaluateDecisionTree(model, features));
        return seq.numeric().open(category);
      });
    };
  }

  public static class TestEvaluateDecisionTree<ResourcePoolT extends ResourcePool>
      extends TestThreadFactory<ResourcePoolT, ProtocolBuilderNumeric> {

    @Override
    public TestThread<ResourcePoolT, ProtocolBuilderNumeric> next() {
      return new TestThread<ResourcePoolT, ProtocolBuilderNumeric>() {
        @Override
        public void test() {
          ModelLoader loader = new ModelLoader();
          DecisionTreeModel treeModel = ExceptionConverter
              .safe(() -> loader.modelFromFile(loader.getFile("dtrees/models/test-model-1.csv")),
                  "Couldn't read model");
          List<List<BigInteger>> featureVectorsOpen = Arrays.asList(
              TestUtils.toBitIntegers(new int[]{11, 3, 5, 7}),
              TestUtils.toBitIntegers(new int[]{11, 0, 5, 7}),
              TestUtils.toBitIntegers(new int[]{5, 0, 5, 12}),
              TestUtils.toBitIntegers(new int[]{4, 0, 5, 7}));

          for (List<BigInteger> currentFeatureVectorOpen : featureVectorsOpen) {
            BigInteger actual = runApplication(constructApp(treeModel,
                currentFeatureVectorOpen));
            PlainEvaluator evaluator = new PlainEvaluator(treeModel);
            BigInteger expected = evaluator.evaluate(currentFeatureVectorOpen);
            Assert.assertEquals(expected, actual);
          }
        }
      };
    }
  }

  public static class TestEvaluateDecisionTreeFour<ResourcePoolT extends ResourcePool>
      extends TestThreadFactory<ResourcePoolT, ProtocolBuilderNumeric> {

    @Override
    public TestThread<ResourcePoolT, ProtocolBuilderNumeric> next() {
      return new TestThread<ResourcePoolT, ProtocolBuilderNumeric>() {
        @Override
        public void test() {
          ModelLoader loader = new ModelLoader();
          DecisionTreeModel treeModel = ExceptionConverter
              .safe(() -> loader.modelFromFile(loader.getFile("dtrees/models/test-model-2.csv")),
                  "Couldn't read model");
          List<List<BigInteger>> featureVectorsOpen = Arrays.asList(
              TestUtils.toBitIntegers(new int[]{1, 1, 0, 1, 0, 0, 0}),
              TestUtils.toBitIntegers(new int[]{1, 1, 0, 0, 0, 0, 0}),
              TestUtils.toBitIntegers(new int[]{1, 0, 0, 0, 1, 0, 0}),
              TestUtils.toBitIntegers(new int[]{1, 0, 0, 0, 0, 0, 0}),
              TestUtils.toBitIntegers(new int[]{0, 0, 1, 0, 0, 1, 0}),
              TestUtils.toBitIntegers(new int[]{0, 0, 1, 0, 0, 0, 0}),
              TestUtils.toBitIntegers(new int[]{0, 0, 0, 0, 0, 0, 1}),
              TestUtils.toBitIntegers(new int[]{0, 0, 0, 0, 0, 0, 0}));

          for (List<BigInteger> currentFeatureVectorOpen : featureVectorsOpen) {
            BigInteger actual = runApplication(constructApp(treeModel,
                currentFeatureVectorOpen));
            PlainEvaluator evaluator = new PlainEvaluator(treeModel);
            BigInteger expected = evaluator.evaluate(currentFeatureVectorOpen);
            Assert.assertEquals(expected, actual);
          }
        }
      };
    }
  }

  public static class TestEvaluateDecisionTreeFive<ResourcePoolT extends ResourcePool> extends
      TestThreadFactory<ResourcePoolT, ProtocolBuilderNumeric> {

    @Override
    public TestThread<ResourcePoolT, ProtocolBuilderNumeric> next() {
      return new TestThread<ResourcePoolT, ProtocolBuilderNumeric>() {
        @Override
        public void test() {
          ModelLoader loader = new ModelLoader();
          DecisionTreeModel treeModel = ExceptionConverter.safe(() -> loader.modelFromFile(loader
              .getFile("dtrees/models/test-model-3.csv")), "Couldn't read model");
          List<List<BigInteger>> featureVectorsOpen = Arrays.asList(TestUtils.toBitIntegers(
              new int[]{0, 0, 1, 0, 0, 0, 0}));

          for (List<BigInteger> currentFeatureVectorOpen : featureVectorsOpen) {
            BigInteger actual = runApplication(constructApp(treeModel, currentFeatureVectorOpen));
            PlainEvaluator evaluator = new PlainEvaluator(treeModel);
            BigInteger expected = evaluator.evaluate(currentFeatureVectorOpen);
            Assert.assertEquals(expected, actual);
          }
        }
      };
    }
  }

  public static class TestEvaluateDecisionTreeSix<ResourcePoolT extends ResourcePool> extends
      TestThreadFactory<ResourcePoolT, ProtocolBuilderNumeric> {

    @Override
    public TestThread<ResourcePoolT, ProtocolBuilderNumeric> next() {
      return new TestThread<ResourcePoolT, ProtocolBuilderNumeric>() {
        @Override
        public void test() {
          ModelLoader loader = new ModelLoader();
          DecisionTreeModel treeModel = ExceptionConverter.safe(() -> loader.modelFromFile(loader
              .getFile("dtrees/models/test-model-4.csv")), "Couldn't read model");
          List<List<BigInteger>> featureVectorsOpen = Arrays.asList(TestUtils.toBitIntegers(
              new int[]{0, 5, 4, 9, 12, 17, 11}));

          for (List<BigInteger> currentFeatureVectorOpen : featureVectorsOpen) {
            BigInteger actual = runApplication(constructApp(treeModel, currentFeatureVectorOpen));
            PlainEvaluator evaluator = new PlainEvaluator(treeModel);
            BigInteger expected = evaluator.evaluate(currentFeatureVectorOpen);
            Assert.assertEquals(expected, actual);
          }
        }
      };
    }
  }
}
