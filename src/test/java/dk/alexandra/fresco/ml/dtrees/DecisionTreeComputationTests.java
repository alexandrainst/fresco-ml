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
import java.util.List;
import org.junit.Assert;

public class DecisionTreeComputationTests {

  public static class TestInputDecisionTree<ResourcePoolT extends ResourcePool>
      extends TestThreadFactory<ResourcePoolT, ProtocolBuilderNumeric> {

    @Override
    public TestThread<ResourcePoolT, ProtocolBuilderNumeric> next() {
      return new TestThread<ResourcePoolT, ProtocolBuilderNumeric>() {
        @Override
        public void test() {
          Application<BigInteger, ProtocolBuilderNumeric> testApplication = root -> {
            ModelLoader loader = new ModelLoader();
            DecisionTreeModel treeModel = ExceptionConverter
                .safe(() -> loader.modelFromFile(loader.getFile("dtrees/models/test-model-1.csv")),
                    "Couldn't read model");

            int inputPartyId = 1;
            int featureVectorSize = 5;

            DRes<DecisionTreeModelClosed> closedModel;
            if (root.getBasicNumericContext().getMyId() == inputPartyId) {
              InputDecisionTree f = new InputDecisionTree(treeModel, featureVectorSize,
                  inputPartyId);
              closedModel = root.seq(f);
            } else {
              InputDecisionTreeAsReceiver f = new InputDecisionTreeAsReceiver(treeModel.getDepth(),
                  featureVectorSize, inputPartyId);
              closedModel = root.seq(f);
            }
            // TODO actually test something
            return () -> BigInteger.ONE;
          };
          runApplication(testApplication);
        }
      };
    }
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
          List<BigInteger> featureVectorOpen = TestUtils.toBitIntegers(new int[] { 11, 3, 5, 7 });

          Application<BigInteger, ProtocolBuilderNumeric> testApplication = root -> {
            int treeInputPartyId = 1;
            int featureInputPartyId = 2;
            int featureVectorSize = featureVectorOpen.size();

            DRes<DecisionTreeModelClosed> closedModelD;
            if (root.getBasicNumericContext().getMyId() == treeInputPartyId) {
              InputDecisionTree f = new InputDecisionTree(treeModel, featureVectorSize,
                  treeInputPartyId);
              closedModelD = root.seq(f);
            } else {
              InputDecisionTreeAsReceiver f = new InputDecisionTreeAsReceiver(treeModel.getDepth(),
                  featureVectorSize, treeInputPartyId);
              closedModelD = root.seq(f);
            }

            DRes<List<DRes<SInt>>> featureVectorD = root.collections().closeList(
                featureVectorOpen,
                featureInputPartyId
            );
            return root.seq(seq -> {
              DecisionTreeModelClosed model = closedModelD.out();
              List<DRes<SInt>> features = featureVectorD.out();
              DRes<SInt> category = seq.seq(new EvaluateDecisionTree(model, features));
              return seq.numeric().open(category);
            });
          };
          BigInteger actual = runApplication(testApplication);
          PlainEvaluator evaluator = new PlainEvaluator(treeModel);
          BigInteger expected = evaluator.evaluate(featureVectorOpen);
          Assert.assertEquals(expected, actual);
        }
      };
    }
  }

}
