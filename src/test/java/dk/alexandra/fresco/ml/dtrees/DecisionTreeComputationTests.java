package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.framework.Application;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.TestThreadRunner.TestThread;
import dk.alexandra.fresco.framework.TestThreadRunner.TestThreadFactory;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.sce.resources.ResourcePool;
import dk.alexandra.fresco.framework.util.ExceptionConverter;
import dk.alexandra.fresco.ml.dtrees.utils.ModelLoader;
import java.math.BigInteger;

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
            return () -> BigInteger.ONE;
          };
          runApplication(testApplication);
        }
      };
    }
  }

}
