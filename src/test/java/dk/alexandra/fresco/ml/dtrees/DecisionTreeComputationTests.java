package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.framework.Application;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.TestThreadRunner.TestThread;
import dk.alexandra.fresco.framework.TestThreadRunner.TestThreadFactory;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.sce.resources.ResourcePool;
import dk.alexandra.fresco.framework.util.ExceptionConverter;
import dk.alexandra.fresco.framework.value.SInt;
import dk.alexandra.fresco.ml.dtrees.utils.DTreeParser;
import dk.alexandra.fresco.ml.dtrees.utils.ModelLoader;
import dk.alexandra.fresco.ml.dtrees.PlainEvaluator;

import java.io.File;
import java.io.IOException;
import java.math.BigInteger;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.junit.Assert;

public class DecisionTreeComputationTests {

  private static Application<BigInteger, ProtocolBuilderNumeric> constructApp(
      DecisionTreeModel treeModel, List<BigInteger> featureVectorOpen) {
    int treeInputPartyId = 1;
    int featureInputPartyId = 2;
    int featureVectorSize = featureVectorOpen.size();

    return root -> {
      DRes<DecisionTreeModelClosed> closedModelD;
      if (root.getBasicNumericContext().getMyId() == treeInputPartyId) {
        InputDecisionTree f = new InputDecisionTree(treeModel, featureVectorSize,
            featureInputPartyId);
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

  public static class TestEvaluateDecisionTreeAdvancedModel<ResourcePoolT extends ResourcePool>
      extends TestThreadFactory<ResourcePoolT, ProtocolBuilderNumeric> {


    @Override
    public TestThread<ResourcePoolT, ProtocolBuilderNumeric> next() {
      return new TestThread<ResourcePoolT, ProtocolBuilderNumeric>() {
        @Override
        public void test() throws IOException {
          int scaling = 1000;
          DTreeParser parser = new DTreeParser(scaling);
          DecisionTreeModel treeModel = parser.parseModel(getClass().getClassLoader().getResource(
              "dtrees/models/breastModel.txt").getPath());
          List<List<BigInteger>> testValues = parser.parseFeatures(getClass().getClassLoader()
              .getResource("dtrees/models/breastTest.csv").getPath());

          CSVParser csvParser = CSVParser.parse(new File(getClass().getClassLoader().getResource(
              "dtrees/models/breastPredictions.csv").getFile()), Charset
              .defaultCharset(), CSVFormat.DEFAULT);
          List<CSVRecord> predictionRecords = csvParser.getRecords();
          csvParser.close();
          List<BigInteger> predictions = predictionRecords.stream().map(rec -> rec.get(0).equals(
              "B") ? BigInteger.ZERO : BigInteger.ONE).collect(Collectors.toList());

          for (int i = 0; i < testValues.size(); i++) {
            BigInteger actual = runApplication(constructApp(treeModel, testValues.get(i)));
            PlainEvaluator evaluator = new PlainEvaluator(treeModel);
            BigInteger expected = evaluator.evaluate(testValues.get(i));

            Assert.assertEquals(expected, actual);
            Assert.assertEquals(predictions.get(i), actual);
          }
        }
      };
    }
  }

}
