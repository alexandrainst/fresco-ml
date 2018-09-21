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
import java.math.BigDecimal;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
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
        InputDecisionTree f = new InputDecisionTree(treeModel, featureVectorSize, treeInputPartyId);
        closedModelD = root.par(f);
      } else {
        InputDecisionTreeAsReceiver f = new InputDecisionTreeAsReceiver(treeModel.getDepth(),
            featureVectorSize, treeInputPartyId);
        closedModelD = root.par(f);
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

    private static List<List<String>> transpose(List<List<String>> records) {
      List<List<String>> res = new ArrayList<>();
      for (int i = 0; i < records.get(0).size(); i++) {
        List<String> currentRecord = new ArrayList<>();
        for (int j = 0; j < records.size(); j++) {
          currentRecord.add(records.get(j).get(i));
        }
        res.add(currentRecord);
      }
      return res;
    }

    static class ListComparator implements Comparator<List<String>> {

      @Override
      public int compare(List<String> o1, List<String> o2) {
        // Compare on the first element of each list
        return o1.get(0).compareTo(o2.get(0));
      }

    }

    @Override
    public TestThread<ResourcePoolT, ProtocolBuilderNumeric> next() {
      return new TestThread<ResourcePoolT, ProtocolBuilderNumeric>() {
        @Override
        public void test() throws IOException {
          int scaling = 1000;
          DTreeParser parser = new DTreeParser(scaling);
          DecisionTreeModel treeModel = parser.parseFile(getClass().getClassLoader().getResource(
              "dtrees/models/breastModel.txt").getPath());
          File breastTest = new File(getClass().getClassLoader().getResource(
              "dtrees/models/breastTest.csv").getFile());
          CSVParser csvParser = CSVParser.parse(breastTest, Charset.defaultCharset(), CSVFormat.DEFAULT);
          List<CSVRecord> testRecords = csvParser.getRecords();
          csvParser.close();
          List<List<String>> stringTestRecords = new ArrayList<>();
          for (CSVRecord currentRec : testRecords) {
            List<String> currentList = new ArrayList<>();
            for (String val : currentRec) {
              currentList.add(val);
            }
            stringTestRecords.add(currentList);
          }
          stringTestRecords = transpose(stringTestRecords);
          Collections.sort(stringTestRecords, new ListComparator());
          stringTestRecords = transpose(stringTestRecords);

          // Remove the first line, which is meta data
          stringTestRecords.remove(0);
          List<List<Double>> testListDouble = new ArrayList<List<Double>>(stringTestRecords.size());
          List<List<BigInteger>> testListBigInteger = new ArrayList<List<BigInteger>>(
              stringTestRecords
              .size());

          csvParser = CSVParser.parse(new File(getClass().getClassLoader().getResource(
              "dtrees/models/breastPredictions.csv").getFile()), Charset
              .defaultCharset(), CSVFormat.DEFAULT);
          List<CSVRecord> predictionRecords = csvParser.getRecords();
          csvParser.close();
          List<BigInteger> predictions = new ArrayList<>();

          for (int i = 0; i < stringTestRecords.size(); i++) {
            List<Double> currentListDouble = new ArrayList<>();
            List<BigInteger> currentListBigInteger = new ArrayList<>();
            for (int j = 0; j < stringTestRecords.get(i).size(); j++) {
              currentListDouble.add(new Double(stringTestRecords.get(i).get(j)));
              currentListBigInteger.add((new BigDecimal(stringTestRecords.get(i).get(j)).multiply(
                  new BigDecimal(scaling)).toBigInteger()));
            }
            testListDouble.add(currentListDouble);
            testListBigInteger.add(currentListBigInteger);
            if (predictionRecords.get(i).get(0).equals("B")) {
              predictions.add(BigInteger.ZERO);
            } else {
              predictions.add(BigInteger.ONE);
            }
          }
          int wrongs = 0;
          List<Integer> vals = new ArrayList<>();
          for (int i = 0; i < testListBigInteger.size(); i++) {
            BigInteger actual = runApplication(constructApp(treeModel, testListBigInteger.get(i)));
            PlainEvaluator evaluator = new PlainEvaluator(treeModel);
            BigInteger expected = evaluator.evaluate(testListBigInteger.get(i));
            Assert.assertEquals(expected, actual);
            vals.add(actual.intValue());
            if (!predictions.get(i).equals(actual)) {
              wrongs++;
            }
            // Assert.assertEquals(predictions.get(i), actual);
          }
          System.out.println(vals);
          System.out.println(predictions);
          System.out.println("wrongs " + wrongs + ", totals " + predictions.size());
        }
      };
    }
  }

}
