package dk.alexandra.fresco.ml.svm;

import java.io.File;
import java.io.IOException;
import java.math.BigInteger;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.junit.Assert;

import dk.alexandra.fresco.framework.Application;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.TestThreadRunner.TestThread;
import dk.alexandra.fresco.framework.TestThreadRunner.TestThreadFactory;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.sce.resources.ResourcePool;
import dk.alexandra.fresco.framework.util.Pair;
import dk.alexandra.fresco.framework.value.SInt;
import dk.alexandra.fresco.ml.svm.utils.SVMParser;

public class SVMComputationTests {

  private static Application<BigInteger, ProtocolBuilderNumeric> constructApp(
      SVMModel svmModel, List<BigInteger> dataInputVectorOpen) {
    int modelInputPartyId = 1;
    int dataInputPartyId = 2;
    int dataInputVectorSize = dataInputVectorOpen.size();

    return root -> {
      DRes<SVMModelClosed> closedModelD;
      if (root.getBasicNumericContext().getMyId() == modelInputPartyId) {
        InputSVMAsSender f = new InputSVMAsSender(svmModel, modelInputPartyId);
        closedModelD = root.par(f);
      } else {
        InputSVMAsReceiver f = new InputSVMAsReceiver(dataInputVectorSize, svmModel
            .getNumSupportVectors(), modelInputPartyId);
        closedModelD = root.par(f);
      }

      DRes<List<DRes<SInt>>> dataVectorD = root.collections().closeList(dataInputVectorOpen,
          dataInputPartyId);
      return root.seq(seq -> {
        SVMModelClosed model = closedModelD.out();
        List<DRes<SInt>> inputData = dataVectorD.out();
        DRes<BigInteger> category = seq.seq(new EvaluateSVM(model, inputData));
        return category;
      });
    };
  }



  public static class TestEvaluateSVM<ResourcePoolT extends ResourcePool> extends
      TestThreadFactory<ResourcePoolT, ProtocolBuilderNumeric> {

    @Override
    public TestThread<ResourcePoolT, ProtocolBuilderNumeric> next() {
      return new TestThread<ResourcePoolT, ProtocolBuilderNumeric>() {
        private void testFiles(String modelFilename, String testFilename, int scaling, int amount)
            throws IOException {
          SVMParser svmParser = new SVMParser(scaling);
          SVMModel model = svmParser.parseModelFromFile(modelFilename);

          Pair<List<BigInteger>, List<List<BigInteger>>> parsedTests = svmParser.parseFeatures(
              testFilename);
          List<BigInteger> trueValues = parsedTests.getFirst();
          List<List<BigInteger>> inputValues = parsedTests.getSecond();

          CSVParser testParser = CSVParser.parse(new File(testFilename), Charset.defaultCharset(),
              CSVFormat.DEFAULT);
          List<CSVRecord> records = testParser.getRecords();
          testParser.close();
          // Remove the true values as they are the first line in the file
          CSVRecord trueValuesRecord = records.remove(0);
          List<List<Double>> inputValuesDouble = CSVListToDouble(records);

          CSVParser csvParser = CSVParser.parse(new File(modelFilename), Charset.defaultCharset(),
              CSVFormat.DEFAULT);
          List<CSVRecord> modelRecords = csvParser.getRecords();
          csvParser.close();
          // The first line contains the bias
          List<Double> biasList = PlainEvaluator.CSVToDouble(modelRecords.get(0));
          // Remove the first line, which is the bias
          modelRecords.remove(0);
          List<List<Double>> modelList = PlainEvaluator.CSVListToDouble(modelRecords);
          PlainEvaluator eval = new PlainEvaluator(modelList, biasList);

          int misses = 0;
          List<Integer> expectedList = new ArrayList<>();
          for (int i = 0; i < inputValues.size() && i < amount; i++) {
            BigInteger actual = runApplication(constructApp(model, inputValues.get(i)));
            int expected = eval.evaluate(inputValuesDouble.get(i));
            expectedList.add(actual.intValue());
            Assert.assertEquals(BigInteger.valueOf(expected), actual);

            if (!actual.equals(trueValues.get(i))) {
              misses++;
            }
          }
          // Check that accuracy is above 90%
          int maxFaults = (int) (inputValues.size() * 0.1);
          System.out.println("faults " + misses + ", inputs " + Math.min(inputValues.size(),
              amount));
          Assert.assertTrue(misses < maxFaults);
        }

        private List<List<Double>> CSVListToDouble(List<CSVRecord> records) {
          return records.stream().map(record -> CSVToDouble(record)).collect(Collectors.toList());
        }

        private List<Double> CSVToDouble(CSVRecord record) {
          List<Double> res = new ArrayList<>(record.size());
          for (int j = 0; j < record.size(); j++) {
            res.add(new Double(record.get(j)));
          }
          return res;
        }

        @Override
        public void test() throws IOException {
          String modelFilename, testFilename;
          modelFilename = getClass().getClassLoader().getResource(
              "svms/models/newModel.csv").getFile();
          testFilename = getClass().getClassLoader().getResource(
              "svms/models/newData.csv").getFile();
          testFiles(modelFilename, testFilename, 1000, 400);

          // modelFilename = getClass().getClassLoader().getResource(
          // "svms/models/cifar2048.csv").getFile();
          // testFilename = getClass().getClassLoader().getResource(
          // "svms/models/smallcifar2048-test.csv").getFile();
          // testFiles(modelFilename, testFilename, 100);

//          modelFilename = getClass().getClassLoader().getResource(
//              "svms/models/mit2048.csv").getFile();
//          testFilename = getClass().getClassLoader().getResource(
//              "svms/models/smallmit2048-test.csv").getFile();
//          testFiles(modelFilename, testFilename, 100);
        }
      };
    }
  }
}
