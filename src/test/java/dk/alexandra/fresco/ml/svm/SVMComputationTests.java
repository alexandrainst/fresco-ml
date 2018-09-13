package dk.alexandra.fresco.ml.svm;

import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import dk.alexandra.fresco.framework.Application;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.TestThreadRunner.TestThread;
import dk.alexandra.fresco.framework.TestThreadRunner.TestThreadFactory;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.sce.resources.ResourcePool;
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
        DRes<SInt> category = seq.seq(new EvaluateSVM(model, inputData));
        return seq.numeric().open(category);
      });
    };
  }

  public static class TestEvaluateSVM<ResourcePoolT extends ResourcePool> extends
      TestThreadFactory<ResourcePoolT, ProtocolBuilderNumeric> {

    @Override
    public TestThread<ResourcePoolT, ProtocolBuilderNumeric> next() {
      return new TestThread<ResourcePoolT, ProtocolBuilderNumeric>() {
        @Override
        public void test() throws IOException {
          int precision = 1000000;
          String filename = getClass().getClassLoader().getResource("svms/models/cifar2048.csv")
              .getFile();
          SVMParser parser = new SVMParser(precision);
          SVMModel model = parser.parseFile(filename);

          filename = getClass().getClassLoader().getResource("svms/models/smallcifar2048-test.csv")
              .getFile();
          CSVParser testParser = CSVParser.parse(new File(filename), Charset.defaultCharset(),
              CSVFormat.DEFAULT);
          List<CSVRecord> records = testParser.getRecords();
          testParser.close();
          // Remove the true values
          records.remove(0);
          // List<BigInteger> expectedValues = CSVToBigInteger(records.remove(0), precision);
          List<List<BigInteger>> inputVectorsOpen = CSVListToBigInteger(records, precision);

          for (List<BigInteger> currentInputVectorOpen : inputVectorsOpen) {
            BigInteger actual = runApplication(constructApp(model, currentInputVectorOpen));
            PlainEvaluator evaluator = new PlainEvaluator(model);
            // int expected = evaluator.evaluate(currentInputVectorOpen);
            // Assert.assertEquals(BigInteger.valueOf(expected), actual);
          }
        }
      };
    }

    private static List<List<BigDecimal>> CSVListToBigDecimal(List<CSVRecord> records) {
      List<List<BigDecimal>> res = new ArrayList<List<BigDecimal>>(records.size());
      for (int i = 0; i < records.size(); i++) {
        List<BigDecimal> currentList = new ArrayList<>();
        for (int j = 0; j < records.get(i).size(); j++) {
          currentList.add(new BigDecimal(records.get(i).get(j)));
        }
        res.add(currentList);
      }
      return res;
    }

    private static List<List<BigInteger>> CSVListToBigInteger(List<CSVRecord> records,
        int precision) {
      List<List<BigInteger>> res = new ArrayList<List<BigInteger>>(records.size());
      for (int i = 0; i < records.size(); i++) {
        res.add(CSVToBigInteger(records.get(i), precision));
      }
      return res;
    }

    private static List<BigInteger> CSVToBigInteger(CSVRecord record, int precision) {
      List<BigInteger> res = new ArrayList<>(record.size());
      for (int j = 0; j < record.size(); j++) {
        BigDecimal currentVal = new BigDecimal(record.get(j));
        res.add(currentVal.multiply(new BigDecimal(precision)).toBigInteger());
      }
      return res;
    }
  }
}
