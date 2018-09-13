package dk.alexandra.fresco.ml.svm;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.math.BigInteger;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.junit.Test;

public class TestPlainEvaluator {

  @Test
  public void testInnerProduct() throws NoSuchMethodException, SecurityException,
      IllegalAccessException, IllegalArgumentException, InvocationTargetException {
    List<BigInteger> first = new ArrayList<>();
    List<BigInteger> second = new ArrayList<>();
    first.add(new BigInteger("42"));
    first.add(new BigInteger("1337"));
    first.add(new BigInteger("0"));
    second.add(new BigInteger("43"));
    second.add(new BigInteger("0"));
    second.add(new BigInteger("1337"));

    PlainEvaluator evaluator = new PlainEvaluator(null);
    Method innerProduct = PlainEvaluator.class.getDeclaredMethod("innerProduct", List.class, List.class);
    innerProduct.setAccessible(true);
    BigInteger returnValue = (BigInteger) innerProduct.invoke(evaluator, first, second);
    assertEquals(new BigInteger("1806"), returnValue);
  }

  @Test
  public void testBreast() throws IOException {
    String filename = getClass().getClassLoader().getResource("svms/models/breastModel.csv")
        .getFile();
    CSVParser csvParser = CSVParser.parse(new File(filename), Charset.defaultCharset(),
        CSVFormat.DEFAULT);
    List<CSVRecord> modelRecords = csvParser.getRecords();
    csvParser.close();
    double bias = new Double(modelRecords.get(0).get(modelRecords.get(0).size() - 1));
    List<Double> biasList = new ArrayList<>();
    biasList.add(bias);
    SVMModel model = new SVMModel(CSVListToDouble(modelRecords), biasList, 100000);
    PlainEvaluator eval = new PlainEvaluator(model);

    filename = getClass().getClassLoader().getResource("svms/models/breastTest.csv")
        .getFile();
    CSVParser testParser = CSVParser.parse(new File(filename), Charset.defaultCharset(),
        CSVFormat.DEFAULT);
    List<CSVRecord> records = testParser.getRecords();
    List<List<Double>> decimalRecords = CSVListToDouble(records);

    filename = getClass().getClassLoader().getResource("svms/models/breastTrue.csv").getFile();
    CSVParser trueParser = CSVParser.parse(new File(filename), Charset.defaultCharset(),
        CSVFormat.DEFAULT);
    List<CSVRecord> trueRecords = trueParser.getRecords();
    List<Integer> expectedValues = trueRecords.stream().map(rec -> {
      if (rec.get(0).equals("B")) {
       return 0;
     } else {
       return 1;
     }
    }).collect(Collectors.toList());
    testParser.close();

    for (int i = 0; i < decimalRecords.size(); i++) {
      int res = eval.evaluate(decimalRecords.get(i));
      // partial = partial.add(new BigDecimal(modelRecords.get(j).get(modelRecords.get(j).size()
      // - 1)));
      // if (partial.compareTo(BigDecimal.ZERO) > 0) {
      // maxIdx = 1;
      // } else {
      // maxIdx = 0;
      // }
      // }
      // System.out.println("expected " + expectedValues.get(i) + "true " + res);
      assertEquals((int) expectedValues.get(i), res);
    }
  }


  private static List<List<Double>> CSVListToDouble(List<CSVRecord> records) {
    List<List<Double>> res = new ArrayList<List<Double>>(records.size());
    for (int i = 0; i < records.size(); i++) {
      res.add(CSVToDouble(records.get(i)));
    }
    return res;
  }

  private static List<Double> CSVToDouble(CSVRecord record) {
    List<Double> res = new ArrayList<>();
    for (int i = 0; i < record.size(); i++) {
      res.add(new Double(record.get(i)));
    }
    return res;
  }
}
