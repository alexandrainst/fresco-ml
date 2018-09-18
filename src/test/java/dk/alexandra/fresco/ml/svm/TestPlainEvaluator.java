package dk.alexandra.fresco.ml.svm;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
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
    List<Double> first = new ArrayList<>();
    List<Double> second = new ArrayList<>();
    first.add(4.2);
    first.add(13.37);
    first.add(0.0);
    second.add(4.3);
    second.add(0.0);
    second.add(13.37);

    PlainEvaluator evaluator = new PlainEvaluator(null, null);
    Method innerProduct = PlainEvaluator.class.getDeclaredMethod("innerProduct", List.class, List.class);
    innerProduct.setAccessible(true);
    double returnValue = (double) innerProduct.invoke(evaluator, first, second);
    assertEquals(18.06, returnValue, 0.001);
  }

  @Test
  public void testBreast() throws IOException {
    String filename = getClass().getClassLoader().getResource("svms/models/breastModel.csv")
        .getFile();
    CSVParser csvParser = CSVParser.parse(new File(filename), Charset.defaultCharset(),
        CSVFormat.DEFAULT);
    List<CSVRecord> modelRecords = csvParser.getRecords();
    csvParser.close();
    // First record contains the bias
    double bias = new Double(modelRecords.remove(0).get(0));
    List<Double> biasList = new ArrayList<>();
    biasList.add(bias);
    PlainEvaluator eval = new PlainEvaluator(PlainEvaluator.CSVListToDouble(modelRecords),
        biasList);

    filename = getClass().getClassLoader().getResource("svms/models/breastTest.csv")
        .getFile();
    CSVParser testParser = CSVParser.parse(new File(filename), Charset.defaultCharset(),
        CSVFormat.DEFAULT);
    List<CSVRecord> records = testParser.getRecords();
    List<List<Double>> decimalRecords = PlainEvaluator.CSVListToDouble(records);

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
      assertEquals((int) expectedValues.get(i), res);
    }
  }

}
