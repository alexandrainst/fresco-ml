package dk.alexandra.fresco.ml.svm.utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import dk.alexandra.fresco.framework.util.Pair;
import dk.alexandra.fresco.ml.svm.SVMModel;

public class SVMParser {
  // Multiply weights by this number and round to nearest integer; this is how we convert from
  // decimal numbers to integers
  private final int scaling;

  private List<Double> bias;
  private List<List<Double>> supportVectors;

  public SVMParser(int scaling) {
    this.scaling = scaling;
  }

  /**
   * Parse a CSV file where the first line contains the bias and every following line the support
   * vectors.
   *
   * @param fileName
   *          Name of the CSV file
   * @return A model based on the data of the file
   */
  public SVMModel parseModelFromFile(String fileName) {
    try {
      CSVParser parser = CSVParser.parse(new File(fileName), Charset.defaultCharset(),
          CSVFormat.DEFAULT);
      List<CSVRecord> records = parser.getRecords();
      parser.close();
      // The bias is stored in the first line
      bias = parseLine(records.get(0));
      // The rest of the lines now contain the support vectors
      records.remove(0);
      setSupportvectors(records);
      return new SVMModel(supportVectors, bias, scaling);

    } catch (FileNotFoundException ex) {
      throw new RuntimeException("Could not load file " + fileName);
    } catch (IOException ex) {
      throw new RuntimeException("Could not read file " + fileName);
    }
  }

  /**
   * Parse a csv file of test data. The format is that the first line contains the expected
   * categories, as an integer. The following lines contain the test data as decimal numbers, one
   * test/classification per line with one feature per cell. Thus the first number in the first line
   * indicated the expected category of the first test. Note the categories do not have to be
   * contiguous, i.e. from 0 to n for n support vectors in the model. It can instead be from 0 to n'
   * where n'>n. This for example occur if the test file has been constructed from the full subset
   * of data. However, it is required that once the unique category numbers are sorted the i'th
   * category number expresses the the category of the i'th support vector in the model which the
   * test data is run on.
   *
   * Decimal numbers are scaled to big integers using a multiplicative constant and then rounding.
   *
   * @param fileName
   *          The path and filename of the test data
   * @return A pair where the first element is the list of expected categories and the second is a
   *         list of list of the test data as big integers
   */
  public Pair<List<BigInteger>, List<List<BigInteger>>> parseFeatures(String fileName) {
    try {
      CSVParser parser = CSVParser.parse(new File(fileName), Charset.defaultCharset(),
          CSVFormat.DEFAULT);
      List<CSVRecord> records = parser.getRecords();
      CSVRecord trueValues = records.remove(0);
      parser.close();

      List<Integer> listOfVal = new ArrayList<>();
      for (String val : trueValues) {
        listOfVal.add((int) Double.parseDouble(val));
      }
      // Compute a sorted list of unique category integers
      Collections.sort(listOfVal);
      listOfVal = listOfVal.stream().distinct().collect(Collectors.toList());

      List<BigInteger> truthValues = new ArrayList<>();
      // Map the i'th category integer to the i'th category
      for (String val : trueValues) {
        truthValues.add(BigInteger.valueOf(listOfVal.indexOf((int) Double.parseDouble(val))));
      }

      List<List<BigInteger>> testValues = new ArrayList<>();
      for (int i = 0; i < records.size(); i++) {
        List<BigInteger> currentTest = new ArrayList<>();
        for (int j = 0; j < records.get(i).size(); j++) {
          currentTest.add(new BigDecimal(records.get(i).get(j)).multiply(new BigDecimal(scaling))
              .toBigInteger());
        }
        testValues.add(currentTest);
      }
      return new Pair<List<BigInteger>, List<List<BigInteger>>>(truthValues, testValues);

    } catch (IOException e) {
      throw new RuntimeException("Could not read file " + fileName);
    }
  }

  private List<Double> parseLine(CSVRecord line) {
    List<Double> res = new ArrayList<>();
    for (String val : line) {
      res.add(Double.parseDouble(val));
    }
    return res;
  }

  private void setSupportvectors(List<CSVRecord> inputLines) {
    supportVectors = new ArrayList<>();
    for (CSVRecord currentLine : inputLines) {
      supportVectors.add(parseLine(currentLine));
    }
  }
}
