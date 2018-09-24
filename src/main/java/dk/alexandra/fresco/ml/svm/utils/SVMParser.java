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
      System.out.println("Unable to open file '" + fileName + "'");
    } catch (IOException ex) {
      System.out.println("Error reading file '" + fileName + "'");
    }
    return null;
  }

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
      Collections.sort(listOfVal);
      listOfVal = listOfVal.stream().distinct().collect(Collectors.toList());

      List<BigInteger> truthValues = new ArrayList<>();
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
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
    return null;
  }

  /**
   * Turn a list of double into a list of BigIntegers as the secure evaluation requires. The doubles
   * get converted by first multiplying with the scaling factor and then rounded down.
   *
   * @param input
   *          The vector to convert
   * @return The converted vector
   */
  public List<BigInteger> parseFeaturesFromDouble(List<Double> input) {
    return input.stream().map(val -> new BigDecimal(val).multiply(new BigDecimal(scaling))
        .toBigInteger()).collect(Collectors.toList());
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
