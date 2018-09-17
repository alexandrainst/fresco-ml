package dk.alexandra.fresco.ml.svm.utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import dk.alexandra.fresco.ml.svm.SVMModel;

public class SVMParser {
  // Multiply weights by this number and round to nearest integer
  private int precision;

  private List<Double> bias;
  private List<List<Double>> supportVectors;

  public SVMParser(int precision) {
    this.precision = precision;
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
      return new SVMModel(supportVectors, bias, precision);

    } catch (FileNotFoundException ex) {
      System.out.println("Unable to open file '" + fileName + "'");
    } catch (IOException ex) {
      System.out.println("Error reading file '" + fileName + "'");
    }
    return null;
  }

  public List<BigInteger> parseFeaturesFromDouble(List<Double> input) {
    return input.stream().map(val -> new BigDecimal(val).multiply(new BigDecimal(precision))
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
