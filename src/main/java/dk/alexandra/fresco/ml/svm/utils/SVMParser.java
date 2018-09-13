package dk.alexandra.fresco.ml.svm.utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import dk.alexandra.fresco.ml.svm.SVMModel;

public class SVMParser {
  // Multiply weights by this number and round to nearest integer
  private static int PRECISION;

  private List<Double> bias;
  private List<List<Double>> supportVectors;

  public SVMParser(int precision) {
    PRECISION = precision;
  }

  /**
   * Parse a CSV file where the first line contains the bias and every following line the support
   * vectors.
   *
   * @param fileName
   *          Name of the CSV file
   * @return A model based on the data of the file
   */
  public SVMModel parseFile(String fileName) {
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
      return new SVMModel(supportVectors, bias, PRECISION);

    } catch (FileNotFoundException ex) {
      System.out.println("Unable to open file '" + fileName + "'");
    } catch (IOException ex) {
      System.out.println("Error reading file '" + fileName + "'");
    }
    return null;
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

  // private SVMModel makeSVMModel() {
  // List<BigInteger> bigBias = new ArrayList<>(bias.size());
  // for (Double currentDouble : bias) {
  // // We must multiply the bias with PRECISION again since the inner products have been shifted
  // // by precision twice as they consist of the sum of the product of shifted values
  // bigBias.add(convertToBigInteger(currentDouble).multiply(BigInteger.valueOf(PRECISION)));
  // }
  //
  // List<List<BigInteger>> bigSupportvectors = new ArrayList<>(supportVectors.size());
  // for (List<Double> currentVector : supportVectors) {
  // List<BigInteger> currentBigVector = new ArrayList<>(currentVector.size());
  // for (Double currentDouble : currentVector) {
  // currentBigVector.add(convertToBigInteger(currentDouble));
  // }
  // bigSupportvectors.add(currentBigVector);
  // }
  // return new SVMModel(bigSupportvectors, bigBias);
  // }


  public static void main(String args[]) {
    SVMParser parser = new SVMParser(1);
    System.out.println(parser.parseFile(args[0]));
  }
}
