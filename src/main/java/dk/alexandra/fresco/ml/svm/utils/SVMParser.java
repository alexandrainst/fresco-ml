package dk.alexandra.fresco.ml.svm.utils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
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
      FileReader fileReader = new FileReader(fileName);
      BufferedReader reader = new BufferedReader(fileReader);
      List<String> collection = reader.lines().filter(line -> !line.trim().isEmpty()).collect(
          Collectors.toList());
      fileReader.close();
      // The bias is stored in the first line
      bias = parseLine(collection.get(0));
      // The rest of the lines now contain the support vectors
      collection.remove(0);
      setSupportvectors(collection);
      return makeSVMModel();

    } catch (FileNotFoundException ex) {
      System.out.println("Unable to open file '" + fileName + "'");
    } catch (IOException ex) {
      System.out.println("Error reading file '" + fileName + "'");
    }
    return null;
  }

  private List<Double> parseLine(String line) {
    List<Double> res = new ArrayList<>();
    line = line.trim();
    int currentIndex = 0;
    while (line.indexOf(',', currentIndex) >= 0) {
      int valueEnd = line.indexOf(',', currentIndex);
      String substring = line.substring(currentIndex, valueEnd);
      res.add(Double.parseDouble(substring.trim()));
      currentIndex = valueEnd + 1;
    }
    // Add the last value
    String substring = line.substring(currentIndex);
    res.add(Double.parseDouble(substring.trim()));
    return res;
  }

  private void setSupportvectors(List<String> inputLines) {
    supportVectors = new ArrayList<>();
    for (String currentLine : inputLines) {
      supportVectors.add(parseLine(currentLine));
    }
  }

  private SVMModel makeSVMModel() {
    List<BigInteger> bigBias = new ArrayList<>(bias.size());
    for (Double currentDouble : bias) {
      bigBias.add(convertToBigInteger(currentDouble));
    }

    List<List<BigInteger>> bigSupportvectors = new ArrayList<>(supportVectors.size());
    for (List<Double> currentVector : supportVectors) {
      List<BigInteger> currentBigVector = new ArrayList<>(currentVector.size());
      for (Double currentDouble : currentVector) {
        currentBigVector.add(convertToBigInteger(currentDouble));
      }
      bigSupportvectors.add(currentBigVector);
    }

    return new SVMModel(bigSupportvectors, bigBias);
  }

  private BigInteger convertToBigInteger(Double input) {
    // We use BigDecimal to avoid loss of precision when converting to integer
    BigDecimal currentBigDouble = new BigDecimal(input);
    // "Shift" to become an "integer"
    currentBigDouble = currentBigDouble.multiply(new BigDecimal(PRECISION));
    // Round down to integer
    return currentBigDouble.toBigInteger();
  }

  public static void main(String args[]) {
    SVMParser parser = new SVMParser(1);
    System.out.println(parser.parseFile(args[0]));
  }
}
