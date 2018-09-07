package dk.alexandra.fresco.ml.svm.utils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import dk.alexandra.fresco.ml.svm.SVMModel;

public class SVMParser {
  // Multiply weights by this number and round to nearest integer
  private static int PRECISION;

  public SVMParser(int precision) {
    PRECISION = precision;
  }

  public SVMModel parseFile(String fileName) {
    try {
      FileReader fileReader = new FileReader(fileName);
      BufferedReader reader = new BufferedReader(fileReader);
      List<String> collection = reader.lines().filter(line -> !line.trim().isEmpty()).collect(
          Collectors.toList());
      fileReader.close();
      setFeatures(collection.stream());
      return makeSVMModel();

    } catch (FileNotFoundException ex) {
      System.out.println("Unable to open file '" + fileName + "'");
    } catch (IOException ex) {
      System.out.println("Error reading file '" + fileName + "'");
    }
    return null;
  }

  private void setFeatures(Stream<String> stream) {

  }

  private SVMModel makeSVMModel() {
    return null;
  }

  public static void main(String args[]) {
    SVMParser parser = new SVMParser(1);
    parser.parseFile(args[0]);
  }
}
