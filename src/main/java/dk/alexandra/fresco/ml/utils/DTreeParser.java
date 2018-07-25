package dk.alexandra.fresco.ml.utils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class DTreeParser {
  private List<String> features;
  private List<String> categories;

  public DTreeParser() {
  }

  public void parseFile(String fileName) {

    try {
      FileReader fileReader = new FileReader(fileName);
      BufferedReader reader = new BufferedReader(fileReader);
      Stream<String> stream = reader.lines();
      setFeatures(stream);
      System.out.println(features);
      setCategories(reader.lines());
      reader.close();
    } catch (FileNotFoundException ex) {
      System.out.println("Unable to open file '" + fileName + "'");
    } catch (IOException ex) {
      System.out.println("Error reading file '" + fileName + "'");
    }
  }

  private void setFeatures(Stream<String> stream) throws IOException {
    features = new ArrayList<String>();
    // Skip the first 6 lines as the tree starts on line 6 and we want to skip the root as well
    stream.skip(6).forEach(line -> {
      line = line.trim();
      // Find node index
      int nodeEnd = line.indexOf(')');
      int nodeIdx = Integer.parseInt(line.substring(0, nodeEnd));
      // String after the first whitespace contains the feature of the node
      int feaStart = line.indexOf(' ');
      int feaEnd = -1;
      if (nodeIdx % 2 == 0) {
        // Even indexed nodes have the < comparison
        feaEnd = line.indexOf('<', feaStart);
      } else {
        // Odd indexed nodes have the > comparison
        feaEnd = line.indexOf('>', feaStart);
      }
      String feature = line.substring(feaStart, feaEnd);
      // Go to next line if we have already added the feature
      if (!features.contains(feature)) {
        // The number we will associate with the feature
        features.add(feature);
      }
    });
    // Sort the list alphabetically to be sure that the order of attributes does not leak info on the tree
    features.sort(null);
  }

  private void setCategories(Stream<String> stream) {
    categories = new ArrayList<String>();
    // Skip the first 6 lines as the tree starts on line 6 and we want to skip the root as well
    stream.skip(6).forEach(line -> {

    });
  }

  public static void main(String args[]) {
    DTreeParser parser = new DTreeParser();
    parser.parseFile(args[0]);
  }
}