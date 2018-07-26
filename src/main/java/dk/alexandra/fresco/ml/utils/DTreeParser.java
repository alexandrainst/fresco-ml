package dk.alexandra.fresco.ml.utils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import dk.alexandra.fresco.framework.util.Pair;

public class DTreeParser {
  private int depth = -1;

  private List<String> features;
  private List<String> categories;

  private List<List<Integer>> featureIdxs;
  private List<List<Double>> weightsIdxs;
  private List<Integer> categoriesIdxs;

  public DTreeParser() {
  }

  public void parseFile(String fileName) {

    try {
      FileReader fileReader = new FileReader(fileName);
      BufferedReader reader = new BufferedReader(fileReader);
      List<String> collection = reader.lines().collect(Collectors.toList());
      fileReader.close();

      setFeatures(collection.stream());
      System.out.println(features);
      setCategories(collection.stream());
      System.out.println(categories);
      setDepth(collection);
      System.out.println(depth);
      constructTree(collection);
      System.out.println(featureIdxs);
      System.out.println(weightsIdxs);
      System.out.println(categoriesIdxs);

    } catch (FileNotFoundException ex) {
      System.out.println("Unable to open file '" + fileName + "'");
    } catch (IOException ex) {
      System.out.println("Error reading file '" + fileName + "'");
    }
  }

  private void constructTree(List<String> list) {
    featureIdxs = new ArrayList<>();
    weightsIdxs = new ArrayList<>();
    categoriesIdxs = new ArrayList<>();
    // Iterate through the fully balanced binary tree, except the category layer
    for (int i = 0; i < depth - 1; i++) {
      List<Integer> featureLayer = new ArrayList<>();
      List<Double> weightLayer = new ArrayList<>();
      for (int j = 0; j < 1 << i; j++) {
        int currentIdx = (1 << i) + j;
        // Find node with 2x node index to find the current node's weight
        Pair<Integer, Double> currentNode = findNode(2 * currentIdx, list);
        if (currentNode != null) {
          featureLayer.add(currentNode.getFirst());
          weightLayer.add(currentNode.getSecond());
        } else {
          // The current node is terminal and we must insert a dummy node
          featureLayer.add(-1);
          weightLayer.add(0.0);
        }
      }
      featureIdxs.add(featureLayer);
      weightsIdxs.add(weightLayer);
    }
    // Set categories
    for (int i = 0; i < 1 << (depth - 1); i++) {
      int nodeIdx = (1 << (depth - 1)) + i;
      // Set category based on node index. And proceed up the tree if it does not exist
      Pair<Integer, Double> currentNode = findNode(nodeIdx, list);
      while (currentNode == null) {
        // Find parent index
        nodeIdx >>= 1;
        currentNode = findNode(nodeIdx, list);
      }
      // Find the whole line for this node so we can find the category
      String category = getCategory(findLine(nodeIdx, list));
      categoriesIdxs.add(categories.indexOf(category));
    }

  }

  private void setDepth(List<String> list) {
    int maxNode = -1;
    for (int i = 6; i < list.size(); i++) {
      int nodeIdx = getNodeIdx(list.get(i));
      if (nodeIdx > maxNode) {
        maxNode = nodeIdx;
      }
    }
    depth = (int) Math.ceil(Math.log(maxNode) / Math.log(2));
  }

  private void setFeatures(Stream<String> stream) throws IOException {
    features = new ArrayList<String>();
    // Skip the first 6 lines as the tree starts on line 6 and we want to skip the root as well
    stream.skip(6).forEach(line -> {
      String feature = getFeature(line);
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
      String category = getCategory(line);
      // Go to next line if we have already added the category
      if (!categories.contains(category)) {
        // The number we will associate with the feature
        categories.add(category);
      }
    });
    // Sort the list alphabetically to be sure that the order of attributes does not leak info on
    // the tree
    categories.sort(null);
  }

  private int getNodeIdx(String line) {
    line = line.trim();
    int nodeEnd = line.indexOf(')');
    return Integer.parseInt(line.substring(0, nodeEnd));
  }

  private String getFeature(String line) {
    line = line.trim();
    // String after the first whitespace contains the feature of the node
    int feaStart = line.indexOf(' ');
    int feaEnd = -1;
    if (getNodeIdx(line) % 2 == 0) {
      // Even indexed nodes have the < comparison
      feaEnd = line.indexOf('<', feaStart);
    } else {
      // Odd indexed nodes have the > comparison
      feaEnd = line.indexOf('>', feaStart);
    }
    // Plus 1 for skipping whitespace
    return line.substring(feaStart + 1, feaEnd);
  }

  private String getCategory(String line) {
    line = line.trim();
    // The last integer has 10 digits so we can easily start at last index minus 3
    // We must skip 2 white spaces
    int temp = line.lastIndexOf(' ', line.length() - 3);
    int catEnd = line.lastIndexOf(' ', temp - 1);
    int catStart = line.lastIndexOf(' ', catEnd - 1);
    // Plus 1 to skip whitespace
    return line.substring(catStart + 1, catEnd);
  }

  private double getWeight(String line) {
    line = line.trim();
    int weightStart = -1;
    int weightEnd = -1;
    // String after the first whitespace contains the feature of the node
    int temp = line.indexOf(' ');
    // And after the second we have the weight if we are in an even node
    if (getNodeIdx(line) % 2 == 0) {
      weightStart = line.indexOf(' ', temp + 1);
      weightEnd = line.indexOf(' ', weightStart + 1);
    } else {
      // For even indexes it is after =
      weightStart = line.indexOf('=', temp + 1);
      weightEnd = line.indexOf(' ', weightStart + 1);
    }
    // Plus 1 for skipping whitespace
    return Double.parseDouble(line.substring(weightStart + 1, weightEnd));
  }

  private Pair<Integer, Double> findNode(int index, List<String> list) {
    Integer featureNum = null;
    Double weightVal = null;
    String line = findLine(index, list);
    if (line == null) {
      return null;
    }
    String feature = getFeature(line);
    featureNum = features.lastIndexOf(feature);
    weightVal = getWeight(line);
    return new Pair<>(featureNum, weightVal);
  }

  private String findLine(int index, List<String> list) {
    // Skip header and start at root node
    for (int i = 5; i < list.size(); i++) {
      String currentLine = list.get(i);
      if (getNodeIdx(currentLine) == index) {
        return currentLine;
      }
    }
    return null;
  }

  public static void main(String args[]) {
    DTreeParser parser = new DTreeParser();
    parser.parseFile(args[0]);
  }
}