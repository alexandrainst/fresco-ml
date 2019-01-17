package dk.alexandra.fresco.ml.dtrees.utils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import dk.alexandra.fresco.framework.util.Pair;
import dk.alexandra.fresco.ml.dtrees.DecisionTreeModel;

public class DTreeParser {

  private class Node {

    protected int feature;
    protected int category;
    protected double weight;
    protected boolean switchAround;

    Node(int feature, int category, double weight, boolean switchAround) {
      this.feature = feature;
      this.category = category;
      this.weight = weight;
      this.switchAround = switchAround;
    }
  }

  // The amount of non-empty lines in the file before the first node
  private static final int META_LINES = 4;

  // Multiply weights by this number and round to nearest integer
  private final int scaling;

  private int depth = -1;

  private List<String> features;
  private List<String> categories;

  private List<List<Integer>> featureIdxs;
  private List<List<Double>> weightsIdxs;
  private List<List<Boolean>> switchIdxs;
  private List<List<Integer>> categoriesIdxs;

  public DTreeParser(int scaling) {
    this.scaling = scaling;
  }

  /**
   * Parse a file of features for evaluation. The file format expected is a CSV where the first
   * lines contains the names (as strings) of the features. Each following line is then a list of
   * features.
   *
   * @param fileName The filename, including path, of the feature file.
   * @return A list of lists of BigIntegers ready for usage in an MPC computation.
   */
  public List<List<BigInteger>> parseFeatures(String fileName) {
    try {
      FileReader fileReader;
      fileReader = new FileReader(fileName);
      BufferedReader reader = new BufferedReader(fileReader);
      List<String> collection = reader.lines().filter(line -> !line.trim().isEmpty()).collect(
          Collectors.toList());
      reader.close();
      // Remove the first line, which is meta data
      collection.remove(0);
      List<List<BigInteger>> featureMatrix = new ArrayList<List<BigInteger>>(collection.size());
      for (int i = 0; i < collection.size(); i++) {
        List<BigInteger> currentListBigInteger = new ArrayList<>();
        List<String> csvList = Arrays.asList(collection.get(i).split(","));
        for (int j = 0; j < csvList.size(); j++) {
          currentListBigInteger.add((new BigDecimal(csvList.get(j)).multiply(
              new BigDecimal(scaling)).toBigInteger()));
        }
        featureMatrix.add(currentListBigInteger);
      }
      return featureMatrix;

    } catch (FileNotFoundException e) {
      throw new RuntimeException("Could not load file " + fileName);
    } catch (IOException e) {
      throw new RuntimeException("Could not read file " + fileName);
    }
  }

  /**
   * Parse a file containing a decision tree model based on the .txt output of the tree from R.
   * However with the change that the first line of the file is a comma separated list of all the
   * possible features of the model.
   */
  public DecisionTreeModel parseModel(String fileName) {

    try {
      FileReader fileReader = new FileReader(fileName);
      BufferedReader reader = new BufferedReader(fileReader);
      List<String> collection = reader.lines().filter(line -> !line.trim().isEmpty()).collect(
          Collectors.toList());
      fileReader.close();
      setFeatures(collection);
      setCategories(collection.stream());
      setDepth(collection);
      constructTree(collection);
      // We need to mirror the tree because the R model assumes you go left if the comparison is
      // true and we assume you go right
      mirrorTree();

      return makeTreeModel();

    } catch (FileNotFoundException ex) {
      throw new RuntimeException("Could not load file " + fileName);
    } catch (IOException ex) {
      throw new RuntimeException("Could not read file " + fileName);
    }
  }

  private void mirrorTree() {
    for (int i = 0; i < featureIdxs.size(); i++) {
      Collections.reverse(featureIdxs.get(i));
      Collections.reverse(weightsIdxs.get(i));
      Collections.reverse(categoriesIdxs.get(i));
    }
  }

  private DecisionTreeModel makeTreeModel() {
    List<List<BigInteger>> bigFeatures = new ArrayList<>();
    List<List<BigInteger>> bigWeights = new ArrayList<>();
    // Skip the last layer as it is simply a dummy layer as it only contains category information
    for (int i = 0; i < featureIdxs.size() - 1; i++) {
      List<BigInteger> currentFeatures = new ArrayList<>();
      List<BigInteger> currentWeights = new ArrayList<>();
      for (int j = 0; j < featureIdxs.get(i).size(); j++) {
        currentFeatures.add(new BigInteger(String.valueOf(featureIdxs.get(i).get(j))));
        // Multiply with scaling to move to integers
        currentWeights.add(new BigInteger(String.valueOf((int) (scaling * weightsIdxs.get(
            i).get(j)))));
      }
      bigFeatures.add(currentFeatures);
      bigWeights.add(currentWeights);
    }
    List<BigInteger> bigCategories = new ArrayList<>();
    for (int i = 0; i < categoriesIdxs.get(categoriesIdxs.size() - 1).size(); i++) {
      bigCategories.add(
          new BigInteger(String.valueOf(categoriesIdxs.get(categoriesIdxs.size() - 1).get(i))));
    }
//    System.out.println(featureIdxs.size() - 1);
//    System.out.println(features.size());
    return new DecisionTreeModel(bigFeatures.size() + 1, features.size(),
        bigFeatures, bigWeights, bigCategories);
  }

  private void switchSubtree(int parentNodeIdx) {
    int leftChildIdx = 2 * parentNodeIdx;
    int childLayer = (int) Math.floor(Math.log(leftChildIdx) / Math.log(2));
    for (int i = childLayer; i < depth; i++) {
      switchLeftRight(leftChildIdx, 1 << (i - childLayer));
      leftChildIdx = 2 * leftChildIdx;
    }
  }

  private void switchLeftRight(int leftChildIdx, int length) {
    int layer = (int) Math.floor(Math.log(leftChildIdx) / Math.log(2));
    int leftChildOffset = leftChildIdx - (1 << layer);
    int rightChildOffset = leftChildOffset + length;
    // Switch nodes in current layer
    for (int i = 0; i < length; i++) {
      Integer leftFeature = featureIdxs.get(layer).get(leftChildOffset + i);
      Integer rightFeature = featureIdxs.get(layer).get(rightChildOffset + i);
      featureIdxs.get(layer).set(leftChildOffset + i, rightFeature);
      featureIdxs.get(layer).set(rightChildOffset + i, leftFeature);
      Double leftWeight = weightsIdxs.get(layer).get(leftChildOffset + i);
      Double rightWeight = weightsIdxs.get(layer).get(rightChildOffset + i);
      weightsIdxs.get(layer).set(leftChildOffset + i, rightWeight);
      weightsIdxs.get(layer).set(rightChildOffset + i, leftWeight);
      Integer leftCategory = categoriesIdxs.get(layer).get(leftChildOffset + i);
      Integer rightCategory = categoriesIdxs.get(layer).get(rightChildOffset + i);
      categoriesIdxs.get(layer).set(leftChildOffset + i, rightCategory);
      categoriesIdxs.get(layer).set(rightChildOffset + i, leftCategory);
    }
  }

  private void constructTree(List<String> list) {
    featureIdxs = new ArrayList<>();
    weightsIdxs = new ArrayList<>();
    categoriesIdxs = new ArrayList<>();
    switchIdxs = new ArrayList<>();
    // Iterate through the fully balanced binary tree, except the category layer
    for (int i = 0; i < depth; i++) {
      List<Integer> featureLayer = new ArrayList<>();
      List<Double> weightLayer = new ArrayList<>();
      List<Boolean> switchLayer = new ArrayList<>();
      List<Integer> categoryLayer = new ArrayList<>();
      for (int j = 0; j < 1 << i; j++) {
        int currentIdx = (1 << i) + j;
        // Find category
        Node currentNode = null;
        if (currentIdx > 1) { // Skip root since it will never have category
          currentNode = findNode(currentIdx, list);
          while (currentNode == null) {
            currentIdx = currentIdx / 2;
            currentNode = findNode(currentIdx, list);
          }
        }
        if (currentNode != null) {
          categoryLayer.add(currentNode.category);
        } else {
          categoryLayer.add(-1);
        }
        // Find node with 2x node index to find the current node's weight
        Node childNode = findNode(2 * currentIdx, list);
        if (childNode != null) {
          featureLayer.add(childNode.feature);
          weightLayer.add(childNode.weight);
          switchLayer.add(childNode.switchAround);
        } else {
          // The current node is terminal and we must insert a dummy node
          featureLayer.add(-1);
          weightLayer.add(0.0);
          switchLayer.add(false);
        }
      }
      featureIdxs.add(featureLayer);
      weightsIdxs.add(weightLayer);
      switchIdxs.add(switchLayer);
      categoriesIdxs.add(categoryLayer);
    }
    // can't use negative one indexes for dummy nodes since that leaks information during the index
    // check, so use 0 instead
    replaceNegativeOnes();
    // Do switches if needed
    for (int i = 0; i < depth; i++) {
      for (int j = 0; j < 1 << i; j++) {
        if (switchIdxs.get(i).get(j)) {
          int currentIdx = (1 << i) + j;
          switchSubtree(currentIdx);
        }
      }
    }
  }

  private void replaceNegativeOnes() {
    for (List<Integer> featureLayer : featureIdxs) {
      for (int i = 0; i < featureLayer.size(); i++) {
        if (featureLayer.get(i).equals(-1)) {
          featureLayer.set(i, 0);
        }
      }
    }
  }

  private void setDepth(List<String> list) {
    int maxNode = -1;
    // Skip meta lines along with the root
    for (int i = META_LINES + 1; i < list.size(); i++) {
      int nodeIdx = getNodeIdx(list.get(i));
      if (nodeIdx > maxNode) {
        maxNode = nodeIdx;
      }
    }
    depth = (int) Math.ceil(Math.log(maxNode) / Math.log(2));
  }

  private void setFeatures(List<String> list) {
    features = new ArrayList<>();
    String featureString = list.get(0).replace("\"", "");
    features = Arrays.asList(featureString.split(","));
  }

  private void setCategories(Stream<String> stream) {
    categories = new ArrayList<>();
    // Skip the first meta lines and an extra since we want to skip the root as well
    stream.skip(META_LINES + 1).forEach(line -> {
      String category = getCategory(line);
      // Go to next line if we have already added the category
      if (!categories.contains(category)) {
        // The number we will associate with the feature
        categories.add(category);
      }
    });
  }

  private int getNodeIdx(String line) {
    line = line.trim();
    int nodeEnd = line.indexOf(')');
    return Integer.parseInt(line.substring(0, nodeEnd));
  }

  private Pair<String, Boolean> getFeature(String line) {
    line = line.trim();
    // String after the first whitespace contains the feature of the node
    int feaStart = line.indexOf(' ');
    int feaEnd = -1;
    boolean shouldSwitch = false;
    if (getNodeIdx(line) % 2 == 0) {
      // Even indexed nodes have the < comparison
      feaEnd = line.indexOf('<', feaStart);
      if (feaEnd == -1) {
        // We have an annoying switch between < and >=
        feaEnd = line.indexOf('>', feaStart);
        shouldSwitch = true;
      }
    } else {
      // Odd indexed nodes have the > comparison
      feaEnd = line.indexOf('>', feaStart);
      if (feaEnd == -1) {
        // We have an annoying switch between < and >=
        feaEnd = line.indexOf('<', feaStart);
        shouldSwitch = true;
      }
    }
    // Plus 1 for skipping whitespace
    return new Pair<>(line.substring(feaStart + 1, feaEnd), shouldSwitch);
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

  private double getWeight(String line, boolean shouldSwitch) {
    line = line.trim();
    int weightStart = -1;
    int weightEnd = -1;
    // String after the first whitespace contains the feature of the node
    int temp = line.indexOf(' ');
    // And after the second we have the weight if we are in an even node
    if (getNodeIdx(line) % 2 == 0 ^ shouldSwitch) {
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

  private Node findNode(int index, List<String> list) {
    Integer featureNum;
    Double weightVal;
    String line = findLine(index, list);
    if (line == null) {
      return null;
    }
    Pair<String, Boolean> feature = getFeature(line);
    boolean shouldSwitch = feature.getSecond();
    featureNum = features.lastIndexOf(feature.getFirst());
    weightVal = getWeight(line, shouldSwitch);
    int category = categories.lastIndexOf(getCategory(line));
    return new Node(featureNum, category, weightVal, shouldSwitch);
  }

  private String findLine(int index, List<String> list) {
    // Skip header and start at root node
    for (int i = META_LINES; i < list.size(); i++) {
      String currentLine = list.get(i);
      if (getNodeIdx(currentLine) == index) {
        return currentLine;
      }
    }
    return null;
  }
}
