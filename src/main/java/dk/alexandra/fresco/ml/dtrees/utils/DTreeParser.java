package dk.alexandra.fresco.ml.dtrees.utils;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.math.BigInteger;
import java.util.ArrayList;
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
  private static int PRECISION;

  private int depth = -1;

  private List<String> features;
  private List<String> categories;

  // the number of features in the training set; this is not the same as the number of distinct
  // feature indexes, since some of the features might not be used in the tree
  private int numOriginalFeatures;
  private List<List<Integer>> featureIdxs;
  private List<List<Double>> weightsIdxs;
  private List<List<Boolean>> switchIdxs;
  private List<List<Integer>> categoriesIdxs;

  public DTreeParser(int precision) {
    PRECISION = precision;
  }

  public DecisionTreeModel parseFile(String fileName) {

    try {
      FileReader fileReader = new FileReader(fileName);
      BufferedReader reader = new BufferedReader(fileReader);
      List<String> collection = reader.lines().filter(line -> !line.trim().isEmpty()).collect(
          Collectors.toList());
      fileReader.close();
      setFeatures(collection.stream());
      setCategories(collection.stream());
      setDepth(collection);
      constructTree(collection);
      // We need to mirror the tree because the R model assumes you go left if the comparison is
      // true and we assume you go right
      mirrorTree();

      return makeTreeModel();

    } catch (FileNotFoundException ex) {
      System.out.println("Unable to open file '" + fileName + "'");
    } catch (IOException ex) {
      System.out.println("Error reading file '" + fileName + "'");
    }
    return null;
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
        // Multiply with PRECISION to move to integers
        currentWeights.add(new BigInteger(String.valueOf((int) (PRECISION * weightsIdxs.get(
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
    return new DecisionTreeModel(depth, numOriginalFeatures, bigFeatures,
        bigWeights, bigCategories);
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
    numOriginalFeatures = Integer.parseInt(list.get(0));
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

  private void setFeatures(Stream<String> stream) {
    features = new ArrayList<>();
    // Skip the first meta lines and an extra since we want to skip the root as well
    stream.skip(META_LINES + 1).forEach(line -> {
      Pair<String, Boolean> featurePair = getFeature(line);
      String feature = featurePair.getFirst();
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
    // Sort the list alphabetically to be sure that the order of attributes does not leak info on
    // the tree
    categories.sort(null);
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

  public static void main(String args[]) {
    DTreeParser parser = new DTreeParser(1);
    parser.parseFile(args[0]);
  }
}
