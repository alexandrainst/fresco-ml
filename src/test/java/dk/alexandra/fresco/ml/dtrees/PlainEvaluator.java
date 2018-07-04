package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.framework.util.Pair;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

public class PlainEvaluator {

  private final DecisionTreeModel treeModel;

  public PlainEvaluator(DecisionTreeModel treeModel) {
    this.treeModel = treeModel;
  }

  // BigInteger evaluate(List<BigInteger> featureVector) {
  // int nodeIdx = 0;
  // Pair<BigInteger, BigInteger> currentNode = treeModel.getEntry(0, nodeIdx);
  // for (int d = 1; d < treeModel.getDepth(); d++) {
  // int featureIndex = currentNode.getFirst().intValueExact();
  // BigInteger featureToTest = featureVector.get(featureIndex);
  // BigInteger weight = currentNode.getSecond();
  // boolean goLeft = (featureToTest.compareTo(weight) >= 0);
  // nodeIdx = (2 * nodeIdx) + (goLeft ? 0 : 1);
  // if (d < treeModel.getDepth() - 1) {
  // currentNode = treeModel.getEntry(d, nodeIdx);
  // }
  // }
  // return treeModel.getCategories().get(nodeIdx);
  // }

  /**
   * Emulate the secure evaluation algorithm
   *
   * @param featureVector
   * @return
   */
  BigInteger evaluate(List<BigInteger> featureVector) {
    List<BigInteger> lessThanFlags = new ArrayList<BigInteger>();
    List<BigInteger> partialVal = new ArrayList<>(lessThanFlags.size());
    for (int j = 0; j < (1 << (treeModel.getDepth() - 1)) - 1; j++) {
      lessThanFlags.add(BigInteger.ONE); // placeholder
      partialVal.add(BigInteger.ONE); // placeholder
    }

    // Compute lessThan flags
    for (int i = 0; i < treeModel.getDepth() - 1; i++) {
      for (int j = 0; j < 1 << i; j++) {
        int index = (1 << i) + j - 1;
        Pair<BigInteger, BigInteger> currentNode = treeModel.getEntry(i, j);
        int featureIndex = currentNode.getFirst().intValueExact();
        BigInteger featureToTest = featureVector.get(featureIndex);
        BigInteger weight = currentNode.getSecond();
        String nodeFlag = featureToTest.compareTo(weight) < 0 ? "1" : "0";
        lessThanFlags.set(index, new BigInteger(nodeFlag));
      }
    }
    for (int j = 0; j < lessThanFlags.size(); j++) {
      partialVal.set(j, BigInteger.ONE);
    }
    int i = 1;
    while (i < treeModel.getDepth() - 1) {
      // Iterate over relevant layers
      for (int j = i - 1; j + 1 < treeModel.getDepth() - 1; j = j + 2 * i) {
        // Iterate over the elements in the layer
        // TODO burde være while der finder det dybeste lag
        int layerSize = Math.min((1 << (j + i)), (1 << (treeModel.getDepth() - 2)));
        int parentLayerSize = 1 << j;
        for (int k = 0; k < layerSize; k++) {
          // Find the index of the deepest node already computed in the subtree
          int currentLeafIdx = layerSize + k;
          int parentNodeIdx = parentLayerSize + ((k * parentLayerSize) / layerSize);// (((1 << (j +
                                                                                    // i)) + k) / (1
                                                                                    // <<
                                                                // i));
//          int subtreeNodeIdx = (((1 << (j + i)) + k) / (1 << (i - 1)));
//          while (subtreeNodeIdx - 1 > 2 * parentNodeIdx) {
//            subtreeNodeIdx = subtreeNodeIdx / 2;
//          }
          int subtreeNodeIdx = (2 * parentLayerSize) + ((k * 2 *parentLayerSize) / layerSize);
          // Adjust for 0-indexing
          // Check if we are left subtree
          BigInteger upperNode = subtreeNodeIdx % 2 == 0 ? BigInteger.ONE.subtract(lessThanFlags
              .get(parentNodeIdx - 1)) : lessThanFlags.get(parentNodeIdx - 1);
          // Compute partial product for upper subtree
          BigInteger upperPartial = upperNode.multiply(partialVal.get(parentNodeIdx - 1));
          BigInteger newPartial = upperPartial.multiply(partialVal.get(currentLeafIdx - 1));
          partialVal.set(currentLeafIdx - 1, newPartial);
        }
      }
      i = 2 * i;
    }
    BigInteger currentRes = null;
    for (i = 0; i < treeModel.getCategories().size(); i++) {
      // Compute indicator bit for category by first selecting parent index
      int parentNodeIdx = (1 << (treeModel.getDepth() - 2)) + (i / 2);
      // Negate parent bit if needed
      BigInteger parentIndicator = i % 2 == 0 ? BigInteger.ONE.subtract(lessThanFlags.get(
          parentNodeIdx - 1)) : lessThanFlags.get(parentNodeIdx - 1);
      // Then compute final indicator for the given leaf (category)
      BigInteger finalIndicator = parentIndicator.multiply(partialVal.get(parentNodeIdx - 1));
      // Multiply indicator with category
      BigInteger temp = finalIndicator.multiply(treeModel.getCategories().get(i));
      // Compute partial sum
      if (currentRes == null) {
        currentRes = temp;
      } else {
        currentRes = currentRes.add(temp);
      }
    }
    return currentRes;
  }

}
