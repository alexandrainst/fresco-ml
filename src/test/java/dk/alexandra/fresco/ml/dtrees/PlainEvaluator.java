package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.framework.util.Pair;
import java.math.BigInteger;
import java.util.List;

public class PlainEvaluator {

  private final DecisionTreeModel treeModel;

  public PlainEvaluator(DecisionTreeModel treeModel) {
    this.treeModel = treeModel;
  }

  BigInteger evaluate(List<BigInteger> featureVector) {
    int nodeIdx = 0;
    Pair<BigInteger, BigInteger> currentNode = treeModel.getEntry(0, nodeIdx);
    for (int d = 1; d < treeModel.getDepth(); d++) {
      int featureIndex = currentNode.getFirst().intValueExact();
      BigInteger featureToTest = featureVector.get(featureIndex);
      BigInteger weight = currentNode.getSecond();
      boolean goLeft = (featureToTest.compareTo(weight) >= 0);
      nodeIdx = (2 * nodeIdx) + (goLeft ? 0 : 1);
      if (d < treeModel.getDepth() - 1) {
        currentNode = treeModel.getEntry(d, nodeIdx);
      }
    }
    return treeModel.getCategories().get(nodeIdx);
  }

}
