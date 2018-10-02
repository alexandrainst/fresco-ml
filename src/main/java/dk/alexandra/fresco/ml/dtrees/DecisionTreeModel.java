package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.framework.util.Pair;
import java.math.BigInteger;
import java.util.List;

/**
 * Representation of a decision tree model.
 */
public class DecisionTreeModel {

  private final int depth;
  private final List<List<BigInteger>> featureIndexes;
  private final List<List<BigInteger>> weights;
  private final List<BigInteger> categories;

  public DecisionTreeModel(int depth,
      List<List<BigInteger>> featureIndexes,
      List<List<BigInteger>> weights,
      List<BigInteger> categories) {
    this.depth = depth;
    this.featureIndexes = featureIndexes;
    this.weights = weights;
    this.categories = categories;
  }

  public DecisionTreeModel(List<List<BigInteger>> featureIndexes,
      List<List<BigInteger>> weights, List<BigInteger> categories) {
    this(featureIndexes.size() + 1, featureIndexes, weights, categories);
  }

  @Override
  public String toString() {
    return "DecisionTreeModelOpen{" +
        "depth=" + depth +
        ", featureIndexes=" + featureIndexes +
        ", weights=" + weights +
        ", categories=" + categories +
        '}';
  }

  public int getDepth() {
    return depth;
  }

  public List<List<BigInteger>> getFeatureIndexes() {
    return featureIndexes;
  }

  public List<List<BigInteger>> getWeights() {
    return weights;
  }

  public List<BigInteger> getCategories() {
    return categories;
  }

  /**
   * Returns feature index and weight given depth d in tree and node index within layer.
   */
  public Pair<BigInteger, BigInteger> getEntry(int d, int idx) {
    BigInteger weight = weights
        .get(d)
        .get(idx);
    BigInteger featureIndex = featureIndexes.get(d).get(idx);
    return new Pair<>(featureIndex, weight);
  }

}
