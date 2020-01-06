package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.value.SInt;
import java.util.List;

/**
 * Decision tree model with secret-shared parameters.
 */
public class DecisionTreeModelClosed {

  private final int depth;
  private final List<List<DRes<SInt>>> featureIndexes;
  private final List<DRes<SInt>> weights;
  private final List<DRes<SInt>> categories;

  public DecisionTreeModelClosed(int depth,
      List<List<DRes<SInt>>> featureIndexes,
      List<DRes<SInt>> weights,
      List<DRes<SInt>> categories) {
    this.depth = depth;
    this.featureIndexes = featureIndexes;
    this.weights = weights;
    this.categories = categories;
  }

  public int getDepth() {
    return depth;
  }

  public int getNumberInternalNodes() {
    return featureIndexes.size();
  }

  public int getNumberLeafNodes() {
    return categories.size();
  }

  public List<List<DRes<SInt>>> getFeatureIndexes() {
    return featureIndexes;
  }

  public List<DRes<SInt>> getWeights() {
    return weights;
  }

  public List<DRes<SInt>> getCategories() {
    return categories;
  }

  @Override
  public String toString() {
    return "DecisionTreeModelClosed{" +
        "depth=" + depth +
        ", featureIndexes=" + featureIndexes +
        ", weights=" + weights +
        ", categories=" + categories +
        '}';
  }

}
