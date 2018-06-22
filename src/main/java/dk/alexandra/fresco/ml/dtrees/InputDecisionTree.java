package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.ComputationParallel;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.value.SInt;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Computation for secret-sharing all parameters of a decision tree. <p>This should be run by party
 * holding the tree model.</p>
 */
public class InputDecisionTree implements
    ComputationParallel<DecisionTreeModelClosed, ProtocolBuilderNumeric> {

  private final DecisionTreeModel treeModel;
  private final int featureVectorSize;
  private final int inputPartyId;

  public InputDecisionTree(DecisionTreeModel treeModel, int featureVectorSize,
      int inputPartyId) {
    this.treeModel = treeModel;
    this.featureVectorSize = featureVectorSize;
    this.inputPartyId = inputPartyId;
  }

  /**
   * Turns index into selection bits.
   */
  private List<BigInteger> convertIndexToBits(BigInteger featureIndex) {
    List<BigInteger> bits = new ArrayList<>(featureVectorSize);
    int indexAsInteger = featureIndex.intValueExact();
    for (int i = 0; i < featureVectorSize; i++) {
      bits.add((i == indexAsInteger) ? BigInteger.ONE : BigInteger.ZERO);
    }
    return bits;
  }

  /**
   * Flattens nested list.
   */
  private <T> List<T> flat(List<List<T>> nested) {
    return nested.stream().flatMap(Collection::stream)
        .collect(Collectors.toCollection(ArrayList::new));
  }

  /**
   * Converts features indexes to bit and secret-shares.
   */
  private List<List<DRes<SInt>>> inputFeatureIndexes(ProtocolBuilderNumeric builder,
      List<BigInteger> featureIndexes) {
    List<List<DRes<SInt>>> featureIndexesClosed = new ArrayList<>(featureIndexes.size());
    for (BigInteger featureIndex : featureIndexes) {
      List<DRes<SInt>> featureIndexBits = new ArrayList<>(featureVectorSize);
      for (BigInteger bit : convertIndexToBits(featureIndex)) {
        featureIndexBits.add(builder.numeric().input(bit, inputPartyId));
      }
      featureIndexesClosed.add(featureIndexBits);
    }
    return featureIndexesClosed;
  }

  /**
   * Inputs list of secrets and returns result as an undeferred list.
   */
  private List<DRes<SInt>> input(ProtocolBuilderNumeric builder, List<BigInteger> values) {
    List<DRes<SInt>> secrets = new ArrayList<>(values.size());
    for (BigInteger value : values) {
      secrets.add(builder.numeric().input(value, inputPartyId));
    }
    return secrets;
  }

  @Override
  public DRes<DecisionTreeModelClosed> buildComputation(ProtocolBuilderNumeric builder) {
    List<BigInteger> featureIndexes = flat(treeModel.getFeatureIndexes());
    List<BigInteger> weights = flat(treeModel.getWeights());
    List<BigInteger> categories = treeModel.getCategories();

    List<List<DRes<SInt>>> featureIndexesClosed = inputFeatureIndexes(builder, featureIndexes);
    List<DRes<SInt>> weightsClosed = input(builder, weights);
    List<DRes<SInt>> categoriesClosed = input(builder, categories);

    DecisionTreeModelClosed closedModel = new DecisionTreeModelClosed(
        treeModel.getDepth(),
        featureIndexesClosed,
        weightsClosed,
        categoriesClosed);
    return () -> closedModel;
  }
}
