package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.util.Pair;
import dk.alexandra.fresco.framework.value.OInt;
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
    Computation<DecisionTreeModelClosed, ProtocolBuilderNumeric> {

  private final DecisionTreeModel treeModel;
  private final int featureVectorSize;
  private final int otherPartyId;

  public InputDecisionTree(DecisionTreeModel treeModel, int featureVectorSize, int otherPartyId) {
    this.treeModel = treeModel;
    this.featureVectorSize = featureVectorSize;
    this.otherPartyId = otherPartyId;
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
  static <T> List<T> flat(List<List<T>> nested) {
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
        featureIndexBits.add(
            builder.numeric().input(bit, builder.getBasicNumericContext().getMyId()));
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
      secrets.add(builder.numeric().input(value, builder.getBasicNumericContext().getMyId()));
    }
    return secrets;
  }

  @Override
  public DRes<DecisionTreeModelClosed> buildComputation(ProtocolBuilderNumeric builder) {
    List<BigInteger> featureIndexes = flat(treeModel.getFeatureIndexes());
    List<BigInteger> weights = flat(treeModel.getWeights());
    List<BigInteger> categories = treeModel.getCategories();

    return builder.par(par -> {
      List<List<DRes<SInt>>> featureIndexesClosed = inputFeatureIndexes(par, featureIndexes);
      List<DRes<SInt>> weightsClosed = input(par, weights);
      List<DRes<SInt>> categoriesClosed = input(par, categories);
      DecisionTreeModelClosed closedModel = new DecisionTreeModelClosed(
          treeModel.getDepth(),
          featureIndexesClosed,
          weightsClosed,
          categoriesClosed);
      return () -> closedModel;
    }).par((par, model) -> {
      List<DRes<SInt>> featureIndexesFlat = flat(model.getFeatureIndexes());
      List<DRes<OInt>> openedBits = new ArrayList<>(featureIndexesFlat.size());
      for (DRes<SInt> bit : featureIndexesFlat) {
        final DRes<SInt> finalBit = bit;
        openedBits.add(
            par.seq(seq -> {
              DRes<SInt> notBit = seq.numeric().subFromOpen(seq.getOIntFactory().one(), finalBit);
              DRes<SInt> and = seq.numeric().mult(finalBit, notBit);
              return seq.numeric().openAsOInt(and, otherPartyId);
            })
        );
      }
      Pair<DecisionTreeModelClosed, List<DRes<OInt>>> pair = new Pair<>(model, openedBits);
      return () -> pair;
    }).seq((seq, pair) -> pair::getFirst);
  }
}
