package dk.alexandra.fresco.ml.dtrees;

import static dk.alexandra.fresco.ml.dtrees.InputDecisionTree.flat;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.MaliciousException;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.util.Pair;
import dk.alexandra.fresco.framework.value.OInt;
import dk.alexandra.fresco.framework.value.SInt;
import java.util.ArrayList;
import java.util.List;

/**
 * Computation for secret-sharing all parameters of a decision tree. <p>This should be run by party
 * not holding the tree model.</p>
 */
public class InputDecisionTreeAsReceiver implements
    Computation<DecisionTreeModelClosed, ProtocolBuilderNumeric> {

  private final int depth;
  private final int featureVectorSize;
  private final int inputPartyId;

  public InputDecisionTreeAsReceiver(int depth, int featureVectorSize, int otherPartyId) {
    this.depth = depth;
    this.featureVectorSize = featureVectorSize;
    this.inputPartyId = otherPartyId;
  }

  /**
   * Receives secret-shares of feature indexes.
   */
  private List<List<DRes<SInt>>> inputFeatureIndexes(ProtocolBuilderNumeric builder,
      int numberOfIndexes) {
    List<List<DRes<SInt>>> featureIndexesClosed = new ArrayList<>(numberOfIndexes);
    for (int i = 0; i < numberOfIndexes; i++) {
      List<DRes<SInt>> featureIndexBits = new ArrayList<>(featureVectorSize);
      for (int j = 0; j < featureVectorSize; j++) {
        featureIndexBits.add(builder.numeric().input(null, inputPartyId));
      }
      featureIndexesClosed.add(featureIndexBits);
    }
    return featureIndexesClosed;
  }

  /**
   * Receives list of secrets and returns result as an undeferred list.
   */
  private List<DRes<SInt>> input(ProtocolBuilderNumeric builder, int numberOfValues) {
    List<DRes<SInt>> secrets = new ArrayList<>(numberOfValues);
    for (int i = 0; i < numberOfValues; i++) {
      secrets.add(builder.numeric().input(null, inputPartyId));
    }
    return secrets;
  }

  @Override
  public DRes<DecisionTreeModelClosed> buildComputation(ProtocolBuilderNumeric builder) {
    return builder.par(par -> {
      int numberLeafNodes = (1 << (depth - 1));
      int numberInternalNodes = numberLeafNodes - 1;
      List<List<DRes<SInt>>> featureIndexesClosed = inputFeatureIndexes(par,
          numberInternalNodes);
      List<DRes<SInt>> weightsClosed = input(par, numberInternalNodes);
      List<DRes<SInt>> categoriesClosed = input(par, numberLeafNodes);
      DecisionTreeModelClosed closedModel = new DecisionTreeModelClosed(
          depth,
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
              return seq.numeric().openAsOInt(and, par.getBasicNumericContext().getMyId());
            })
        );
      }
      Pair<DecisionTreeModelClosed, List<DRes<OInt>>> pair = new Pair<>(model, openedBits);
      return () -> pair;
    }).seq((seq, pair) -> {
      for (DRes<OInt> checkBit : pair.getSecond()) {
        if (!seq.getOIntArithmetic().isZero(checkBit.out())) {
          throw new MaliciousException("Bit value was not a bit");
        }
      }
      return pair::getFirst;
    });
  }
}
