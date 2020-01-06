package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.ComputationParallel;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.value.SInt;
import java.util.ArrayList;
import java.util.List;

/**
 * Computation for secret-sharing all parameters of a decision tree. <p>This should be run by party
 * not holding the tree model.</p>
 */
public class InputDecisionTreeAsReceiver implements
    ComputationParallel<DecisionTreeModelClosed, ProtocolBuilderNumeric> {

  private final int depth;
  private final int featureVectorSize;
  private final int inputPartyId;

  public InputDecisionTreeAsReceiver(int depth, int featureVectorSize, int inputPartyId) {
    this.depth = depth;
    this.featureVectorSize = featureVectorSize;
    this.inputPartyId = inputPartyId;
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
    int numberLeafNodes = (1 << (depth - 1));
    int numberInternalNodes = numberLeafNodes - 1;

    List<List<DRes<SInt>>> featureIndexesClosed = inputFeatureIndexes(builder, numberInternalNodes);
    List<DRes<SInt>> weightsClosed = input(builder, numberInternalNodes);
    List<DRes<SInt>> categoriesClosed = input(builder, numberLeafNodes);

    DecisionTreeModelClosed closedModel = new DecisionTreeModelClosed(
        depth,
        featureIndexesClosed,
        weightsClosed,
        categoriesClosed);
    return () -> closedModel;
  }
}
