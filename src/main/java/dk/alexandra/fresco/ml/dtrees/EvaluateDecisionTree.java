package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.value.SInt;
import java.util.ArrayList;
import java.util.List;

/**
 * Computation for evaluating a decision tree model on a feature vector.
 */
public class EvaluateDecisionTree implements Computation<SInt, ProtocolBuilderNumeric> {

  private final DecisionTreeModelClosed treeModel;
  private final List<DRes<SInt>> featureVector;

  public EvaluateDecisionTree(
      DecisionTreeModelClosed treeModel,
      List<DRes<SInt>> featureVector) {
    this.treeModel = treeModel;
    this.featureVector = featureVector;
  }

  @Override
  public DRes<SInt> buildComputation(ProtocolBuilderNumeric builder) {
    // TODO zero check
    return builder.par(par -> {
      List<List<DRes<SInt>>> featureIndexes = treeModel.getFeatureIndexes();
      List<DRes<SInt>> selectedFeatures = new ArrayList<>(treeModel.getNumberInternalNodes());
      for (List<DRes<SInt>> featureIndex : featureIndexes) {
        selectedFeatures.add(par.advancedNumeric().innerProduct(featureVector, featureIndex));
      }
      return () -> selectedFeatures;
    }).par((par, selectedFeatures) -> {
      List<DRes<SInt>> weights = treeModel.getWeights();
      List<DRes<SInt>> lessThanFlags = new ArrayList<>(weights.size());
      for (int i = 0; i < weights.size(); i++) {
        lessThanFlags.add(par.comparison().compareLT(selectedFeatures.get(i), weights.get(i)));
      }
      return () -> lessThanFlags;
    }).par((par, lessThanFlags) -> {
      return null;
    });
  }
}
