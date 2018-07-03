package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.util.Pair;
import dk.alexandra.fresco.framework.value.SInt;

import java.math.BigInteger;
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
      List<DRes<SInt>> partialVal = new ArrayList<>(weights.size());
      for (int i = 0; i < weights.size(); i++) {
        lessThanFlags.add(par.comparison().compareLT(selectedFeatures.get(i), weights.get(i)));
        partialVal.add(par.numeric().known(BigInteger.ONE));
      }
      return () -> new Pair<>(new Pair<>(lessThanFlags, partialVal), 1);
    }).whileLoop(pair -> pair.getSecond() <= treeModel.getDepth() - 2, (prevPar, pair) -> prevPar
        .par(par -> {
          List<DRes<SInt>> lessThanFlags = pair.getFirst().getFirst();
          // Preprocess partial indicator bits
          List<DRes<SInt>> partialVal = pair.getFirst().getSecond();
          // TODO is it necessary to construct a new object
          List<DRes<SInt>> newPartialVal = new ArrayList<>(partialVal.size());
          for (int i = 0; i < partialVal.size(); i++) {
            newPartialVal.add(partialVal.get(i)); // placeholder
          }
          // Iterate over the chunks
          // The index i will be the chunk size
          // for (int i = 1; i < treeModel.getDepth(); i = 2 * i) {
          int i = pair.getSecond();
          // Iterate over relevant layers
          for (int j = i - 1; j <= treeModel.getDepth() - 2; j = j + 2 * i) {
            // Iterate over the elements in the layer
            int layerSize = Math.min((1 << (j + i)), (1 << (treeModel.getDepth() - 2)));
            for (int k = 0; k < layerSize; k++) {
              // Find the index of the deepest node already computed in the subtree
              int currentLeafIdx = layerSize + k;
              int parentNodeIdx = (currentLeafIdx / (1 << i));
              int subtreeNodeIdx = (currentLeafIdx / (1 << (i - 1)));
              // Adjust for 0-indexing
              // Check if we are left subtree
              DRes<SInt> upperNode = subtreeNodeIdx % 2 == 0 ? par.logical().not(lessThanFlags.get(
                  parentNodeIdx - 1)) : lessThanFlags.get(parentNodeIdx - 1);
              // Compute partial product for upper subtree
              DRes<SInt> upperPartial = par.logical().and(upperNode, partialVal.get(parentNodeIdx
                  - 1));
              DRes<SInt> newPartial = par.logical().and(upperPartial, partialVal.get(currentLeafIdx
                  - 1));
              partialVal.set(currentLeafIdx - 1, newPartial);
            }
          }
          return () -> new Pair<>(new Pair<>(lessThanFlags, partialVal), 2 * pair.getSecond());
        })).par((par, res) -> {
          DRes<SInt> currentRes = null;
          for (int i = 0; i < treeModel.getCategories().size(); i++) {
            List<DRes<SInt>> lessThanFlags = res.getFirst().getFirst();
            // Preprocess partial indicator bits
            List<DRes<SInt>> partialVal = res.getFirst().getSecond();
            // Compute indicator bit for category by first selecting parent index
            int parentNodeIdx = (1 << (treeModel.getDepth() - 2)) + (i / 2);
            // Negate parent bit if needed
            DRes<SInt> parentIndicator = i % 2 == 0 ? par.logical().not(lessThanFlags.get(
                parentNodeIdx - 1)) : lessThanFlags.get(parentNodeIdx - 1);
            // Then compute final indicator for the given leaf (category)
            DRes<SInt> finalIndicator = par.logical().and(parentIndicator, partialVal.get(
                parentNodeIdx - 1));
            // Multiply indicator with category
            DRes<SInt> temp = par.numeric().mult(finalIndicator, treeModel.getCategories().get(i));
            // Compute partial sum
            if (currentRes == null) {
              currentRes = temp;
            } else {
              currentRes = par.numeric().add(currentRes, temp);
            }
          }
          return currentRes;
        });
  }
}
