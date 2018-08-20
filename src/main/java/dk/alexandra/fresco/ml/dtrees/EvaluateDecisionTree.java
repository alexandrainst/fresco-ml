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
        // Compute lessThan for each node in the tree, based on the selected feature
        lessThanFlags.add(par.comparison().compareLT(selectedFeatures.get(i), weights.get(i)));
        // Construct a placeholder list used in the process of computing the final output
        // TODO should this be close?
        partialVal.add(par.numeric().known(BigInteger.ONE));
      }
      Pair<Pair<List<DRes<SInt>>, List<DRes<SInt>>>, Integer> pairIntegerPair = new Pair<>(
          new Pair<>(lessThanFlags, partialVal), 1);
      return () -> pairIntegerPair;
    }).whileLoop(pair -> pair.getSecond() < treeModel.getDepth() - 1, (prevPar, pair) -> prevPar
        .par(par -> {
          List<DRes<SInt>> lessThanFlags = pair.getFirst().getFirst();
          List<DRes<SInt>> partialVal = pair.getFirst().getSecond();
          // Let i be an exponentially increasing chunks size
          int i = pair.getSecond();
          // Iterate over the chunks in the tree
          for (int j = i - 1; j + 1 < treeModel.getDepth() - 1; j = j + 2 * i) {
            // Compute the amount of element in the current layer
            int layerSize = Math.min((1 << (j + i)), (1 << (treeModel.getDepth() - 2)));
            // Compute the amounts of elements in the upper layer of the subtree
            int subtreeLayerSize = 1 << (j + 1);
            // Iterate over the elements in the layer of the given chunk
            for (int k = 0; k < layerSize; k++) {
              // Find the index of the deepest node already computed in the subtree
              int currentLeafIdx = layerSize + k;
              // Compute the index of the highest element in the chunk-subtree
              int subtreeNodeIdx = subtreeLayerSize + ((k * subtreeLayerSize) / layerSize);
              // Compute the index of parent of the chunk
              int parentNodeIdx = subtreeNodeIdx / 2;
              // TODO this is hacky, find cleaner way
              par.seq(seq -> {
                // this is a different scope and the state has changed...
                DRes<SInt> upperNode =
                    subtreeNodeIdx % 2 == 0 ? seq.logical().not(lessThanFlags.get(
                        parentNodeIdx - 1)) : lessThanFlags.get(parentNodeIdx - 1);
                // Compute partial product for upper subtree
                DRes<SInt> upperPartial = seq.logical().and(upperNode,
                    partialVal.get(parentNodeIdx - 1));
                DRes<SInt> newPartial = seq.logical()
                    .and(upperPartial, partialVal.get(currentLeafIdx - 1));
                partialVal.set(currentLeafIdx - 1, newPartial);
                return null;
              });
            }
          }
          Pair<Pair<List<DRes<SInt>>, List<DRes<SInt>>>, Integer> pairIntegerPair = new Pair<>(
              new Pair<>(lessThanFlags, partialVal), 2 * pair.getSecond());
          return () -> pairIntegerPair;
        })).seq((par, res) -> {
      DRes<SInt> currentRes = null;
      // Compute the correct category by iterating over each of them
      for (int i = 0; i < treeModel.getCategories().size(); i++) {
        List<DRes<SInt>> lessThanFlags = res.getFirst().getFirst();
        List<DRes<SInt>> partialVal = res.getFirst().getSecond();
        // Find the index of the inner node that is the parent of the i'th leaf (category)
        int parentNodeIdx = (1 << (treeModel.getDepth() - 2)) + (i / 2);
        // Negate the bit of the parent in case the leaf (category) is a left child
        DRes<SInt> parentIndicator = i % 2 == 0 ? par.logical().not(lessThanFlags.get(
            parentNodeIdx - 1)) : lessThanFlags.get(parentNodeIdx - 1);
        // Then compute final indicator for the given leaf (category)
        DRes<SInt> finalIndicator = par.logical().and(parentIndicator, partialVal.get(
            parentNodeIdx - 1));
        // Multiply indicator bit with category value
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
