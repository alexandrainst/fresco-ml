package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.value.SInt;
import java.util.List;
import java.util.stream.Collectors;

/**
 * A computation to normalize a set of {@link WeightedModelParams}. I.e., divide each weighted
 * parameter by the weight.
 *
 * <p>
 * This assumes we are working in a field and proceeds in the following steps:
 * <ol>
 * <li>Compute the multiplicative inverse of the weight
 * <li>Multiply all parameters by the inverse weight.
 * </ol>
 * This method requires the open parameters to translated into a rational number using, e.g., the
 * method {@link FlTestUtils#gauss(java.math.BigInteger, java.math.BigInteger)}
 * </p>
 */
class Normalize implements Computation<List<DRes<SInt>>, ProtocolBuilderNumeric> {

  private final WeightedModelParams<SInt> weightedParams;

  /**
   * A new normalization computation.
   * @param weightedParams the parameters to normalize
   */
  Normalize(WeightedModelParams<SInt> weightedParams) {
    this.weightedParams = weightedParams;
  }

  @Override
  public DRes<List<DRes<SInt>>> buildComputation(ProtocolBuilderNumeric builder) {
    return builder.seq(seqBuilder -> {
      DRes<SInt> invWeight = seqBuilder.advancedNumeric().invert(weightedParams.getWeight());
      return seqBuilder.par(parBuilder -> {
        List<DRes<SInt>> result = weightedParams.getWeightedParams().stream()
            .map(w -> parBuilder.numeric().mult(invWeight, w)).collect(Collectors.toList());
        return () -> result;
      });
    });
  }
}