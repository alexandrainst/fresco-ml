package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.value.SInt;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * An MPC computation to sum two sets of weighted model parameters.
 */
class SumParams implements Computation<WeightedModelParams<SInt>, ProtocolBuilderNumeric> {

  private final WeightedModelParams<SInt> right;
  private final WeightedModelParams<SInt> left;

  SumParams(WeightedModelParams<SInt> a, WeightedModelParams<SInt> b) {
    this.left = a;
    this.right = b;
  }

  @Override
  public DRes<WeightedModelParams<SInt>> buildComputation(ProtocolBuilderNumeric builder) {
    DRes<WeightedModelParams<SInt>> sumParams = builder.par(parBuilder -> {
      DRes<SInt> weight = parBuilder.numeric().add(left.getWeight(), right.getWeight());
      List<DRes<SInt>> leftParams = left.getWeightedParams();
      List<DRes<SInt>> rightParams = right.getWeightedParams();
      List<DRes<SInt>> params = IntStream.range(0, leftParams.size())
          .mapToObj(i -> parBuilder.numeric().add(leftParams.get(i), rightParams.get(i)))
          .collect(Collectors.toList());
      return () -> new WeightedModelParamsImpl<>(weight, params);
    });
    return sumParams;
  }

}