package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.Application;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.value.SInt;
import java.util.List;
import java.util.stream.Collectors;

class Normalize implements Application<List<DRes<SInt>>, ProtocolBuilderNumeric> {

  private final WeightedModelParams<SInt> weightedParams;

  Normalize(WeightedModelParams<SInt> weightedParams) {
    this.weightedParams = weightedParams;
  }

  @Override
  public DRes<List<DRes<SInt>>> buildComputation(ProtocolBuilderNumeric builder) {
    return builder.seq(seqBuilder -> {
      DRes<SInt> invWeight = seqBuilder.advancedNumeric().invert(weightedParams.getWeight());
      return seqBuilder.par(parBuilder -> {
        List<DRes<SInt>> result = weightedParams.getWeightedParams().stream()
        .map(w -> parBuilder.numeric().mult(invWeight, w))
        .collect(Collectors.toList());
        return () -> result;
      });
    });

  }
}