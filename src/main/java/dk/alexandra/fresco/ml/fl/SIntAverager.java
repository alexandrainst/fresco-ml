package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.sce.resources.ResourcePool;
import dk.alexandra.fresco.framework.value.SInt;
import java.util.List;

/**
 * An Averager of SInt based WeightedModelParams.
 *
 * @param <ResourcePoolT> the resource pool to use in the computations.
 */
public class SIntAverager<ResourcePoolT extends ResourcePool>
    implements Averager<SInt, ProtocolBuilderNumeric> {

  private WeightedModelParams<SInt> accumulator;

  /**
   * A new SIntAverager
   */
  SIntAverager() {
    this.accumulator = null;
  }

  @Override
  public Computation<Void, ProtocolBuilderNumeric> addToAverage(WeightedModelParams<SInt> params) {
    if (accumulator == null) {
      accumulator = params;
      return builder -> null;
    } else {
      return builder -> {
        return builder.seq(seq -> {
          return seq.seq(new SumParams(accumulator, params));
        }).seq((seq2, sum) -> {
          accumulator = sum;
          return null;
        });
      };
    }
  }

  @Override
  public Computation<List<DRes<SInt>>, ProtocolBuilderNumeric> getAveragedParams() {
    return builder -> builder.seq(new Normalize(accumulator));
  }

}
