package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.ProtocolBuilder;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import java.util.List;

/**
 * Interface for computing a running average of {@link WeightedModelParams}.
 *
 * @param <T> The value type average over (e.g., SInt)
 * @param <BuilderT> The {@link ProtocolBuilder} type used to build the computaitons
 */
public interface Averager<T, BuilderT extends ProtocolBuilderNumeric> {

  /**
   * Add a set of weighted parameters to be averaged.
   * @param params weighted parameters
   */
  Computation<Void, BuilderT> addToAverage(WeightedModelParams<T> params);

  /**
   * Gives the weighted average of the parameters added to this AveragingService.
   * @return the weighted average
   */
  Computation<List<DRes<T>>, BuilderT> getAveragedParams();

}
