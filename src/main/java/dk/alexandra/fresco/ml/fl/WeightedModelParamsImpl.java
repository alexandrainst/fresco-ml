package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.DRes;
import java.util.List;

/**
 * A generic implementation of the WeightedModelParams interface.
 *
 * @param <T> the value type for the parameters
 */
public class WeightedModelParamsImpl<T> implements WeightedModelParams<T> {

  private final DRes<T> weight;
  private final List<DRes<T>> weightedParams;

  public WeightedModelParamsImpl(DRes<T> weight, List<DRes<T>> weightedParams) {
    this.weight = weight;
    this.weightedParams = weightedParams;
  }

  @Override
  public DRes<T> getWeight() {
    return weight;
  }

  @Override
  public List<DRes<T>> getWeightedParams() {
    return weightedParams;
  }

}
