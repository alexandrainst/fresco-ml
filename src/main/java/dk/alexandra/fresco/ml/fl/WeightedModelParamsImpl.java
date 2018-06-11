package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.DRes;
import java.util.List;

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

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("(weight=").append(weight.out()).append(", params=");
    for (DRes<T> p : weightedParams) {
      sb.append(p.out()).append(",");
    }
    sb.deleteCharAt(sb.lastIndexOf(","));
    sb.append(")");
    return sb.toString();
  }

}
