package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.DRes;
import java.util.List;

public interface AveragingService<T> {

  /**
   * Add a list of weighted parameters to be averaged.
   * @param params weighted parameters
   */
  void addToAverage(WeightedModelParams<T> params);

  /**
   * Gives the weighted average of the parameters added to this AveragingService.
   * @return the weighted average
   */
  List<DRes<T>> getAveragedParams();

}
