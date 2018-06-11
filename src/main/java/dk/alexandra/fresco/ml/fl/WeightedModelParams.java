package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.DRes;
import java.util.List;

/**
 * Represents a list of weighted NN model parameters.
 *
 * <p>
 * The weighted parameters has a weight <i>w</i> and a list of weighted parameters <i>p<sub>1</sub>,
 * ... , p<sub>n</sub></i>. The actual model parameters should then be taken as <i>p<sub>1</sub>/w,
 * ... , p<sub>n</sub>/w</i>.
 * </p>
 */
public interface WeightedModelParams<T> {

  /**
   * Returns the <i>weight</i> of the model, i.e., the number of samples the model was trained on.
   *
   * @return the weight of the model
   */
  DRes<T> getWeight();

  /**
   * Returns the weighted parameters of the model. To be used as model parameters these should be
   * normalized by dividing by the weight of the model.
   *
   * @return the weighted parameters
   */
  List<DRes<T>> getWeightedParams();

}
