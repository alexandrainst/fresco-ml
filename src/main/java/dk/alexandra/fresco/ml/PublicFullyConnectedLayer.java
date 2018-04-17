package dk.alexandra.fresco.ml;

import dk.alexandra.fresco.lib.real.SReal;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.lib.collections.Matrix;
import java.math.BigDecimal;

/**
 * This class represents a fully connected layer in a feed-forward neural network.
 */
public class PublicFullyConnectedLayer implements Layer {

  private DRes<Matrix<DRes<SReal>>> v;
  private FullyConnectedLayerParameters<BigDecimal> parameters;

  public PublicFullyConnectedLayer(FullyConnectedLayerParameters<BigDecimal> parameters,
      DRes<Matrix<DRes<SReal>>> v) {
    this.parameters = parameters;
    this.v = v;
  }

  @Override
  public DRes<Matrix<DRes<SReal>>> buildComputation(ProtocolBuilderNumeric builder) {
    return builder.seq(r1 -> {
      return r1.realLinAlg().add(parameters.getBias(),
          r1.realLinAlg().mult(parameters.getWeights(), v));
    }).seq((r2, w) -> {
      ActivationFunctions activation = new DefaultActivationFunctions(r2);
      return activation.activation(parameters.getActivation(), w);
    });
  }

}
