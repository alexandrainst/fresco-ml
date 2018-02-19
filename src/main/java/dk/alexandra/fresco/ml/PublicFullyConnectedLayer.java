package dk.alexandra.fresco.ml;

import dk.alexandra.fresco.decimal.RealNumeric;
import dk.alexandra.fresco.decimal.RealNumericProvider;
import dk.alexandra.fresco.decimal.SReal;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.lib.collections.Matrix;
import java.math.BigDecimal;

/**
 * This class represents a fully connected layer in a feed-forward neural network.
 * 
 * @author Jonas Lindstrøm (jonas.lindstrom@alexandra.dk)
 *
 */
public class PublicFullyConnectedLayer implements Layer {

  private RealNumericProvider provider;
  private DRes<Matrix<DRes<SReal>>> v;
  private FullyConnectedLayerParameters<BigDecimal> parameters;

  public PublicFullyConnectedLayer(FullyConnectedLayerParameters<BigDecimal> parameters,
      DRes<Matrix<DRes<SReal>>> v, RealNumericProvider provider) {
    this.parameters = parameters;
    this.v = v;
    this.provider = provider;
  }

  @Override
  public DRes<Matrix<DRes<SReal>>> buildComputation(ProtocolBuilderNumeric builder) {
    return builder.seq(r1 -> {
      RealNumeric numeric = provider.apply(r1);
      return numeric.linalg().add(parameters.getBias(),
          numeric.linalg().mult(parameters.getWeights(), v));
    }).seq((r2, w) -> {
      ActivationFunctions activation = new DefaultActivationFunctions(r2, provider);
      return activation.activation(parameters.getActivation(), w);
    });
  }

}
