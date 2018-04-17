package dk.alexandra.fresco.ml;

import dk.alexandra.fresco.lib.real.SReal;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.lib.collections.Matrix;
import java.math.BigDecimal;
import java.util.List;

public class NeuralNetwork implements Computation<Matrix<DRes<SReal>>, ProtocolBuilderNumeric> {

  private List<FullyConnectedLayerParameters<BigDecimal>> layerParameters;
  private DRes<Matrix<DRes<SReal>>> input;

  public NeuralNetwork(List<FullyConnectedLayerParameters<BigDecimal>> layers,
      DRes<Matrix<DRes<SReal>>> input) {
    this.layerParameters = layers;
    this.input = input;
  }

  @Override
  public DRes<Matrix<DRes<SReal>>> buildComputation(ProtocolBuilderNumeric builder) {
    return builder.seq(seq -> {
      DRes<Matrix<DRes<SReal>>> x = input;
      for (FullyConnectedLayerParameters<BigDecimal> parameters : layerParameters) {
        // TODO: For now, only public fully connected layers are supported
        x = seq.seq(new PublicFullyConnectedLayer(parameters, x));
      }
      return x;
    });
  }

}
