package dk.alexandra.fresco.ml.nn;

import dk.alexandra.fresco.lib.real.SReal;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.lib.collections.Matrix;

public interface ActivationFunction extends Computation<DRes<Matrix<DRes<SReal>>>, ProtocolBuilderNumeric> {

}
