package dk.alexandra.fresco.ml;

import dk.alexandra.fresco.lib.real.SReal;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.lib.collections.Matrix;

public interface Layer extends Computation<Matrix<DRes<SReal>>, ProtocolBuilderNumeric> {

}
