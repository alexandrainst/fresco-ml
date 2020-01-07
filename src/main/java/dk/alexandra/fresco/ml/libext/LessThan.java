package dk.alexandra.fresco.ml.libext;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.value.SInt;

/**
 * Return 1 if a < b and 0 otherwise.
 */
public class LessThan implements Computation<SInt, ProtocolBuilderNumeric> {

    private final DRes<SInt> a;
    private final DRes<SInt> b;

    public LessThan(DRes<SInt> a, DRes<SInt> b) {
        this.a = a;
        this.b = b;
    }

    @Override
    public DRes<SInt> buildComputation(ProtocolBuilderNumeric builder) {
        return builder.seq(seq -> {
            return seq.numeric().sub(1, seq.comparison().compareLEQ(b, a));
        });
   }
}
