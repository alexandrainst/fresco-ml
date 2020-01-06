package dk.alexandra.fresco.ml.lr;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.lib.real.SReal;

public class LogisticRegressionPrediction implements Computation<SReal, ProtocolBuilderNumeric> {

    private List<DRes<SReal>> row;
    private List<DRes<SReal>> b;

    public LogisticRegressionPrediction(List<DRes<SReal>> row, List<DRes<SReal>> b) {
        assert (row.size() == b.size() - 1);

        this.row = row;
        this.b = b;
    }

    @Override
    public DRes<SReal> buildComputation(ProtocolBuilderNumeric builder) {
        return builder.par(par -> {
            List<DRes<SReal>> terms = new ArrayList<>();
            terms.add(b.get(0));
            for (int i = 0; i < row.size(); i++) {
                terms.add(par.realNumeric().mult(b.get(i + 1), row.get(i)));
            }
            return () -> terms;
        }).seq((seq, terms) -> {
            DRes<SReal> sum = seq.realAdvanced().sum(terms);
            // TODO: use reciprocal when it's included
            DRes<SReal> yHat = seq.realNumeric().div(seq.realNumeric().known(BigDecimal.ONE),
                    seq.realNumeric().add(BigDecimal.ONE, seq.realAdvanced().exp(seq.realNumeric().sub(BigDecimal.ZERO, sum))));
            return yHat;
        });
    }

}
