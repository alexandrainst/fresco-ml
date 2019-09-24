package dk.alexandra.fresco.ml.lr;

import java.util.ArrayList;
import java.util.List;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.lib.collections.Matrix;
import dk.alexandra.fresco.lib.real.SReal;

public class LogisticRegressionSGD
    implements Computation<List<DRes<SReal>>, ProtocolBuilderNumeric> {

  private Matrix<DRes<SReal>> data;
  private List<DRes<SReal>> expected;
  private double rate;
  private List<DRes<SReal>> b;

  public LogisticRegressionSGD(Matrix<DRes<SReal>> data, List<DRes<SReal>> expected, double rate,
      List<DRes<SReal>> b) {

    assert (data.getWidth() == b.size() - 1);
    assert (data.getHeight() == expected.size());

    this.data = data;
    this.expected = expected;
    this.rate = rate;
    this.b = b;
  }

  @Override
  public DRes<List<DRes<SReal>>> buildComputation(ProtocolBuilderNumeric builder) {
    return builder.seq(seq -> {
      return new IterationState(0, () -> b);
    }).whileLoop(state -> state.i < data.getHeight(), (seq, state) -> {
      int i = state.i;
      DRes<List<DRes<SReal>>> newB =
          new RowGradient(data.getRow(i), expected.get(i), state.b.out(), rate)
              .buildComputation(seq);
      return new IterationState(i + 1, newB);
    }).seq((seq, state) -> {
      return state.b;
    });
  }

  private static class RowGradient
      implements Computation<List<DRes<SReal>>, ProtocolBuilderNumeric> {
    private List<DRes<SReal>> row;
    private DRes<SReal> expected;
    private List<DRes<SReal>> b;
    private double rate;

    private RowGradient(List<DRes<SReal>> row, DRes<SReal> expected, List<DRes<SReal>> b,
        double rate) {
      this.row = row;
      this.expected = expected;
      this.b = b;
      this.rate = rate;
    }

    @Override
    public DRes<List<DRes<SReal>>> buildComputation(ProtocolBuilderNumeric builder) {
      return builder.seq(seq -> {
        DRes<SReal> yHat = new LogisticRegressionPrediction(row, b).buildComputation(seq);
        DRes<SReal> error = seq.realNumeric().sub(expected, yHat);
        DRes<SReal> t =
            seq.realNumeric().mult(seq.realNumeric().mult(yHat, seq.realNumeric().sub(1.0, yHat)),
                seq.realNumeric().mult(rate, error));
        return t;
      }).par((par, t) -> {
        List<DRes<SReal>> delta = new ArrayList<>(b.size());
        delta.add(t);
        for (DRes<SReal> ri : row) {
          delta.add(par.realNumeric().mult(t, ri));
        }
        return () -> delta;
      }).par((par, delta) -> {
        List<DRes<SReal>> newB = new ArrayList<>(b.size());
        for (int i = 0; i < delta.size(); i++) {
          newB.add(par.realNumeric().add(delta.get(i), b.get(i)));
        }
        return () -> newB;
      });
    }

  }

  private static class IterationState implements DRes<IterationState> {

    private int i;
    private final DRes<List<DRes<SReal>>> b;

    private IterationState(int round, DRes<List<DRes<SReal>>> value) {
      this.i = round;
      this.b = value;
    }

    @Override
    public IterationState out() {
      return this;
    }

  }

}
