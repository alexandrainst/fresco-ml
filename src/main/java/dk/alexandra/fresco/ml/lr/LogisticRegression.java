package dk.alexandra.fresco.ml.lr;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.lib.collections.Matrix;
import dk.alexandra.fresco.lib.real.SReal;

/**
 * A naive implementation of logistic regression, not optimized for secure computation. Given a data
 * set consisting of column vectors and a list of expected classifications (0 or 1), this
 * computation performs a linear regression on the data and the expected outcome as log-odds. See
 * also <a href=
 * "https://en.wikipedia.org/wiki/Logistic_regression">https://en.wikipedia.org/wiki/Logistic_regression</a>.
 * 
 */
public class LogisticRegression implements Computation<List<DRes<SReal>>, ProtocolBuilderNumeric> {

  private Matrix<DRes<SReal>> data;
  private List<DRes<SReal>> expected;
  private double rate;
  private int epochs;

  public LogisticRegression(Matrix<DRes<SReal>> data, List<DRes<SReal>> expected, double rate,
      int epochs) {
    this.data = data;
    this.expected = expected;
    this.rate = rate;
    this.epochs = epochs;
  }

  @Override
  public DRes<List<DRes<SReal>>> buildComputation(ProtocolBuilderNumeric builder) {

    return builder.seq(seq -> {
      int round = 0;
      List<DRes<SReal>> b =
          new ArrayList<>(Collections.nCopies(data.getWidth() + 1, seq.realNumeric().known(0.0)));
      return new IterationState(round, () -> b);
    }).whileLoop((state) -> state.round < epochs, (seq, state) -> {
      seq.debug().marker("Round " + state.round, System.out);
      DRes<List<DRes<SReal>>> newB =
          new LogisticRegressionSGD(data, expected, rate, state.b.out()).buildComputation(seq);
      return new IterationState(state.round + 1, newB);
    }).seq((seq, state) -> {
      return state.b;
    });
  }

  private static final class IterationState implements DRes<IterationState> {

    private int round;
    private final DRes<List<DRes<SReal>>> b;

    private IterationState(int round, DRes<List<DRes<SReal>>> value) {
      this.round = round;
      this.b = value;
    }

    @Override
    public IterationState out() {
      return this;
    }

  }

}
