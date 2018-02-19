package dk.alexandra.fresco.ml;

import dk.alexandra.fresco.decimal.RealNumeric;
import dk.alexandra.fresco.decimal.RealNumericProvider;
import dk.alexandra.fresco.decimal.SReal;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.util.Pair;
import dk.alexandra.fresco.framework.value.SInt;
import dk.alexandra.fresco.lib.collections.Matrix;
import dk.alexandra.fresco.ml.utils.LinearAlgebraUtils;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.Collectors;

public class DefaultActivationFunctions implements ActivationFunctions {

  private ProtocolBuilderNumeric builder;
  private RealNumericProvider provider;

  public DefaultActivationFunctions(ProtocolBuilderNumeric builder, RealNumericProvider provider) {
    this.builder = builder;
    this.provider = provider;
  }

  @Override
  public DRes<Matrix<DRes<SReal>>> activation(Type type, Matrix<DRes<SReal>> v) {

    switch (type) {
      case RELU:
        return relu(v);

      case SIGMOID:
        return sigmoid(v);

      case IDENTITY:
        return identity(v);

      case SOFTMAX:
        return softmax(v);
    }

    throw new IllegalArgumentException("Unsupported activation function type, " + type);
  }

  private BiFunction<ProtocolBuilderNumeric, DRes<SReal>, DRes<SReal>> ebeRelu() {
    return (builder, x) -> {
      return builder.seq(r1 -> {
        RealNumeric numeric = provider.apply(r1);
        DRes<SInt> compare = numeric.advanced().leq(numeric.numeric().known(BigDecimal.ZERO), x);
        return numeric.numeric().mult(x, numeric.numeric().fromSInt(compare));
      });
    };
  }

  @Override
  public DRes<Matrix<DRes<SReal>>> relu(Matrix<DRes<SReal>> v) {
    return ebe(v, ebeRelu());
  }

  @Override
  public DRes<Matrix<DRes<SReal>>> softmax(Matrix<DRes<SReal>> v) {
    return builder.par(r1 -> {
      RealNumeric numeric = provider.apply(r1);
      List<DRes<SReal>> exps = v.getColumn(0).stream().map(e -> numeric.advanced().exp(e))
          .collect(Collectors.toList());
      return () -> exps;
    }).seq((r2, l) -> {
      RealNumeric numeric = provider.apply(r2);
      DRes<SReal> sum = numeric.advanced().sum(l);
      return () -> new Pair<>(l, sum);
    }).par((r3, p) -> {
      RealNumeric numeric = provider.apply(r3);
      Matrix<DRes<SReal>> vector = new LinearAlgebraUtils().createColumnVector(p.getFirst().stream()
          .map(e -> numeric.numeric().div(e, p.getSecond())).collect(Collectors.toList()));
      return () -> vector;
    });
  }

  @Override
  public DRes<Matrix<DRes<SReal>>> sigmoid(Matrix<DRes<SReal>> v) {
    return ebe(v, ebeSigmoid());
  }

  @Override
  public DRes<Matrix<DRes<SReal>>> identity(Matrix<DRes<SReal>> v) {
    return builder.seq(seq -> () -> v);
  }

  private BiFunction<ProtocolBuilderNumeric, DRes<SReal>, DRes<SReal>> ebeSigmoid() {
    return (builder, x) -> {
      return builder.seq(r1 -> {
        RealNumeric numeric = provider.apply(r1);
        DRes<SReal> exp = numeric.advanced().exp(x);
        return numeric.numeric().div(exp, numeric.numeric().add(BigDecimal.ONE, exp));
      });
    };
  }
  
  /**
   * Apply the given map to all elements in the given matrix.
   * 
   * @param m 
   * @param map
   * @return
   */
  private DRes<Matrix<DRes<SReal>>> ebe(Matrix<DRes<SReal>> m,
      BiFunction<ProtocolBuilderNumeric, DRes<SReal>, DRes<SReal>> map) {
    return builder.par(par -> {
      ArrayList<ArrayList<DRes<SReal>>> rows = new ArrayList<>(m.getHeight());
      for (int i = 0; i < m.getHeight(); i++) {
        ArrayList<DRes<SReal>> row = new ArrayList<>();
        for (int j = 0; j < m.getWidth(); j++) {
          row.add(map.apply(par, m.getRow(i).get(j)));
        }
        rows.add(row);
      }
      return () -> new Matrix<>(m.getHeight(), m.getWidth(), rows);
    });
  }
}
