package dk.alexandra.fresco.ml.debug;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.Numeric;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.lib.collections.Matrix;
import dk.alexandra.fresco.lib.real.RealNumeric;
import dk.alexandra.fresco.lib.real.SReal;

import java.io.PrintStream;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * <b>NB: Use with caution as using this class will open values to all MPC parties.</b>
 * 
 * This class opens a number for debugging purposes and prints a message along with the revealed
 * value(s).
 * 
 */
public class RealArithmeticOpenAndPrint implements Computation<Void, ProtocolBuilderNumeric> {

  private DRes<SReal> number = null;
  private List<DRes<SReal>> vector = null;
  private Matrix<DRes<SReal>> matrix = null;
  private String label;
  private PrintStream stream;

  public RealArithmeticOpenAndPrint(String label, DRes<SReal> number, PrintStream stream) {
    this.label = label;
    this.number = number;
    this.stream = stream;
  }

  public RealArithmeticOpenAndPrint(String label, List<DRes<SReal>> vector, PrintStream stream) {
    this.label = label;
    this.vector = vector;
    this.stream = stream;
  }

  public RealArithmeticOpenAndPrint(String label, Matrix<DRes<SReal>> matrix, PrintStream stream) {
    this.label = label;
    this.matrix = matrix;
    this.stream = stream;
  }

  @Override
  public DRes<Void> buildComputation(ProtocolBuilderNumeric builder) {
    return builder.seq(seq -> {
      RealNumeric num = seq.realNumeric();
      List<DRes<BigDecimal>> res = new ArrayList<>();
      if (number != null) {
        res.add(num.open(number));
      } else if (vector != null) {
        for (DRes<SReal> c : vector) {
          res.add(num.open(c));
        }
      } else {
        // matrix
        for (int i = 0; i < matrix.getHeight(); i++) {
          List<DRes<SReal>> l = matrix.getRow(i);
          for (DRes<SReal> c : l) {
            res.add(num.open(c));
          }
        }
      }
      return () -> res;
    }).seq((seq, res) -> {
      StringBuilder sb = new StringBuilder();
      sb.append(label);
      sb.append("\n");
      if (number != null) {
        sb.append(res.get(0).out());
      } else if (vector != null) {
        for (DRes<BigDecimal> v : res) {
          sb.append(v.out() + ", ");
        }
      } else {
        Iterator<DRes<BigDecimal>> it = res.iterator();
        for (int i = 0; i < this.matrix.getHeight(); i++) {
          for (int j = 0; j < this.matrix.getWidth(); j++) {
            sb.append(it.next().out() + ", ");
          }
          sb.append("\n");
        }
      }
      seq.debug().marker(sb.toString(), stream);
      return null;
    });
  }
}
