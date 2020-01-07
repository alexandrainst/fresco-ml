package dk.alexandra.fresco.ml.svm;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.ComputationParallel;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.value.SInt;

public class InputSVMAsSender implements
    ComputationParallel<SVMModelClosed, ProtocolBuilderNumeric> {

  private final SVMModel model;
  private final int senderId;

  public InputSVMAsSender(SVMModel model, int senderId) {
    this.model = model;
    this.senderId = senderId;
  }

  /**
   * Inputs list of secrets and returns result as an undeferred list.
   */
  private List<DRes<SInt>> input(ProtocolBuilderNumeric builder, List<BigInteger> values) {
    List<DRes<SInt>> secrets = new ArrayList<>(values.size());
    for (BigInteger value : values) {
      secrets.add(builder.numeric().input(value, senderId));
    }
    return secrets;
  }

  @Override
  public DRes<SVMModelClosed> buildComputation(ProtocolBuilderNumeric builder) {
    List<List<BigInteger>> supportVectors = model.getSupportVectors();
    List<BigInteger> bias = model.getBias();
    List<List<DRes<SInt>>> supportVectorsClosed = new ArrayList<>(supportVectors.size());
    for (List<BigInteger> supportVector : supportVectors) {
      supportVectorsClosed.add(input(builder, supportVector));
    }
    List<DRes<SInt>> biasClosed = input(builder, bias);

    SVMModelClosed closedModel = new SVMModelClosed(supportVectorsClosed, biasClosed);
    return () -> closedModel;
  }
}
