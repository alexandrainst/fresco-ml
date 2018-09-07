package dk.alexandra.fresco.ml.svm;

import java.util.ArrayList;
import java.util.List;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.ComputationParallel;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.value.SInt;

public class InputSVMAsReceiver implements
    ComputationParallel<SVMModelClosed, ProtocolBuilderNumeric> {

  private final int features;
  private final int categories;
  private final int receiverId;

  public InputSVMAsReceiver(int features, int categories, int receiverId) {
    this.features = features;
    this.categories = categories;
    this.receiverId = receiverId;
  }

  /**
   * Inputs list of secrets and returns result as an undeferred list.
   */
  private List<DRes<SInt>> input(ProtocolBuilderNumeric builder, int numberOfValues) {
    List<DRes<SInt>> secrets = new ArrayList<>(numberOfValues);
    for (int i = 0; i < numberOfValues; i++) {
      secrets.add(builder.numeric().input(null, receiverId));
    }
    return secrets;
  }

  @Override
  public DRes<SVMModelClosed> buildComputation(ProtocolBuilderNumeric builder) {
    List<List<DRes<SInt>>> supportVectorsClosed = new ArrayList<>(categories);
    for (int i = 0; i < categories; i++) {
      supportVectorsClosed.add(input(builder, features));
    }
    List<DRes<SInt>> biasClosed = new ArrayList<>(categories);

    SVMModelClosed closedModel = new SVMModelClosed(supportVectorsClosed, biasClosed);
    return () -> closedModel;
  }

}
