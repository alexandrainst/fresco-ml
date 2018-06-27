package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.suite.dummy.arithmetic.DummyArithmeticResourcePool;
import java.util.HashMap;
import java.util.Map;


/**
 * An implementation of {@link MpcMnistFlTest} using the SPDZ protocol suite to do the MPC
 * computations.
 */
public class DummyAritMnistFlTest extends MpcMnistFlTest<DummyArithmeticResourcePool> {

  @Override
  public Map<Integer, TestSetup<DummyArithmeticResourcePool, ProtocolBuilderNumeric>> getSetups(
      int numParties) {
    return new HashMap<>(DummyArithTestSetup.builder(numParties).build());

  }

}
