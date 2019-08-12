package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.ml.fl.demo.SpdzTestSetup;
import dk.alexandra.fresco.ml.fl.demo.TestSetup;
import dk.alexandra.fresco.suite.spdz.SpdzResourcePool;
import java.util.HashMap;
import java.util.Map;

/**
 * An implementation of {@link MpcMnistFlTest} using the SPDZ protocol suite to do the MPC
 * computations.
 */
public class SpdzMnistFlTest extends MpcMnistFlTest<SpdzResourcePool> {

  @Override
  public Map<Integer, TestSetup<SpdzResourcePool, ProtocolBuilderNumeric>> getSetups(int parties) {
    Map<Integer, SpdzTestSetup> test = SpdzTestSetup.builder(parties).build();
    return new HashMap<>(test);
  }

}
