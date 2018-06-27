package dk.alexandra.fresco.ml.fl;

import java.util.Map;
import org.junit.Test;

public class DummyArithTestSetupTest {

  private Map<Integer, DummyArithTestSetup> setups;

  @Test
  public void test() throws Exception {
    setups = DummyArithTestSetup.builder(5).build();
    for (TestSetup<?, ?> setup : setups.values()) {
      setup.close();
    }
  }

}
