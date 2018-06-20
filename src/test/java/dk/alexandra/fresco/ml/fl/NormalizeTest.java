package dk.alexandra.fresco.ml.fl;

import static org.junit.Assert.assertEquals;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.value.SInt;
import java.io.IOException;
import java.math.BigInteger;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class NormalizeTest {

  SinglePartyTestSetup setup;

  @Before
  public void setUp() {
    this.setup = new SinglePartyTestSetup();
  }

  @After
  public void tearDown() throws IOException {
    this.setup.close();
  }

  @Test
  public void test() {
    // Setup
    WeightedModelParams<BigInteger> openParams = FlTestUtils.createPlainParams(
        BigInteger.valueOf(12), BigInteger.valueOf(1), BigInteger.valueOf(24),
        BigInteger.valueOf(2000), BigInteger.valueOf(-12));
    // Compute
    List<WeightedModelParams<SInt>> closedParams = setup.getSce()
        .runApplication(FlTestUtils.closeModelParams(openParams), setup.getRp(), setup.getNet());
    List<DRes<SInt>> closedResult = setup.getSce()
        .runApplication(builder -> builder.seq(new Normalize(closedParams.get(0))),
            setup.getRp(), setup.getNet());
    // Test results
    List<DRes<BigInteger>> list = setup.getSce().runApplication(builder -> {
      List<DRes<BigInteger>> res = closedResult.stream().map(r -> builder.numeric().open(r))
          .collect(Collectors.toList());
      return () -> res;
    }, setup.getRp(), setup.getNet());
    for (int i = 0; i < list.size(); i++) {
      BigInteger[] fraction = FlTestUtils.gauss(list.get(i).out(), setup.getRp().getModulus());
      double expected = openParams.getWeightedParams().get(i).out().doubleValue()
          / openParams.getWeight().out().doubleValue();
      assertEquals(expected, fraction[0].doubleValue() / fraction[1].doubleValue(), 0.000001);
    }

  }

}
