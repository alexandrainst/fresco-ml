package dk.alexandra.fresco.ml.fl;

import static org.junit.Assert.assertEquals;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.util.Pair;
import dk.alexandra.fresco.framework.value.SInt;
import java.io.IOException;
import java.math.BigInteger;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class SumParamsTest {

  SinglePartyTestSetup setup;

  @Before
  public void setUp() {
    setup = new SinglePartyTestSetup();
  }

  @After
  public void tearDown() throws IOException {
    setup.close();
  }

  @Test
  public void test() {
    // Setup
    WeightedModelParams<BigInteger> openParamA = FlTestUtils
        .createPlainParams(BigInteger.valueOf(10), BigInteger.valueOf(10), BigInteger.valueOf(30));
    WeightedModelParams<BigInteger> openParamB = FlTestUtils
        .createPlainParams(BigInteger.valueOf(5), BigInteger.valueOf(5), BigInteger.valueOf(15));
    List<WeightedModelParams<SInt>> params = setup.getSce().runApplication(
        FlTestUtils.closeModelParams(openParamA, openParamB), setup.getRp(), setup.getNet());

    // Compute
    WeightedModelParams<SInt> output = setup.getSce()
        .runApplication(new SumParams(params.get(0), params.get(1)), setup.getRp(), setup.getNet());

    // Test results
    Pair<DRes<BigInteger>, DRes<List<DRes<BigInteger>>>> result = setup.getSce()
        .runApplication(builder -> {
          DRes<BigInteger> w = builder.numeric().open(output.getWeight());
          DRes<List<DRes<BigInteger>>> p = builder.collections()
              .openList(() -> output.getWeightedParams());
          return () -> new Pair<>(w, p);
        }, setup.getRp(), setup.getNet());
    BigInteger expectedWeight = openParamA.getWeight().out().add(openParamB.getWeight().out());
    List<BigInteger> expectedParams = IntStream.range(0, openParamA.getWeightedParams().size())
        .mapToObj(i -> new Pair<>(openParamA.getWeightedParams().get(i),
            openParamB.getWeightedParams().get(i)))
        .map(p -> p.getFirst().out().add(p.getSecond().out())).collect(Collectors.toList());
    BigInteger actualWeight = result.getFirst().out();
    List<BigInteger> actualParams = result.getSecond().out().stream().map(DRes::out)
        .collect(Collectors.toList());
    assertEquals(expectedWeight, actualWeight);
    assertEquals(expectedParams, actualParams);
  }

}
