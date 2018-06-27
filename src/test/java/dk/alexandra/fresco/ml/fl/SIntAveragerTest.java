package dk.alexandra.fresco.ml.fl;

import static org.junit.Assert.assertEquals;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.util.Pair;
import dk.alexandra.fresco.framework.value.SInt;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Functional tests of the SIntAverager.
 */
public class SIntAveragerTest {

  SinglePartyTestSetup setup;

  @Before
  public void setUp() throws Exception {
    this.setup = new SinglePartyTestSetup();
  }

  @After
  public void tearDown() throws Exception {
    this.setup.close();
  }

  @Test
  public void test() {
    // Setup test
    final int numModels = 8;
    final int numParams = 5;
    final int weightBitLength = 12;
    final int paramBitLength = 12;
    List<WeightedModelParams<BigInteger>> weightedParams = new ArrayList<>(numModels);
    Random rand = new Random(0);
    for (int i = 0; i < numModels; i++) {
      BigInteger weight = new BigInteger(weightBitLength, rand);
      BigInteger[] params = new BigInteger[numParams];
      for (int j = 0; j < numParams; j++) {
        params[j] = new BigInteger(paramBitLength, rand);
      }
      weightedParams.add(FlUtils.createPlainParams(weight, params));
    }
    // Do test
    Averager<SInt, ProtocolBuilderNumeric> service = new SIntAverager<>();
    List<WeightedModelParams<SInt>> closedParams = setup.getSce().runApplication(
        FlUtils.closeModelParams(weightedParams), setup.getRp(), setup.getNet());
    List<DRes<SInt>> closedAverage = setup.getSce().runApplication(builder -> {
      for (WeightedModelParams<SInt> w : closedParams) {
        builder.seq(service.addToAverage(w));
      }
      return builder.seq(service.getAveragedParams());
    }, setup.getRp(), setup.getNet());
    // Test results
    List<Double> actualAverage = setup.getSce().runApplication(builder -> {
      List<DRes<BigInteger>> result = closedAverage.stream()
          .map(builder.numeric()::open)
          .collect(Collectors.toList());
      return () -> result.stream()
          .map(DRes::out)
          .map(p -> FlUtils.gauss(p, setup.getRp().getModulus()))
          .map(f -> f[0].doubleValue() / f[1].doubleValue())
          .collect(Collectors.toList());
    }, setup.getRp(), setup.getNet());
    List<Double> expectedAverage = computePlainAverage(weightedParams);
    assertEquals(expectedAverage, actualAverage);
  }

  List<Double> computePlainAverage(List<WeightedModelParams<BigInteger>> params) {
    WeightedModelParams<BigInteger> summedParams = params.stream().reduce(this::sum).get();
    double summedWeight = summedParams.getWeight().out().doubleValue();
    return summedParams.getWeightedParams().stream().map(p -> p.out().doubleValue())
        .map(d -> d / summedWeight).collect(Collectors.toList());
  }

  WeightedModelParams<BigInteger> sum(WeightedModelParams<BigInteger> a,
      WeightedModelParams<BigInteger> b) {
    List<BigInteger> params = IntStream.range(0, a.getWeightedParams().size())
        .mapToObj(
            i -> new Pair<>(a.getWeightedParams().get(i).out(), b.getWeightedParams().get(i).out()))
        .map(p -> p.getFirst().add(p.getSecond())).collect(Collectors.toList());
    BigInteger weight = a.getWeight().out().add(b.getWeight().out());
    return FlUtils.createPlainParams(weight, params);
  }
}
