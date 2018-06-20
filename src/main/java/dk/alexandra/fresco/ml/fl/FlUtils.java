package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.Application;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.util.Pair;
import dk.alexandra.fresco.framework.value.SInt;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Utility functions for FL.
 */
class FlUtils {


  private FlUtils() {
    // Do not instantiate
  }

  static WeightedModelParams<BigInteger> createPlainParams(final BigInteger weight,
      final List<BigInteger> bs) {
    return new WeightedModelParamsImpl<>(dressify(weight), dressify(bs));
  }

  static WeightedModelParams<BigInteger> createPlainParams(final BigInteger weight,
      final BigInteger... bs) {
    return createPlainParams(weight, Arrays.asList(bs));
  }

  static <T> List<DRes<T>> dressify(List<T> list) {
    return list.stream().map(FlUtils::dressify).collect(Collectors.toList());
  }

  static <T> DRes<T> dressify(T e) {
    return () -> e;
  }

  @SafeVarargs
  static Application<List<WeightedModelParams<SInt>>, ProtocolBuilderNumeric> closeModelParams(
      final WeightedModelParams<BigInteger>... params) {
    return closeModelParams(Arrays.asList(params));
  }

  static Application<List<WeightedModelParams<SInt>>, ProtocolBuilderNumeric> closeModelParams(
      final List<WeightedModelParams<BigInteger>> params) {
    return builder -> {
      List<Pair<DRes<SInt>, DRes<List<DRes<SInt>>>>> resultList = new ArrayList<>(params.size());
      for (WeightedModelParams<BigInteger> p : params) {
        DRes<List<DRes<SInt>>> paramsA = builder.collections().closeList(
            p.getWeightedParams().stream().map(DRes::out).collect(Collectors.toList()), 1);
        DRes<SInt> weightA = builder.numeric().input(p.getWeight().out(), 1);
        resultList.add(new Pair<>(weightA, paramsA));
      }
      return () -> resultList.stream()
          .map(r -> new WeightedModelParamsImpl<>(r.getFirst(), r.getSecond().out()))
          .collect(Collectors.toList());
    };
  }

  static Application<List<SInt>, ProtocolBuilderNumeric> closeNumbers(
      final BigInteger... numbers) {
    return builder -> {
      List<DRes<SInt>> result = new ArrayList<>(numbers.length);
      for (BigInteger b : numbers) {
        result.add(builder.numeric().input(b, 1));
      }
      return () -> result.stream().map(DRes::out).collect(Collectors.toList());
    };
  }

  /**
   * Converts a number of the form <i>t = r*s<sup>-1</sup> mod N</i> to the rational number
   * <i>r/s</i> represented as a reduced fraction.
   * <p>
   * This is useful outputting non-integer rational numbers from MPC, when outputting a non-reduced
   * fraction may leak too much information. The technique used is adapted from the paper
   * "CryptoComputing With Rationals" of Fouque et al. Financial Cryptography 2002. This methods
   * restricts us to integers <i>t = r*s<sup>-1</sup> mod N</i> so that <i>2r*s < N</i>. See
   * <a href="https://www.di.ens.fr/~stern/data/St100.pdf">https://www.di.ens.
   * fr/~stern/data/St100.pdf</a>
   * </p>
   *
   * @param product The integer <i>t = r*s<sup>-1</sup>mod N</i>. Note that we must have that
   *     <i>2r*s < N</i>.
   * @param mod the modulus, i.e., <i>N</i>.
   * @return The fraction as represented as the rational number <i>r/s</i>.
   */
  static BigInteger[] gauss(BigInteger product, BigInteger mod) {
    product = product.mod(mod);
    BigInteger[] u = {mod, BigInteger.ZERO};
    BigInteger[] v = {product, BigInteger.ONE};
    BigInteger two = BigInteger.valueOf(2);
    BigInteger uv = innerproduct(u, v);
    BigInteger vv = innerproduct(v, v);
    BigInteger uu = innerproduct(u, u);
    do {
      BigInteger[] q = uv.divideAndRemainder(vv);
      boolean negRes = q[1].signum() == -1;
      if (!negRes) {
        if (vv.compareTo(q[1].multiply(two)) <= 0) {
          q[0] = q[0].add(BigInteger.ONE);
        }
      } else {
        if (vv.compareTo(q[1].multiply(two.negate())) <= 0) {
          q[0] = q[0].subtract(BigInteger.ONE);
        }
      }
      BigInteger r0 = u[0].subtract(v[0].multiply(q[0]));
      BigInteger r1 = u[1].subtract(v[1].multiply(q[0]));
      u = v;
      v = new BigInteger[]{r0, r1};
      uu = vv;
      uv = innerproduct(u, v);
      vv = innerproduct(v, v);
    } while (uu.compareTo(vv) > 0);
    return new BigInteger[]{u[0], u[1]};
  }

  static BigInteger innerproduct(BigInteger[] u, BigInteger[] v) {
    return u[0].multiply(v[0]).add(u[1].multiply(v[1]));
  }

}
