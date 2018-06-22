package dk.alexandra.fresco.ml.dtrees;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class TestUtils {

  private TestUtils() {
  }

  public static List<BigInteger> toBitIntegers(int[] indexes) {
    return Arrays.stream(indexes).mapToObj(BigInteger::valueOf)
        .collect(Collectors.toCollection(() -> new ArrayList<>(indexes.length)));
  }

}
