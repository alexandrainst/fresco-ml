package dk.alexandra.fresco.ml.svm;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.util.Pair;
import dk.alexandra.fresco.framework.value.SInt;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class EvaluateSVM implements Computation<BigInteger, ProtocolBuilderNumeric> {

  private final SVMModelClosed model;
  private final List<DRes<SInt>> featureVector;

  public EvaluateSVM(SVMModelClosed model, List<DRes<SInt>> featureVector) {
    this.model = model;
    this.featureVector = featureVector;
  }

  @Override
  public DRes<BigInteger> buildComputation(ProtocolBuilderNumeric builder) {
    return builder.par(par -> {
      List<List<DRes<SInt>>> supportVectors = model.getSupportVectors();
      List<DRes<SInt>> products = new ArrayList<>(supportVectors.size());
      // Compute the inner product of the each of the support vectors with the feature vector
      for (int i = 0; i < supportVectors.size(); i++) {
        int finalI = i;
        final DRes<SInt> prod = par.seq(seq -> {
          DRes<SInt> temp = seq.advancedNumeric()
              .innerProduct(supportVectors.get(finalI), featureVector);
          return seq.numeric().add(temp, model.getBias().get(finalI));
        });
        products.add(prod);
      }
      return () -> products;
    }).par((par, products) -> {
      /*
       * Compute half of the bitlength (i.e. max positive value) and subtract the actual integer
       * this ensure that the negative numbers get mapped to the lower integer, the positive to the
       * upper and that min actually computed max
       */
      BigInteger halfSize = BigInteger.ONE.pow(par.getBasicNumericContext().getMaxBitLength() - 1);
      for (int i = 0; i < products.size(); i++) {
        products.set(i, par.numeric().sub(halfSize, products.get(i)));
      }
      return () -> products;
    }).par((par, products) -> {
      DRes<Pair<List<DRes<SInt>>, SInt>> min = par.comparison().argMin(products);
      // The list is a bitvector indicating whether the value of the particular index is the minimum
      // number in the entire list
      return () -> min.out().getFirst();
    }).par((par, indexIndicators) -> {
      List<DRes<SInt>> resultList = new ArrayList<>(indexIndicators.size());
      // Multiply each indicator variable with its index to get one non-zero value in the list,
      // which is the result of the entire computation
      for (int i = 0; i < indexIndicators.size(); i++) {
        DRes<SInt> adjustedIndicator = par.numeric().mult(BigInteger.valueOf(i), indexIndicators
            .get(i));
        resultList.add(adjustedIndicator);
      }
      return () -> resultList;
    }).par((par, indexIndicators) -> {
      List<DRes<BigInteger>> res = indexIndicators.stream().map(val -> par.numeric().open(val))
          .collect(Collectors
              .toList());
      return () -> res;
    }).seq((seq, list) -> {
      // Return the single value in the list which is not zero
      return list.stream().filter(val -> !val.out().equals(BigInteger.ZERO)).findFirst().orElse(
          null);
    });
  }
}
