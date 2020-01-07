package dk.alexandra.fresco.ml.svm;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.value.SInt;
import dk.alexandra.fresco.ml.libext.ArgMin;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

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
      final BigInteger zero = BigInteger.ZERO;
      // Compute the inner product of the each of the support vectors with the feature vector
      for (int i = 0; i < supportVectors.size(); i++) {
        int finalI = i;
        final DRes<SInt> prod = par.seq(seq -> {
          DRes<SInt> temp = seq.advancedNumeric().innerProduct(
              supportVectors.get(finalI), featureVector);
          final DRes<SInt> sum = seq.numeric().add(temp, model.getBias().get(finalI));
          // Negate to have argmin work as argmax.
          return seq.numeric().sub(zero, sum);
        });
        products.add(prod);
      }
      return () -> products;
    }).par((par, products) -> new ArgMin(products).buildComputation(par))
        .par((par, argMin) -> {
          List<DRes<SInt>> indexIndicators = argMin.getFirst();
          List<DRes<BigInteger>> opened = new ArrayList<>(indexIndicators.size());
          for (DRes<SInt> indexIndicator : indexIndicators) {
            opened.add(par.numeric().open(indexIndicator));
          }
          return () -> opened;
        }).par((par, indexIndicators) -> {
          for (int i = 0; i < indexIndicators.size(); i++) {
            if (indexIndicators.get(i).out().equals(BigInteger.ONE)) {
              BigInteger index = BigInteger.valueOf(i);
              return () -> index;
            }
          }
          throw new IllegalStateException("Index not found!");
        });
  }
}
