package dk.alexandra.fresco.ml.svm;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.value.OInt;
import dk.alexandra.fresco.framework.value.OIntArithmetic;
import dk.alexandra.fresco.framework.value.SInt;
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
    }).par((par, products) -> par.comparison().argMin(products))
        .par((par, argMin) -> {
          List<DRes<SInt>> indexIndicators = argMin.getFirst();
          List<DRes<OInt>> opened = new ArrayList<>(indexIndicators.size());
          for (DRes<SInt> indexIndicator : indexIndicators) {
            opened.add(par.numeric().openAsOInt(indexIndicator));
          }
          return () -> opened;
        }).par((par, indexIndicators) -> {
          OIntArithmetic ar = par.getOIntArithmetic();
          for (int i = 0; i < indexIndicators.size(); i++) {
            if (ar.isOne(indexIndicators.get(i).out())) {
              BigInteger index = BigInteger.valueOf(i);
              return () -> index;
            }
          }
          throw new IllegalStateException("Index not found!");
        });
  }
}
