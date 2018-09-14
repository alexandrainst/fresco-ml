package dk.alexandra.fresco.ml.svm;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

import dk.alexandra.fresco.framework.util.Pair;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.value.SInt;
import dk.alexandra.fresco.lib.math.integer.min.Minimum;

public class EvaluateSVM implements Computation<SInt, ProtocolBuilderNumeric> {
  private final SVMModelClosed model;
  private final List<DRes<SInt>> featureVector;
  private final int sampleParty;

  public EvaluateSVM(SVMModelClosed model, List<DRes<SInt>> featureVector,
      int sampleParty) {
    this.model = model;
    this.featureVector = featureVector;
    this.sampleParty = sampleParty;
  }

  public EvaluateSVM(SVMModelClosed model, List<DRes<SInt>> featureVector) {
    this(model, featureVector, 1);
  }

  @Override
  public DRes<SInt> buildComputation(ProtocolBuilderNumeric builder) {
    return builder.seq(par -> {
      List<List<DRes<SInt>>> supportVectors = model.getSupportVectors();
      List<DRes<SInt>> products = new ArrayList<>(supportVectors.size());
      for (int i = 0; i < supportVectors.size(); i++) {
        DRes<SInt> temp = par.advancedNumeric().innerProduct(supportVectors.get(i), featureVector);
        products.add(par.numeric().add(temp, model.getBias().get(i)));
      }
      return () -> products;
    }).par((par, products) -> {
      /**
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
      DRes<Pair<List<DRes<SInt>>, SInt>> min = par.seq(new Minimum(products));
      return () -> min.out().getSecond();
    });


    // }).whileLoop(pair -> pair.getSecond() < pair.getFirst().size(), (prevPar, pair) ->
    // prevPar.par(
    // par -> {
    // List<DRes<SInt>> products = pair.getFirst();
    // int offset = pair.getSecond();
    // for (int i = 0; i + offset < products.size(); i = i + 2 * offset) {
    // int leftPointer = i;
    // int rightPointer = i + offset;
    // for (int j = i; j < start + 2 * offset; j++) {
    // DRes<SInt> flag = par.comparison().compareLT(products.get(leftPointer), products.get(
    // rightPointer));
    // DRes<SInt> notFlag = par.logicalArithmetic().not(flag);
    // // Swap if needed
    // // First compute the new "left" element
    // DRes<SInt> temp1 = par.numeric().mult(flag, products.get(leftPointer));
    // DRes<SInt> temp2 = par.numeric().mult(notFlag, products.get(rightPointer));
    // DRes<SInt> leftElement = par.numeric().add(temp1, temp2);
    // // Compute the new "right" element
    // temp1 = par.numeric().mult(notFlag, products.get(j));
    // temp2 = par.numeric().mult(flag, products.get(j + offset));
    // DRes<SInt> rightElement = par.numeric().add(temp1, temp2);
    // products.set(j, leftElement);
    // products.set(j + offset, rightElement);
    // }
    // }
    // return () -> new Pair<List<DRes<SInt>>, Integer>(products, 2 * pair.getSecond());
    // }));
  }
}
