package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.Application;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.numeric.NumericResourcePool;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.network.Network;
import dk.alexandra.fresco.framework.sce.SecureComputationEngine;
import dk.alexandra.fresco.framework.value.SInt;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DirectMpcFlHandler<ResourcePoolT extends NumericResourcePool>
    implements ClientFlProtocolHandler, ServerFlProtocolHandler {

  private static final int DEFAULT_SCALE_UP = 10000;
  private final SecureComputationEngine<ResourcePoolT, ProtocolBuilderNumeric> sce;
  private final ResourcePoolT rp;
  private final Network network;
  private LocalModel globalModel;

  public DirectMpcFlHandler(SecureComputationEngine<ResourcePoolT,
      ProtocolBuilderNumeric> sce,
      ResourcePoolT rp, Network network) {
    this.sce = sce;
    this.rp = rp;
    this.network = network;
  }

  @Override
  public LocalModel getGlobalModel() {
    return globalModel;
  }

  @Override
  public void submitLocalModel(LocalModel model) {
    List<BigInteger> params = prepareForInput(model); // Client side
    AveragingService<SInt> service = new SIntAveragingService<>(sce, network, rp);
    for (int i = 1; i <= rp.getNoOfParties(); i++) {
      WeightedModelParams<SInt> weightedParams;
      if (i == rp.getMyId()) {
        BigInteger examples = BigInteger.valueOf(model.getExamples());
        weightedParams = inputMyParams(params, examples);// Client-Server protocol side
      } else {
        weightedParams = inputOtherParams(params.size(), i); // Client-Server (other server) side
      }
      service.addToAverage(weightedParams); // Server side
    }
    List<SInt> closedAverage = service.getAveragedParams().stream()
        .map(DRes::out)
        .collect(Collectors.toList());
    List<BigInteger> average = openAverage(closedAverage); // Client-Server protocol
    List<int[]> weightShapes = model.getWeights().stream().map(INDArray::shape)
        .collect(Collectors.toList());
    List<int[]> biasesShapes = model.getBiases().stream().map(INDArray::shape)
        .collect(Collectors.toList());
    globalModel = postProcessModel(average, weightShapes, biasesShapes);
  }

  private LocalModel postProcessModel(List<BigInteger> average,
      List<int[]> weightShapes,
      List<int[]> biasShapes) {
    double[] params = average.stream()
        .map(p -> FlTestUtils.gauss(p, rp.getModulus()))
        .mapToDouble(f -> f[0].doubleValue() / f[1].doubleValue())
        .map(d -> d / DEFAULT_SCALE_UP)
        .toArray();
    List<INDArray> weights = new ArrayList<>(weightShapes.size());
    int offset = 0;
    for (int[] shape: weightShapes) {
      int size = IntStream.of(shape).sum();
      weights.add(Nd4j.create(params, shape, offset));
      offset += size;
    }
    List<INDArray> biases = new ArrayList<>(biasShapes.size());
    for (int[] shape: biasShapes) {
      int size = IntStream.of(shape).sum();
      biases.add(Nd4j.create(params, shape, offset));
      offset += size;
    }
    return new LocalModel(weights, biases, -1);
  }

  private <OutputT> OutputT run(Application<OutputT, ProtocolBuilderNumeric> app) {
    return sce.runApplication(app, rp, network);
  }

  private List<BigInteger> openAverage(List<SInt> closedAverage) {
    return run(builder -> {
      return builder.collections().openList(() -> closedAverage);
    }).stream().map(DRes::out).collect(Collectors.toList());
  }

  private WeightedModelParams<SInt> inputOtherParams(int size, final int id) {
    WeightedModelParams<SInt> weightedParams;
    weightedParams = run(builder -> builder.par(parBuilder -> {
      DRes<List<DRes<SInt>>> closedPars = parBuilder.collections().closeList(size, id);
      DRes<SInt> closedWeight = parBuilder.numeric().input(null, id);
      return () -> new WeightedModelParamsImpl<>(closedWeight, closedPars.out());
    }));
    return weightedParams;
  }

  private WeightedModelParams<SInt> inputMyParams(List<BigInteger> params, BigInteger examples) {
    WeightedModelParams<SInt> weightedParams;
    weightedParams = run(builder -> builder.par(parBuilder -> {
      DRes<List<DRes<SInt>>> closedParams = parBuilder.collections()
          .closeList(params, rp.getMyId());
      DRes<SInt> closedWeight = parBuilder.numeric().input(examples, rp.getMyId());
      return () -> new WeightedModelParamsImpl<>(closedWeight, closedParams.out());
    }));
    return weightedParams;
  }

  private List<BigInteger> prepareForInput(LocalModel model) {
    List<BigInteger> params = Stream.concat(model.getWeights().stream(), model.getBiases().stream())
        .map(w -> w.mul(model.getExamples() * DEFAULT_SCALE_UP))
        .map(INDArray::toIntVector)
        .map(Arrays::stream)
        .flatMap(IntStream::boxed)
        .map(BigInteger::valueOf)
        .collect(Collectors.toList());
    return params;
  }

}
