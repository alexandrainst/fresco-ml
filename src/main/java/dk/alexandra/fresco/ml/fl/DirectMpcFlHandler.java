package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.Application;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.Computation;
import dk.alexandra.fresco.framework.builder.numeric.NumericResourcePool;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.network.Network;
import dk.alexandra.fresco.framework.sce.SecureComputationEngine;
import dk.alexandra.fresco.framework.util.ExceptionConverter;
import dk.alexandra.fresco.framework.value.SInt;
import java.math.BigInteger;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A simple implementation of Federated Learning where clients directly implement the averaging
 * server using MPC (as opposed to an outsourcing model where clients and parties impl).
 *
 * <p>
 * This assumes that each party will submit exactly one locally trained model for an externally
 * specified number of rounds. This does not implement any further coordination of averaging rounds
 * (i.e., does not do timeouts, handling missing inputs and so on).
 * </p>
 *
 * @param <ResourcePoolT>
 *          the resource pool type used
 */
public final class DirectMpcFlHandler<ResourcePoolT extends NumericResourcePool>
    implements ClientFlHandler, ServerFlHandler {

  private static final int DEFAULT_SCALE_UP = 1000_000_000;
  private final SecureComputationEngine<ResourcePoolT, ProtocolBuilderNumeric> sce;
  private final ResourcePoolT rp;
  private final Network network;
  private Future<FlatModel> modelFuture;

  /**
   * Constructs the handler for MPC based Federated Learning.
   *
   * @param sce
   *          the engine to run the MPC computations on
   * @param rp
   *          the resource pool for the computation
   * @param network
   *          the network to communicate over
   */
  public DirectMpcFlHandler(SecureComputationEngine<ResourcePoolT, ProtocolBuilderNumeric> sce,
      ResourcePoolT rp, Network network) {
    this.sce = sce;
    this.rp = rp;
    this.network = network;
  }

  @Override
  public FlatModel getAveragedModel() {
    return ExceptionConverter.safe(() -> modelFuture.get(),
        "Error occured while computing the averaged model");
  }

  @Override
  public void submitLocalModel(FlatModel model) {
    modelFuture = this.sce.startApplication(processModel(model), this.rp, this.network);
  }

  /**
   * Constructs an MPC Application to process a submitted locally trained model.
   *
   * <p>
   * This will take as input a locally trained model from each party and add it to a running
   * average.
   * </p>
   *
   * @param model
   *          the locally trained model of this party
   * @return an MPC Application to compute the average of all parties local models.
   */
  private Application<FlatModel, ProtocolBuilderNumeric> processModel(final FlatModel model) {
    return builder -> {
      Averager<SInt, ProtocolBuilderNumeric> averager = new SIntAverager<>();
      final int numParams = model.getParams().length();
      DRes<FlatModel> m = builder.seq(s -> () -> 1)
          .whileLoop(index -> index < rp.getNoOfParties() + 1, (whileBuilder, index) -> {
            if (index == rp.getMyId()) {
              List<BigInteger> params = prepareForInput(model);
              BigInteger examples = BigInteger.valueOf(model.getExamples());
              whileBuilder.seq(inputMyParams(params, examples))
                  .seq((seq, closedParams) -> seq.seq(averager.addToAverage(closedParams)));
            } else {
              whileBuilder.seq(inputOtherParams(numParams, index))
                  .seq((seq, closedParams) -> seq.seq(averager.addToAverage(closedParams)));
            }
            return () -> index + 1;
          }).seq((seq, index) -> seq.seq(averager.getAveragedParams()))
          .seq((seq, closedAverage) -> seq.seq(openAverage(closedAverage)))
          .seq((seq, openAverage) -> () -> postProcessModel(
              openAverage.stream().map(DRes::out).collect(Collectors.toList())));
      return m;
    };
  }

  private Computation<List<DRes<BigInteger>>, ProtocolBuilderNumeric> openAverage(
      List<DRes<SInt>> params) {
    return builder -> builder.collections().openList(() -> params);
  }

  private Computation<WeightedModelParams<SInt>, ProtocolBuilderNumeric> inputOtherParams(int size,
      final int id) {
    return builder -> builder.par(parBuilder -> {
      DRes<List<DRes<SInt>>> closedPars = parBuilder.collections().closeList(size, id);
      DRes<SInt> closedWeight = parBuilder.numeric().input(null, id);
      return () -> new WeightedModelParamsImpl<>(closedWeight, closedPars.out());
    });
  }

  private Computation<WeightedModelParams<SInt>, ProtocolBuilderNumeric> inputMyParams(
      List<BigInteger> params, BigInteger examples) {
    return builder -> builder.par(parBuilder -> {
      DRes<List<DRes<SInt>>> closedParams = parBuilder.collections().closeList(params,
          rp.getMyId());
      DRes<SInt> closedWeight = parBuilder.numeric().input(examples, rp.getMyId());
      return () -> new WeightedModelParamsImpl<>(closedWeight, closedParams.out());
    });
  }

  /**
   * Converts locally trained model parameters to a representation that we can input to the MPC
   * computation.
   *
   * <p>
   * This works as follows:
   * <ol>
   * <li>We scale the parameters by the by the weight of the model (i.e., the number of examples it
   * was trained on). This is to save us having to do this multiplication in MPC.
   * <li>We scale up the floating point representation of the model parameters by a factor
   * {@link DirectMpcFlHandler#DEFAULT_SCALE_UP}.
   * <li>We truncate this value to an integer (BigInteger).
   * </ol>
   * </p>
   *
   * @param model
   *          the locally trained model parameters
   * @return a representation of the model parameters fit for MPC
   */
  private List<BigInteger> prepareForInput(FlatModel model) {
    double[] doubleParams = model.getParams().mul(model.getExamples()).mul(DEFAULT_SCALE_UP)
        .toDoubleVector();
    if (Arrays.stream(doubleParams).anyMatch(d -> d > Long.MAX_VALUE)) {
      throw new IllegalStateException();
    }
    return Arrays.stream(doubleParams).mapToLong(d -> (long) d).mapToObj(BigInteger::valueOf)
        .collect(Collectors.toList());
  }

  /**
   * Converts the representation of model parameters used in MPC back into one that can be used for
   * local training.
   *
   * <p>
   * Essentially, this reverses what we do in {@link DirectMpcFlHandler#prepareForInput(FlatModel)}.
   * However, here we also apply rational reconstruction in order go from a field element to a
   * double value.
   * </p>
   *
   * @param average
   *          a list of averaged model parameters output from the MPC computation
   * @return a representation of the model parameters for for further local training.
   */
  private FlatModel postProcessModel(List<BigInteger> average) {
    double[] params = average.stream().map(p -> FlTestUtils.gauss(p, rp.getModulus()))
        .mapToDouble(f -> f[0].divide(f[1]).doubleValue()).map(d -> d / DEFAULT_SCALE_UP).toArray();
    return new FlatModel(Nd4j.create(params), -1);
  }

}
