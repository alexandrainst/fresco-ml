package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.network.Network;
import dk.alexandra.fresco.framework.sce.SecureComputationEngine;
import dk.alexandra.fresco.framework.sce.resources.ResourcePool;
import dk.alexandra.fresco.framework.value.SInt;
import java.util.List;

public class SIntAveragingService<ResourcePoolT extends ResourcePool>
    implements AveragingService<SInt> {

  private WeightedModelParams<SInt> accumulator;
  private final SecureComputationEngine<ResourcePoolT, ProtocolBuilderNumeric> sce;
  private final ResourcePoolT rp;
  private final Network network;

  SIntAveragingService(SecureComputationEngine<ResourcePoolT, ProtocolBuilderNumeric> sce,
      Network network, ResourcePoolT rp) {
    this.accumulator = null;
    this.sce = sce;
    this.network = network;
    this.rp = rp;
  }

  @Override
  public void addToAverage(WeightedModelParams<SInt> params) {
    if (accumulator == null) {
      accumulator = params;
    } else {
      accumulator = sce.runApplication(new SumParams(accumulator, params), rp, network);
    }
  }

  @Override
  public List<DRes<SInt>> getAveragedParams() {
    return sce.runApplication(new Normalize(accumulator), rp, network);
  }

}
