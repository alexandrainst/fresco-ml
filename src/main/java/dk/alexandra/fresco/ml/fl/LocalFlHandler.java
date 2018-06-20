package dk.alexandra.fresco.ml.fl;

import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * An handler for Federated Learning implementing both the client and server side and does the
 * averaging locally.
 *
 * <p>
 * All 'parties' are simulated locally and the averaging is done directly in the clear. This is
 * meant mainly as a way to test and validate the FL approach.
 *
 * Note: this class is not thread safe.
 * </p>
 *
 */
public final class LocalFlHandler implements ClientFlHandler, ServerFlHandler {

  private enum State {
    COLLECT_MODELS, DISTRIBUTE_MODEL
  }

  private final List<FlatModel> localModels;
  private FlatModel averageModel;
  private State state;

  public LocalFlHandler() {
    this.localModels = new ArrayList<>();
    this.state = State.COLLECT_MODELS;
  }

  @Override
  public void submitLocalModel(FlatModel model) {
    state = State.COLLECT_MODELS;
    localModels.add(model);
  }

  @Override
  public FlatModel getAveragedModel() {
    if (state == State.COLLECT_MODELS) {
      INDArray weightedSum = localModels.stream()
          .map(model -> model.getParams().mul(model.getExamples())).reduce((m1, m2) -> m1.add(m2))
          .get();
      int totalExamples = localModels.stream().mapToInt(FlatModel::getExamples).sum();
      averageModel = new FlatModel(weightedSum.div(totalExamples), totalExamples);
      localModels.clear();
      state = State.DISTRIBUTE_MODEL;
    }
    return averageModel;
  }

}
