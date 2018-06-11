package dk.alexandra.fresco.ml.fl;

import java.util.ArrayList;
import java.util.Collection;

public final class ClientLocalFlHandler implements ClientFlProtocolHandler {

  private enum State {
    COLLECT_MODELS, DISTRIBUTE_MODEL
  }

  private final Collection<LocalModel> localModels;
  private LocalModel globalModel;
  private State state;

  public ClientLocalFlHandler() {
    this.localModels = new ArrayList<>();
    this.state = State.COLLECT_MODELS;
  }

  @Override
  public void submitLocalModel(LocalModel model) {
    state = State.COLLECT_MODELS;
    localModels.add(model);
  }

  @Override
  public LocalModel getGlobalModel() {
    if (state == State.COLLECT_MODELS) {
      LocalModel sumModel = localModels.stream().reduce(LocalModel::addModels).get();
      globalModel = LocalModel.divModel(sumModel, sumModel.examples);
      localModels.clear();
      state = State.DISTRIBUTE_MODEL;
    }
    return globalModel;
  }

}
