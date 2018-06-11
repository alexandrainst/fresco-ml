package dk.alexandra.fresco.ml.fl;

public interface ClientFlProtocolHandler {

  LocalModel getGlobalModel();

  void submitLocalModel(LocalModel model);

}