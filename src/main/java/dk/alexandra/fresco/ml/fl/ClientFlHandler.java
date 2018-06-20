package dk.alexandra.fresco.ml.fl;

/**
 * Handles the client side of a Federated Learning protocol. This should be the interface towards
 * the server doing the model averaging. I.e., this handles submitting of a locally trained model
 * to a server doing the averaging and the fetching of a global averaged model.
 */
public interface ClientFlHandler {

  /**
   * Fetch the model resulting from averaging the submitted local models (the global model) from the
   * averaging server(s).
   * <p>
   * This call is expected to block until the model is ready.
   * </p>
   *
   * @return the averaged model
   */
  FlatModel getAveragedModel();

  /**
   * Submit a locally trained model to be averaged along with models submitted from other parties.
   *
   * @param model
   *          a locally trained model
   */
  void submitLocalModel(FlatModel model);

}