package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.network.CloseableNetwork;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Deque;

/**
 * An dummy implementation of a network consisting of just one party.
 * Useful for testing with the dummy protocol suite.
 */
public final class SinglePartyTestNetwork implements CloseableNetwork {
  Deque<byte[]> queue = new ArrayDeque<>();

  @Override
  public void send(int partyId, byte[] data) {
    if (partyId == 1) {
      queue.add(data);
    }

  }

  @Override
  public byte[] receive(int partyId) {
    if (partyId == 1) {
      return queue.poll();
    } else {
      return null;
    }
  }

  @Override
  public int getNoOfParties() {
    return 1;
  }

  @Override
  public void close() throws IOException {
    queue.clear();
  }
}