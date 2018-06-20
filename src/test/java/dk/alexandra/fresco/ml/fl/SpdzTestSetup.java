package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.Party;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.configuration.NetworkConfiguration;
import dk.alexandra.fresco.framework.configuration.NetworkConfigurationImpl;
import dk.alexandra.fresco.framework.configuration.NetworkTestUtils;
import dk.alexandra.fresco.framework.network.AsyncNetwork;
import dk.alexandra.fresco.framework.network.CloseableNetwork;
import dk.alexandra.fresco.framework.sce.SecureComputationEngine;
import dk.alexandra.fresco.framework.sce.SecureComputationEngineImpl;
import dk.alexandra.fresco.framework.sce.evaluator.BatchedProtocolEvaluator;
import dk.alexandra.fresco.framework.sce.evaluator.BatchedStrategy;
import dk.alexandra.fresco.framework.util.AesCtrDrbg;
import dk.alexandra.fresco.framework.util.ExceptionConverter;
import dk.alexandra.fresco.framework.util.ModulusFinder;
import dk.alexandra.fresco.framework.util.OpenedValueStoreImpl;
import dk.alexandra.fresco.suite.spdz.SpdzProtocolSuite;
import dk.alexandra.fresco.suite.spdz.SpdzResourcePool;
import dk.alexandra.fresco.suite.spdz.SpdzResourcePoolImpl;
import dk.alexandra.fresco.suite.spdz.storage.SpdzDummyDataSupplier;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * A TestSetup using the SPDZ protocol suite to run MPC computations. This will use dummy
 * preprocessing.
 */
public class SpdzTestSetup implements TestSetup<SpdzResourcePool, ProtocolBuilderNumeric> {

  private final CloseableNetwork network;
  private final SpdzResourcePool resourcePool;
  private final SecureComputationEngine<SpdzResourcePool, ProtocolBuilderNumeric> sce;

  /**
   * Constructs a SpdzTestSetup given the required resources.
   *
   * @param network a network
   * @param resourcePool a SpdzResourcePool
   * @param sce an sce
   */
  public SpdzTestSetup(CloseableNetwork network, SpdzResourcePool resourcePool,
      SecureComputationEngine<SpdzResourcePool, ProtocolBuilderNumeric> sce) {
    this.network = network;
    this.resourcePool = resourcePool;
    this.sce = sce;
  }

  /**
   * Returns a new {@link Builder} used to build tests setups for a given number of parties.
   *
   * @param parties
   *          the number of parties.
   * @return a new Builder
   */
  public static Builder builder(int parties) {
    return new Builder(parties);
  }

  @Override
  public CloseableNetwork getNet() {
    return network;
  }

  @Override
  public SpdzResourcePool getRp() {
    return resourcePool;
  }

  @Override
  public SecureComputationEngine<SpdzResourcePool, ProtocolBuilderNumeric> getSce() {
    return sce;
  }

  @Override
  public void close() throws IOException {
    sce.shutdownSCE();
    network.close();
  }

  /**
   * Builder class used to configure and build test setups for a set of parties.
   */
  public static class Builder {

    private static final int DEFAULT_MOD_BIT_LENGTH = 128;
    private static final int DEFAULT_MAX_BIT_LENGTH = 128;

    private int maxLength = DEFAULT_MAX_BIT_LENGTH;
    private int modLength = DEFAULT_MOD_BIT_LENGTH;
    private int parties;

    public Builder(int parties) {
      this.parties = parties;
    }

    public Builder modLength(int modLength) {
      this.modLength = modLength;
      return this;
    }

    public Builder maxLength(int maxLength) {
      this.maxLength = maxLength;
      return this;
    }

    /**
     * Builds test setups for a number of parties using the specified parameters or default values
     * if none are given.
     *
     * @return a Map from party id to test setup
     */
    public Map<Integer, SpdzTestSetup> build() {
      Map<Integer, CloseableNetwork> netMap = createNetworks(parties);
      Map<Integer, SpdzTestSetup> setups = new HashMap<>(parties);
      for (int i = 1; i < parties + 1; i++) {
        SpdzDummyDataSupplier supplier = new SpdzDummyDataSupplier(i, parties,
            ModulusFinder.findSuitableModulus(modLength));
        SpdzResourcePool rp = new SpdzResourcePoolImpl(i, parties, new OpenedValueStoreImpl<>(),
            supplier, new AesCtrDrbg(new byte[32]));
        SpdzProtocolSuite suite = new SpdzProtocolSuite(maxLength);
        SecureComputationEngine<SpdzResourcePool, ProtocolBuilderNumeric> sce =
            new SecureComputationEngineImpl<>(
            suite, new BatchedProtocolEvaluator<>(new BatchedStrategy<>(), suite));
        setups.put(i, new SpdzTestSetup(netMap.get(i), rp, sce));
      }
      return setups;
    }

    private Map<Integer, CloseableNetwork> createNetworks(int parties) {
      return createNetworks(getNetConfs(parties));
    }

    private Map<Integer, CloseableNetwork> createNetworks(List<NetworkConfiguration> confs) {
      int numParties = confs.get(0).noOfParties();
      Map<Integer, CloseableNetwork> netMap = new HashMap<>(numParties);
      Map<Integer, Future<CloseableNetwork>> futureMap = new HashMap<>(numParties);
      ExecutorService es = Executors.newFixedThreadPool(numParties);
      for (int i = 1; i < numParties + 1; i++) {
        final NetworkConfiguration conf = confs.get(i - 1);
        Future<CloseableNetwork> f = es.submit(() -> {
          return new AsyncNetwork(conf);
        });
        futureMap.put(i, f);
      }
      for (int i = 1; i < numParties + 1; i++) {
        Future<CloseableNetwork> f = futureMap.get(i);
        CloseableNetwork net = ExceptionConverter.safe(() -> f.get(), "Unable to create networks");
        netMap.put(i, net);
      }
      return netMap;
    }

    private List<NetworkConfiguration> getNetConfs(int numParties) {
      Map<Integer, Party> parties = new HashMap<>(numParties);
      List<NetworkConfiguration> confs = new ArrayList<>(numParties);
      List<Integer> ports = NetworkTestUtils.getFreePorts(numParties);
      int id = 1;
      for (Integer port : ports) {
        parties.put(id, new Party(id, "localhost", port));
        id++;
      }
      for (int i = 1; i <= numParties; i++) {
        confs.add(new NetworkConfigurationImpl(i, parties));
      }
      return confs;
    }
  }

}
