package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.Party;
import dk.alexandra.fresco.framework.ProtocolEvaluator;
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
import dk.alexandra.fresco.framework.util.ExceptionConverter;
import dk.alexandra.fresco.framework.util.ModulusFinder;
import dk.alexandra.fresco.suite.ProtocolSuiteNumeric;
import dk.alexandra.fresco.suite.dummy.arithmetic.DummyArithmeticProtocolSuite;
import dk.alexandra.fresco.suite.dummy.arithmetic.DummyArithmeticResourcePool;
import dk.alexandra.fresco.suite.dummy.arithmetic.DummyArithmeticResourcePoolImpl;
import java.io.IOException;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * A test setup for running tests using the {@link DummyArithmeticProtocolSuite}.
 */
public class DummyArithTestSetup
    implements TestSetup<DummyArithmeticResourcePool, ProtocolBuilderNumeric> {

  private final CloseableNetwork network;
  private final DummyArithmeticResourcePool resourcePool;
  private final SecureComputationEngine<DummyArithmeticResourcePool, ProtocolBuilderNumeric> sce;

  /**
   * Constructs a test setup given the required resources.
   *
   * <p>
   * For conveniently building test setups for multiple parties see
   * {@link DummyArithTestSetup#builder(int)}.
   * </p>
   *
   * @param network
   *          a network
   * @param resourcePool
   *          a resource pool
   * @param sce
   *          an sce
   */
  public DummyArithTestSetup(CloseableNetwork network, DummyArithmeticResourcePool resourcePool,
      SecureComputationEngine<DummyArithmeticResourcePool, ProtocolBuilderNumeric> sce) {
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
  public DummyArithmeticResourcePool getRp() {
    return resourcePool;
  }

  @Override
  public SecureComputationEngine<DummyArithmeticResourcePool, ProtocolBuilderNumeric> getSce() {
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

    public static final int DEFAULT_MOD_LENGTH = 128;
    public static final int DEFAULT_MAX_LENGTH = 128;
    public static final int DEFAULT_PRECISION = 128;

    private int modLength = DEFAULT_MOD_LENGTH;
    private int maxLength = DEFAULT_MAX_LENGTH;
    private int precision = DEFAULT_PRECISION;
    private final int parties;

    private Builder(int parties) {
      this.parties = parties;
    }

    /**
     * Sets the modulus bit length of the test setups to be build.
     *
     * @param modLength
     *          the modulus bit length
     * @return <code>this</code>
     */
    public Builder modLength(int modLength) {
      this.modLength = modLength;
      return this;
    }

    /**
     * Sets the max bit length of the test setups to be build.
     *
     * @param maxLength
     *          the max bit length
     * @return <code>this</code>
     */
    public Builder maxLength(int maxLength) {
      this.maxLength = maxLength;
      return this;
    }

    /**
     * Sets the precision (used for real valued computations) of the test setups to be build.
     *
     * @param precision
     *          the precision
     * @return <code>this</code>
     */
    public Builder precision(int precision) {
      this.precision = precision;
      return this;
    }

    /**
     * Builds test setups for a number of parties using the specified parameters or default values
     * if none are given.
     *
     * @return a Map from party id to test setup
     */
    public Map<Integer, DummyArithTestSetup> build() {
      Map<Integer, DummyArithTestSetup> setups = new HashMap<>(parties);
      Map<Integer, CloseableNetwork> networks = createNetworks(parties);
      for (int i = 1; i < parties + 1; i++) {
        BigInteger modulus = ModulusFinder.findSuitableModulus(modLength);
        ProtocolSuiteNumeric<DummyArithmeticResourcePool> ps = new DummyArithmeticProtocolSuite(
            modulus, maxLength, precision);
        ProtocolEvaluator<DummyArithmeticResourcePool> evaluator = new BatchedProtocolEvaluator<>(
            new BatchedStrategy<>(), ps);
        CloseableNetwork net = networks.get(i);
        DummyArithmeticResourcePoolImpl rp = new DummyArithmeticResourcePoolImpl(i, parties,
            modulus);
        SecureComputationEngine<DummyArithmeticResourcePool, ProtocolBuilderNumeric> sce =
            new SecureComputationEngineImpl<>(
            ps, evaluator);
        setups.put(i, new DummyArithTestSetup(net, rp, sce));
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
