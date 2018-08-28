package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.ProtocolEvaluator;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.network.CloseableNetwork;
import dk.alexandra.fresco.framework.sce.SecureComputationEngine;
import dk.alexandra.fresco.framework.sce.SecureComputationEngineImpl;
import dk.alexandra.fresco.framework.sce.evaluator.BatchedProtocolEvaluator;
import dk.alexandra.fresco.framework.sce.evaluator.BatchedStrategy;
import dk.alexandra.fresco.framework.util.ModulusFinder;
import dk.alexandra.fresco.ml.fl.demo.TestSetup;
import dk.alexandra.fresco.suite.ProtocolSuiteNumeric;
import dk.alexandra.fresco.suite.dummy.arithmetic.DummyArithmeticProtocolSuite;
import dk.alexandra.fresco.suite.dummy.arithmetic.DummyArithmeticResourcePool;
import dk.alexandra.fresco.suite.dummy.arithmetic.DummyArithmeticResourcePoolImpl;
import java.io.IOException;
import java.math.BigInteger;

/**
 * A TestSetup for just a single party using the Dummy Arithmetic protocol suite. Useful for doing
 * basic testing of MPC computations.
 */
public class SinglePartyTestSetup
    implements TestSetup<DummyArithmeticResourcePool, ProtocolBuilderNumeric> {

  private final CloseableNetwork net;
  private final DummyArithmeticResourcePool rp;
  private final SecureComputationEngine<DummyArithmeticResourcePool, ProtocolBuilderNumeric> sce;
  private static final int DEFAULT_MOD_LENGTH = 128;
  private static final int DEFAULT_MAX_LENGTH = 128;
  private static final int DEFAULT_PRECISION = 128;

  /**
   * Creates a single party test set up.
   */
  public SinglePartyTestSetup() {
    CloseableNetwork net = new SinglePartyTestNetwork();
    BigInteger modulus = ModulusFinder.findSuitableModulus(DEFAULT_MOD_LENGTH);
    ProtocolSuiteNumeric<DummyArithmeticResourcePool> ps = new DummyArithmeticProtocolSuite(modulus,
        DEFAULT_MAX_LENGTH, DEFAULT_PRECISION);
    ProtocolEvaluator<DummyArithmeticResourcePool> evaluator = new BatchedProtocolEvaluator<>(
        new BatchedStrategy<>(), ps);
    this.net = net;
    this.rp = new DummyArithmeticResourcePoolImpl(1, 1, modulus);
    this.sce = new SecureComputationEngineImpl<>(ps, evaluator);
  }

  /*
   * (non-Javadoc)
   *
   * @see dk.alexandra.fresco.ml.fl.TestSetup#getNet()
   */
  @Override
  public CloseableNetwork getNet() {
    return net;
  }

  /*
   * (non-Javadoc)
   *
   * @see dk.alexandra.fresco.ml.fl.TestSetup#getRp()
   */
  @Override
  public DummyArithmeticResourcePool getRp() {
    return rp;
  }

  /*
   * (non-Javadoc)
   *
   * @see dk.alexandra.fresco.ml.fl.TestSetup#getSce()
   */
  @Override
  public SecureComputationEngine<DummyArithmeticResourcePool, ProtocolBuilderNumeric> getSce() {
    return sce;
  }

  /*
   * (non-Javadoc)
   *
   * @see dk.alexandra.fresco.ml.fl.TestSetup#close()
   */
  @Override
  public void close() throws IOException {
    this.getNet().close();
    this.getSce().shutdownSCE();
  }

}
