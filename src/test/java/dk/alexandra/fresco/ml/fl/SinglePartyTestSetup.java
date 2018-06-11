package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.ProtocolEvaluator;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.network.CloseableNetwork;
import dk.alexandra.fresco.framework.sce.SecureComputationEngine;
import dk.alexandra.fresco.framework.sce.SecureComputationEngineImpl;
import dk.alexandra.fresco.framework.sce.evaluator.BatchedProtocolEvaluator;
import dk.alexandra.fresco.framework.sce.evaluator.BatchedStrategy;
import dk.alexandra.fresco.framework.util.ModulusFinder;
import dk.alexandra.fresco.suite.ProtocolSuiteNumeric;
import dk.alexandra.fresco.suite.dummy.arithmetic.DummyArithmeticProtocolSuite;
import dk.alexandra.fresco.suite.dummy.arithmetic.DummyArithmeticResourcePool;
import dk.alexandra.fresco.suite.dummy.arithmetic.DummyArithmeticResourcePoolImpl;
import java.io.IOException;
import java.math.BigInteger;

public class SinglePartyTestSetup {

  private final CloseableNetwork net;
  private final DummyArithmeticResourcePool rp;
  private final SecureComputationEngine<DummyArithmeticResourcePool, ProtocolBuilderNumeric> sce;
  private static final int DEFAULT_MOD_LENGTH = 32;
  private static final int DEFAULT_MAX_LENGTH = 16;
  private static final int DEFAULT_PRECISION = 16;

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

  public CloseableNetwork getNet() {
    return net;
  }

  public DummyArithmeticResourcePool getRp() {
    return rp;
  }

  public SecureComputationEngine<DummyArithmeticResourcePool, ProtocolBuilderNumeric> getSce() {
    return sce;
  }

  public void close() throws IOException {
    this.getNet().close();
    this.getSce().shutdownSCE();
  }

}
