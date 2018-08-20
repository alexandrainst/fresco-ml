package dk.alexandra.fresco.ml.dtrees;

import dk.alexandra.fresco.framework.network.Network;
import dk.alexandra.fresco.framework.sce.evaluator.EvaluationStrategy;
import dk.alexandra.fresco.framework.util.AesCtrDrbg;
import dk.alexandra.fresco.suite.ProtocolSuiteNumeric;
import dk.alexandra.fresco.suite.spdz2k.AbstractSpdz2kTest;
import dk.alexandra.fresco.suite.spdz2k.Spdz2kProtocolSuite128;
import dk.alexandra.fresco.suite.spdz2k.datatypes.CompUInt128;
import dk.alexandra.fresco.suite.spdz2k.datatypes.CompUInt128Factory;
import dk.alexandra.fresco.suite.spdz2k.datatypes.CompUIntFactory;
import dk.alexandra.fresco.suite.spdz2k.resource.Spdz2kResourcePool;
import dk.alexandra.fresco.suite.spdz2k.resource.Spdz2kResourcePoolImpl;
import dk.alexandra.fresco.suite.spdz2k.resource.storage.Spdz2kDummyDataSupplier;
import dk.alexandra.fresco.suite.spdz2k.resource.storage.Spdz2kOpenedValueStoreImpl;
import java.util.function.Supplier;
import org.junit.Test;

public class TestDecisionTreeComputationsSpdz2k extends
    AbstractSpdz2kTest<Spdz2kResourcePool<CompUInt128>> {

  @Test
  public void testEvaluateDecisionTree() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTree<>(),
        EvaluationStrategy.SEQUENTIAL_BATCHED,
        2);
  }

  @Test
  public void testEvaluateDecisionTreeFour() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTreeFour<>(),
        EvaluationStrategy.SEQUENTIAL_BATCHED,
        2);
  }

  @Test
  public void testEvaluateDecisionTreeFive() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTreeFive<>(),
        EvaluationStrategy.SEQUENTIAL_BATCHED,
        2);
  }

  @Test
  public void testEvaluateDecisionTreeSix() {
    runTest(new DecisionTreeComputationTests.TestEvaluateDecisionTreeSix<>(),
        EvaluationStrategy.SEQUENTIAL_BATCHED,
        2);
  }

  @Override
  protected Spdz2kResourcePool<CompUInt128> createResourcePool(int playerId, int noOfParties,
      Supplier<Network> networkSupplier) {
    CompUIntFactory<CompUInt128> factory = new CompUInt128Factory();
    Spdz2kResourcePool<CompUInt128> resourcePool =
        new Spdz2kResourcePoolImpl<>(
            playerId,
            noOfParties, null,
            new Spdz2kOpenedValueStoreImpl<>(),
            new Spdz2kDummyDataSupplier<>(playerId, noOfParties, factory.createRandom(), factory),
            factory);
    resourcePool.initializeJointRandomness(networkSupplier, AesCtrDrbg::new, 32);
    return resourcePool;
  }

  @Override
  protected ProtocolSuiteNumeric<Spdz2kResourcePool<CompUInt128>> createProtocolSuite() {
    return new Spdz2kProtocolSuite128(true);
  }

}
