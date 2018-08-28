package dk.alexandra.fresco.ml.fl.demo;

import dk.alexandra.fresco.framework.builder.ProtocolBuilder;
import dk.alexandra.fresco.framework.network.CloseableNetwork;
import dk.alexandra.fresco.framework.sce.SecureComputationEngine;
import dk.alexandra.fresco.framework.sce.resources.ResourcePool;
import java.io.IOException;

/**
 * Interface for conveniently handling MPC test setups that holds all the elements needed to run an
 * MPC application.
 *
 * @param <ResourcePoolT>
 *          the resource pool type to use in the MPC computation
 * @param <BuilderT>
 *          the builder type used
 */
public interface TestSetup<ResourcePoolT extends ResourcePool, BuilderT extends ProtocolBuilder> {

  /**
   * Returns a network connected to the other parties to be used in MPC computations.
   *
   * @return a network
   */
  CloseableNetwork getNet();

  /**
   * Returns a resource pool to be used in MPC computations.
   *
   * @return a resource pool
   */
  ResourcePoolT getRp();

  /**
   * A SecureComputationEngine which can run an MPC computation.
   *
   * @return an sce
   */
  SecureComputationEngine<ResourcePoolT, BuilderT> getSce();

  /**
   * Closes the resources held in the setup.
   *
   * @throws IOException
   *           if an IO related exception occurs while closing resources (e.g., the network).
   */
  void close() throws IOException;

}