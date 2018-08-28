package dk.alexandra.fresco.ml.fl.demo;

import dk.alexandra.fresco.framework.Party;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.configuration.NetworkConfiguration;
import dk.alexandra.fresco.framework.configuration.NetworkConfigurationImpl;
import dk.alexandra.fresco.framework.network.AsyncNetwork;
import dk.alexandra.fresco.framework.network.CloseableNetwork;
import dk.alexandra.fresco.framework.sce.SecureComputationEngine;
import dk.alexandra.fresco.framework.sce.SecureComputationEngineImpl;
import dk.alexandra.fresco.framework.sce.evaluator.BatchedProtocolEvaluator;
import dk.alexandra.fresco.framework.sce.evaluator.BatchedStrategy;
import dk.alexandra.fresco.framework.util.AesCtrDrbg;
import dk.alexandra.fresco.framework.util.ModulusFinder;
import dk.alexandra.fresco.framework.util.OpenedValueStoreImpl;
import dk.alexandra.fresco.ml.fl.ClientFlHandler;
import dk.alexandra.fresco.ml.fl.DirectMpcFlHandler;
import dk.alexandra.fresco.ml.fl.FlTrainer;
import dk.alexandra.fresco.ml.fl.FlTrainerImpl;
import dk.alexandra.fresco.suite.spdz.SpdzProtocolSuite;
import dk.alexandra.fresco.suite.spdz.SpdzResourcePool;
import dk.alexandra.fresco.suite.spdz.SpdzResourcePoolImpl;
import dk.alexandra.fresco.suite.spdz.storage.SpdzDummyDataSupplier;
import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MpcFlMnistParty {

  private static final int BATCH_SIZE = 100000;
  private static final int MOD_BIT_LENGTH = 128;
  private static final int MAX_BIT_LENGTH = (MOD_BIT_LENGTH / 2) - 60;
  private static final int BASE_PORT = 3000;
  private static Logger logger = LoggerFactory.getLogger(MpcFlMnistParty.class);

  public static void main(String[] args) throws IOException, ParseException {
    CmdLineOptionsParser optionsParser = new CmdLineOptionsParser(args);
    // Setup context for FL
    SpdzTestSetup setup = createSetup(optionsParser.getMyId(), optionsParser.getNumParties());
    MnistTestContext context = MnistTestContext.builder()
        .localExamples(optionsParser.getNumExamples()).build();
    FlTrainer trainer = buildFlTrainer(setup, context);
    trainAndEval(setup, context, trainer);
    setup.close();
  }

  private static class CmdLineOptionsParser {
    private int myId;
    private int numParties;
    private int numExamples;

    CmdLineOptionsParser(String[] args) {
      // Parse demo parameters from commandline
      Options options = new Options();
      Option partyOpt = Option.builder("n").longOpt("num-parties").hasArg().argName("number")
          .desc("The number of parties to participate in the protocol "
              + "(must be larger than 0, all parties must use the same number)")
          .build();
      options.addOption(partyOpt);
      Option numExamplesOpt = Option.builder("l").longOpt("local-examples").hasArg()
          .argName("number").desc("The number of local examples used for training by this party "
              + "(must be in range [1, ..., 60000], all parties must use the same number)")
          .build();
      options.addOption(numExamplesOpt);
      Option idOpt = Option.builder("i").longOpt("party-id").hasArg().argName("id").required()
          .desc("The id of this party (party ids are consecutive and start from 1)").build();
      options.addOption(idOpt);
      CommandLineParser parser = new DefaultParser();
      try {
        CommandLine cmd = parser.parse(options, args);
        this.myId = Integer.parseInt(cmd.getOptionValue('i'));
        this.numParties = Integer.parseInt(cmd.getOptionValue('n', "1"));
        this.numExamples = Integer
            .parseInt(cmd.getOptionValue('l', "" + MnistTestContext.DEFAULT_LOCAL_EXAMPLES));
        if (this.numParties < 1) {
          throw new ParseException("Invalid number of parties: " + this.numParties);
        }
        if (this.myId < 1 || this.myId > this.numParties) {
          throw new ParseException("Invalid party id: " + this.myId);
        }
        if (this.numExamples < 1 || this.numExamples > 60000) {
          throw new ParseException("Invalid number of examples: " + this.numExamples);
        }
      } catch (ParseException e) {
        logger.error("Unable to parse commandline: " + e.getLocalizedMessage());
        HelpFormatter formatter = new HelpFormatter();
        formatter.printHelp("", options);
        System.exit(1);
      }
    }

    private int getMyId() {
      return myId;
    }

    private int getNumParties() {
      return numParties;
    }

    private int getNumExamples() {
      return numExamples;
    }

  }

  /**
   * Trains and evaluates a model with FL using SPDZ.
   *
   * @param setup
   *          the SPDZ setup
   * @param context
   *          the context for the MNIST test
   * @param trainer
   *          the trainer used to train the model
   * @throws IOException
   *           if the test data set cannot be read.
   */
  private static void trainAndEval(SpdzTestSetup setup, MnistTestContext context, FlTrainer trainer)
      throws IOException {
    Duration localTime = Duration.ZERO;
    Duration mpcTime = Duration.ZERO;
    Instant start = Instant.now();
    for (int i = 0; i < context.getGlobalEpochs(); i++) {
      Instant beforeLocal = Instant.now();
      trainer.fitLocalModel();
      Instant afterLocal = Instant.now();
      localTime = localTime.plus(Duration.between(beforeLocal, afterLocal));
      if (setup.getRp().getNoOfParties() > 1) {
        Instant beforeUpdate = Instant.now();
        trainer.updateGlobalModel();
        Instant afterUpdate = Instant.now();
        mpcTime = mpcTime.plus(Duration.between(beforeUpdate, afterUpdate));
      }
    }
    Instant stop = Instant.now();
    MnistDataSetIterator it = new MnistDataSetIterator(MnistTestContext.DEFAULT_BATCHSIZE, false,
        MnistTestContext.DEFAULT_SEED);
    Evaluation eval = trainer.getModel().evaluate(it);
    logger.info(eval.stats());
    Duration dur = Duration.between(start, stop);
    logger.info("Trained with {} parties each with {} local examples",
        setup.getRp().getNoOfParties(), context.getLocalExamples());
    logger.info("Total Time: {}", dur);
    logger.info("Local training time: {}", localTime);
    logger.info("MPC update time: {}", mpcTime);
  }

  private static FlTrainer buildFlTrainer(SpdzTestSetup setup, MnistTestContext context)
      throws IOException {
    ClientFlHandler flHandler = new DirectMpcFlHandler<>(setup.getSce(), setup.getRp(),
        setup.getNet());
    //DataSetIterator trainSet = new MnistDataSetIterator(context.getBatchSize(),
    //    context.getLocalExamples(), false, true, true, setup.getRp().getMyId());
    DataSetIterator trainSet = getTrainingData(setup.getRp().getMyId(), context.getLocalExamples(), context.getBatchSize());
    MultiLayerNetwork model = new MultiLayerNetwork(context.getConf());
    model.init();
    FlTrainer trainer = new FlTrainerImpl(flHandler, model, trainSet, context.getLocalEpochs(),
        context.getLocalExamples());
    return trainer;
  }

  private static DataSetIterator getTrainingData(int id, int examples, int batch) throws IOException {
    MnistDataSetIterator it = new MnistDataSetIterator(60000, 60000, false);
    DataSet data = it.next();
    data = data.filterBy(IntStream.range(0, 10).filter(i -> i != id).toArray());
    data = data.sample(examples);
    System.out.println("Examples: " + data.numExamples());
    return new ExistingDataSetIterator(data.batchBy(batch));
  }

  private static SpdzTestSetup createSetup(int myId, int numParties) {
    Map<Integer, Party> parties = new HashMap<>(numParties);
    for (int i = 1; i < numParties + 1; i++) {
      parties.put(i, new Party(i, "localhost", BASE_PORT + i));
    }
    NetworkConfiguration conf = new NetworkConfigurationImpl(myId, parties);
    CloseableNetwork net = new AsyncNetwork(conf);
    SpdzDummyDataSupplier supplier = new SpdzDummyDataSupplier(myId, numParties,
        ModulusFinder.findSuitableModulus(MOD_BIT_LENGTH));
    SpdzResourcePool rp = new SpdzResourcePoolImpl(myId, numParties, new OpenedValueStoreImpl<>(),
        supplier, new AesCtrDrbg());
    SpdzProtocolSuite suite = new SpdzProtocolSuite(MAX_BIT_LENGTH);
    SecureComputationEngine<SpdzResourcePool, ProtocolBuilderNumeric> sce = new SecureComputationEngineImpl<>(
        suite, new BatchedProtocolEvaluator<>(new BatchedStrategy<>(), suite, BATCH_SIZE));
    return new SpdzTestSetup(net, rp, sce);
  }

}
