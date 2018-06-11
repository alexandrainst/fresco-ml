package dk.alexandra.fresco.ml.fl;

import dk.alexandra.fresco.framework.util.Pair;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.ndarray.INDArray;

public final class LocalFederatedTrainingServer {

  private int numClients;
  private AtomicInteger counter;
  private TrainingRound[] rounds;

  public LocalFederatedTrainingServer(int numClients) {
    this.numClients = numClients;
    this.counter = new AtomicInteger(0);
    this.rounds = new TrainingRound[2];
    rounds[0] = new TrainingRound(numClients);
  }

  TrainingRound nextRound() {
    int currentCount = counter.getAndIncrement();
    int currentRound = (currentCount / numClients);
    // If we are the first thread getting the current round, we should prepare next round
    if (currentCount % numClients == 0) {
      rounds[1 - (currentRound % 2)] = new TrainingRound(numClients);
    }
    return rounds[currentRound % 2];
  }

  static class TrainingRound {

    private Collection<LocalModel> localModels = new ConcurrentLinkedQueue<>();
    private CountDownLatch collectLatch;

    TrainingRound(int numClients) {
      this.collectLatch = new CountDownLatch(numClients);
    }

    TrainingRound addLocalModel(List<INDArray> weights, List<INDArray> biases, int examples) {
      localModels.add(new LocalModel(weights, biases, examples));
      collectLatch.countDown();
      return this;
    }

    Pair<List<INDArray>, List<INDArray>> getGobalModel() {
      try {
        collectLatch.await();
      } catch (InterruptedException e) {
        e.printStackTrace();
        throw new RuntimeException("Something went wrong and we cant handle it", e);
      }
      LocalModel sumModel = localModels.stream().reduce((a, b) -> {
        List<INDArray> weights = IntStream.range(0, a.getWeights().size())
            .mapToObj(i -> a.getWeights().get(i).add(b.getWeights().get(i)))
            .collect(Collectors.toList());
        List<INDArray> biases = IntStream.range(0, a.getBiases().size())
            .mapToObj(i -> a.getBiases().get(i).add(b.getBiases().get(i)))
            .collect(Collectors.toList());
        int examples = a.getExamples() + b.getExamples();
        return new LocalModel(weights, biases, examples);
      }).get();
      List<INDArray> weights = sumModel.getWeights().stream().map(w -> w.div(sumModel.examples)).collect(Collectors.toList());
      List<INDArray> biases = sumModel.getBiases().stream().map(b -> b.div(sumModel.examples)).collect(Collectors.toList());
      return new Pair<>(weights, biases);
    }
  }

  private static class LocalModel {

    List<INDArray> weights;
    List<INDArray> biases;
    int examples;

    private LocalModel(List<INDArray> weights, List<INDArray> biases, int examples) {
      this.weights = weights;
      this.biases = biases;
      this.examples = examples;
    }

    private List<INDArray> getWeights() {
      return weights;
    }

    private List<INDArray> getBiases() {
      return biases;
    }

    private int getExamples() {
      return examples;
    }

  }

}
