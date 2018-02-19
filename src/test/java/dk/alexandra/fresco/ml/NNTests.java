package dk.alexandra.fresco.ml;

import dk.alexandra.fresco.decimal.RealNumeric;
import dk.alexandra.fresco.decimal.RealNumericProvider;
import dk.alexandra.fresco.decimal.SReal;
import dk.alexandra.fresco.decimal.floating.binary.BinaryFloatNumeric;
import dk.alexandra.fresco.framework.Application;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.TestThreadRunner.TestThread;
import dk.alexandra.fresco.framework.TestThreadRunner.TestThreadFactory;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.sce.resources.ResourcePool;
import dk.alexandra.fresco.lib.collections.Matrix;
import dk.alexandra.fresco.lib.collections.MatrixUtils;
import dk.alexandra.fresco.ml.utils.LinearAlgebraUtils;
import dk.alexandra.fresco.ml.utils.ModelLoader;
import java.io.File;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.commons.math3.linear.RealVector;
import org.junit.Assert;

public class NNTests {

  private static RealNumericProvider defaultRealNumeric = scope -> new BinaryFloatNumeric(scope);
  private static int defaultPrecision = 4;

  public static class TestNN1layer<ResourcePoolT extends ResourcePool>
      extends TestThreadFactory<ResourcePoolT, ProtocolBuilderNumeric> {

    @Override
    public TestThread<ResourcePoolT, ProtocolBuilderNumeric> next() {
      return new TestThread<ResourcePoolT, ProtocolBuilderNumeric>() {
        final int tests = 3;

        @Override
        public void test() throws Exception {
          ModelLoader loader = new ModelLoader(defaultRealNumeric, defaultPrecision);

          List<FullyConnectedLayerParameters<BigDecimal>> layers = Arrays.asList(
              loader.fullyConnectedLayerFromCsv(
                  new File(
                      getClass().getClassLoader().getResource("mnist/1-layer/0W.csv").getFile()),
                  new File(
                      getClass().getClassLoader().getResource("mnist/1-layer/0b.csv").getFile()),
                  ActivationFunctions.Type.RELU),
              loader.fullyConnectedLayerFromCsv(
                  new File(
                      getClass().getClassLoader().getResource("mnist/1-layer/1W.csv").getFile()),
                  new File(
                      getClass().getClassLoader().getResource("mnist/1-layer/1b.csv").getFile()),
                  ActivationFunctions.Type.SOFTMAX));

          // Test data
          Matrix<BigDecimal> testVectors = loader.matrixFromCsv(
              new File(getClass().getClassLoader().getResource("test.csv").getFile()));

          // Expected
          Stream<String> expectedLines = Files
              .lines(Paths.get(getClass().getClassLoader().getResource("labels.csv").getFile()));
          List<Integer> expected =
              expectedLines.map(s -> Integer.parseInt(s)).collect(Collectors.toList());
          expectedLines.close();

          LinearAlgebraUtils utils = new LinearAlgebraUtils();

          Application<List<Matrix<BigDecimal>>, ProtocolBuilderNumeric> testApplication = root -> {
            RealNumeric numeric = defaultRealNumeric.apply(root);

            List<DRes<Matrix<DRes<BigDecimal>>>> opened = new ArrayList<>();
            for (int i = 0; i < tests; i++) {
              DRes<Matrix<DRes<SReal>>> testVector =
                  numeric.linalg().input(utils.createColumnVector(testVectors.getRow(i)), 1);
              DRes<Matrix<DRes<SReal>>> out =
                  root.seq(new NeuralNetwork(layers, testVector, defaultRealNumeric));
              opened.add(numeric.linalg().open(out));
            }

            return () -> opened.stream().map(l -> new MatrixUtils().unwrapMatrix(l))
                .collect(Collectors.toList());
          };

          List<Matrix<BigDecimal>> output = runApplication(testApplication);
          for (int i = 0; i < output.size(); i++) {
            RealVector a = utils.convert(output.get(i)).getColumnVector(0);
            Assert.assertEquals(expected.get(i).intValue(), a.getMaxIndex());
          }
        }
      };
    }
  }

  public static class TestNN2layer<ResourcePoolT extends ResourcePool>
      extends TestThreadFactory<ResourcePoolT, ProtocolBuilderNumeric> {

    @Override
    public TestThread<ResourcePoolT, ProtocolBuilderNumeric> next() {
      return new TestThread<ResourcePoolT, ProtocolBuilderNumeric>() {
        final int tests = 3;

        @Override
        public void test() throws Exception {
          ModelLoader loader = new ModelLoader(defaultRealNumeric, defaultPrecision);

          List<FullyConnectedLayerParameters<BigDecimal>> layers = Arrays.asList(
              loader.fullyConnectedLayerFromCsv(
                  new File(
                      getClass().getClassLoader().getResource("mnist/2-layer/0W.csv").getFile()),
                  new File(
                      getClass().getClassLoader().getResource("mnist/2-layer/0b.csv").getFile()),
                  ActivationFunctions.Type.RELU),
              loader.fullyConnectedLayerFromCsv(
                  new File(
                      getClass().getClassLoader().getResource("mnist/2-layer/1W.csv").getFile()),
                  new File(
                      getClass().getClassLoader().getResource("mnist/2-layer/1b.csv").getFile()),
                  ActivationFunctions.Type.RELU),
              loader.fullyConnectedLayerFromCsv(
                  new File(
                      getClass().getClassLoader().getResource("mnist/2-layer/2W.csv").getFile()),
                  new File(
                      getClass().getClassLoader().getResource("mnist/2-layer/2b.csv").getFile()),
                  ActivationFunctions.Type.SOFTMAX));

          // Test data
          Matrix<BigDecimal> testVectors = loader.matrixFromCsv(
              new File(getClass().getClassLoader().getResource("test.csv").getFile()));

          // Expected
          Stream<String> expectedLines = Files
              .lines(Paths.get(getClass().getClassLoader().getResource("labels.csv").getFile()));
          List<Integer> expected =
              expectedLines.map(s -> Integer.parseInt(s)).collect(Collectors.toList());
          expectedLines.close();

          LinearAlgebraUtils utils = new LinearAlgebraUtils();

          Application<List<Matrix<BigDecimal>>, ProtocolBuilderNumeric> testApplication = root -> {
            RealNumeric numeric = defaultRealNumeric.apply(root);

            List<DRes<Matrix<DRes<BigDecimal>>>> opened = new ArrayList<>();
            for (int i = 0; i < tests; i++) {
              DRes<Matrix<DRes<SReal>>> testVector =
                  numeric.linalg().input(utils.createColumnVector(testVectors.getRow(i)), 1);
              DRes<Matrix<DRes<SReal>>> out =
                  root.seq(new NeuralNetwork(layers, testVector, defaultRealNumeric));
              opened.add(numeric.linalg().open(out));
            }

            return () -> opened.stream().map(l -> new MatrixUtils().unwrapMatrix(l))
                .collect(Collectors.toList());
          };

          List<Matrix<BigDecimal>> output = runApplication(testApplication);
          for (int i = 0; i < output.size(); i++) {
            RealVector a = utils.convert(output.get(i)).getColumnVector(0);
            Assert.assertEquals(expected.get(i).intValue(), a.getMaxIndex());
          }
        }
      };
    }
  }

  public static class TestFederatedLearning<ResourcePoolT extends ResourcePool>
      extends TestThreadFactory<ResourcePoolT, ProtocolBuilderNumeric> {

    @Override
    public TestThread<ResourcePoolT, ProtocolBuilderNumeric> next() {
      return new TestThread<ResourcePoolT, ProtocolBuilderNumeric>() {

        int tests = 2;

        @Override
        public void test() throws Exception {
          ModelLoader loader = new ModelLoader(defaultRealNumeric, defaultPrecision);

          Matrix<BigDecimal> b00 = loader.matrixFromCsv(
              new File(getClass().getClassLoader().getResource("mnist/2-layer/0b.csv").getFile()));
          Matrix<BigDecimal> b01 = loader.matrixFromCsv(new File(
              getClass().getClassLoader().getResource("mnist/2-layer/alt-model/0b.csv").getFile()));
          Matrix<BigDecimal> b10 = loader.matrixFromCsv(
              new File(getClass().getClassLoader().getResource("mnist/2-layer/1b.csv").getFile()));
          Matrix<BigDecimal> b11 = loader.matrixFromCsv(new File(
              getClass().getClassLoader().getResource("mnist/2-layer/alt-model/1b.csv").getFile()));
          Matrix<BigDecimal> b20 = loader.matrixFromCsv(
              new File(getClass().getClassLoader().getResource("mnist/2-layer/2b.csv").getFile()));
          Matrix<BigDecimal> b21 = loader.matrixFromCsv(new File(
              getClass().getClassLoader().getResource("mnist/2-layer/alt-model/2b.csv").getFile()));

          Matrix<BigDecimal> w00 = loader.matrixFromCsv(
              new File(getClass().getClassLoader().getResource("mnist/2-layer/0W.csv").getFile()));
          Matrix<BigDecimal> w01 = loader.matrixFromCsv(new File(
              getClass().getClassLoader().getResource("mnist/2-layer/alt-model/0W.csv").getFile()));
          Matrix<BigDecimal> w10 = loader.matrixFromCsv(
              new File(getClass().getClassLoader().getResource("mnist/2-layer/1W.csv").getFile()));
          Matrix<BigDecimal> w11 = loader.matrixFromCsv(new File(
              getClass().getClassLoader().getResource("mnist/2-layer/alt-model/1W.csv").getFile()));
          Matrix<BigDecimal> w20 = loader.matrixFromCsv(
              new File(getClass().getClassLoader().getResource("mnist/2-layer/2W.csv").getFile()));
          Matrix<BigDecimal> w21 = loader.matrixFromCsv(new File(
              getClass().getClassLoader().getResource("mnist/2-layer/alt-model/2W.csv").getFile()));

          // Test data
          Matrix<BigDecimal> testVectors = loader.matrixFromCsv(
              new File(getClass().getClassLoader().getResource("test.csv").getFile()));

          // Expected
          Stream<String> expectedLines = Files
              .lines(Paths.get(getClass().getClassLoader().getResource("labels.csv").getFile()));
          List<Integer> expected =
              expectedLines.map(s -> Integer.parseInt(s)).collect(Collectors.toList());
          expectedLines.close();

          LinearAlgebraUtils utils = new LinearAlgebraUtils();

          Application<List<Matrix<BigDecimal>>, ProtocolBuilderNumeric> testApplication =
              root -> root.seq(r1 -> {
                RealNumeric numeric = defaultRealNumeric.apply(r1);

                List<DRes<Matrix<DRes<BigDecimal>>>> opened = new ArrayList<>();

                DRes<Matrix<DRes<SReal>>> weights00 = numeric.linalg().input(w00, 1);
                DRes<Matrix<DRes<SReal>>> weights01 = numeric.linalg().input(w01, 2);

                DRes<Matrix<DRes<SReal>>> weights10 = numeric.linalg().input(w10, 1);
                DRes<Matrix<DRes<SReal>>> weights11 = numeric.linalg().input(w11, 2);

                DRes<Matrix<DRes<SReal>>> weights20 = numeric.linalg().input(w20, 1);
                DRes<Matrix<DRes<SReal>>> weights21 = numeric.linalg().input(w21, 2);

                DRes<Matrix<DRes<SReal>>> bias00 = numeric.linalg().input(b00, 1);
                DRes<Matrix<DRes<SReal>>> bias01 = numeric.linalg().input(b01, 2);

                DRes<Matrix<DRes<SReal>>> bias10 = numeric.linalg().input(b10, 1);
                DRes<Matrix<DRes<SReal>>> bias11 = numeric.linalg().input(b11, 2);

                DRes<Matrix<DRes<SReal>>> bias20 = numeric.linalg().input(b20, 1);
                DRes<Matrix<DRes<SReal>>> bias21 = numeric.linalg().input(b21, 2);

                // TODO: If there are enough matrices, we should do this via inner products instead
                DRes<Matrix<DRes<SReal>>> w0 =
                    numeric.linalg().add(numeric.linalg().scale(BigDecimal.valueOf(0.4), weights00),
                        numeric.linalg().scale(BigDecimal.valueOf(0.6), weights01));
                DRes<Matrix<DRes<SReal>>> w1 =
                    numeric.linalg().add(numeric.linalg().scale(BigDecimal.valueOf(0.4), weights10),
                        numeric.linalg().scale(BigDecimal.valueOf(0.6), weights11));
                DRes<Matrix<DRes<SReal>>> w2 =
                    numeric.linalg().add(numeric.linalg().scale(BigDecimal.valueOf(0.4), weights20),
                        numeric.linalg().scale(BigDecimal.valueOf(0.6), weights21));

                DRes<Matrix<DRes<SReal>>> b0 =
                    numeric.linalg().add(numeric.linalg().scale(BigDecimal.valueOf(0.4), bias00),
                        numeric.linalg().scale(BigDecimal.valueOf(0.6), bias01));
                DRes<Matrix<DRes<SReal>>> b1 =
                    numeric.linalg().add(numeric.linalg().scale(BigDecimal.valueOf(0.4), bias10),
                        numeric.linalg().scale(BigDecimal.valueOf(0.6), bias11));
                DRes<Matrix<DRes<SReal>>> b2 =
                    numeric.linalg().add(numeric.linalg().scale(BigDecimal.valueOf(0.4), bias20),
                        numeric.linalg().scale(BigDecimal.valueOf(0.6), bias21));

                opened.add(numeric.linalg().open(w0));
                opened.add(numeric.linalg().open(w1));
                opened.add(numeric.linalg().open(w2));
                opened.add(numeric.linalg().open(b0));
                opened.add(numeric.linalg().open(b1));
                opened.add(numeric.linalg().open(b2));

                return () -> opened.stream().map(x -> new MatrixUtils().unwrapMatrix(x))
                    .collect(Collectors.toList());
              }).seq((r2, l) -> {

                RealNumeric numeric = defaultRealNumeric.apply(r2);

                List<DRes<Matrix<DRes<BigDecimal>>>> opened = new ArrayList<>();

                for (int i = 0; i < tests; i++) {
                  DRes<Matrix<DRes<SReal>>> testVector =
                      numeric.linalg().input(utils.createColumnVector(testVectors.getRow(i)), 1);

                  DRes<Matrix<DRes<SReal>>> out =
                      r2.seq(
                          new NeuralNetwork(
                              Arrays.asList(
                                  new FullyConnectedLayerParameters<>(l.get(0), l.get(3),
                                      ActivationFunctions.Type.RELU),
                                  new FullyConnectedLayerParameters<>(l.get(1), l.get(4),
                                      ActivationFunctions.Type.RELU),
                                  new FullyConnectedLayerParameters<>(l.get(2), l.get(5),
                                      ActivationFunctions.Type.SOFTMAX)),
                              testVector, defaultRealNumeric));

                  opened.add(numeric.linalg().open(out));
                }

                return () -> opened.stream().map(x -> new MatrixUtils().unwrapMatrix(x))
                    .collect(Collectors.toList());
              });

          List<Matrix<BigDecimal>> output = runApplication(testApplication);
          for (int i = 0; i < output.size(); i++) {
            RealVector a = utils.convert(output.get(i)).getColumnVector(0);
            Assert.assertEquals(expected.get(i).intValue(), a.getMaxIndex());
          }
        }
      };
    }
  }

}
