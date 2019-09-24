package dk.alexandra.fresco.ml;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import org.junit.Assert;

import dk.alexandra.fresco.framework.Application;
import dk.alexandra.fresco.framework.DRes;
import dk.alexandra.fresco.framework.TestThreadRunner.TestThread;
import dk.alexandra.fresco.framework.TestThreadRunner.TestThreadFactory;
import dk.alexandra.fresco.framework.builder.numeric.ProtocolBuilderNumeric;
import dk.alexandra.fresco.framework.sce.resources.ResourcePool;
import dk.alexandra.fresco.lib.collections.Matrix;
import dk.alexandra.fresco.lib.real.SReal;
import dk.alexandra.fresco.ml.lr.LogisticRegression;
import dk.alexandra.fresco.ml.lr.LogisticRegressionPrediction;
import dk.alexandra.fresco.ml.lr.LogisticRegressionSGD;

public class LRTests {

  public static class TestLogRegPrediction<ResourcePoolT extends ResourcePool>
      extends TestThreadFactory<ResourcePoolT, ProtocolBuilderNumeric> {

    @Override
    public TestThread<ResourcePoolT, ProtocolBuilderNumeric> next() {
      return new TestThread<ResourcePoolT, ProtocolBuilderNumeric>() {

        @Override
        public void test() throws Exception {

          List<Double> row = Arrays.asList(1.5, 2.5);

          List<Double> b = Arrays.asList(0.1, 0.2, 0.3);

          Application<BigDecimal, ProtocolBuilderNumeric> testApplication = seq -> {

            List<DRes<SReal>> secretRow =
                row.stream().map(i -> seq.realNumeric().known(i)).collect(Collectors.toList());

            List<DRes<SReal>> secretB =
                b.stream().map(i -> seq.realNumeric().known(i)).collect(Collectors.toList());
            
            DRes<SReal> y =
                new LogisticRegressionPrediction(secretRow, secretB).buildComputation(seq);
            
            return seq.realNumeric().open(y);
          };
          double expected = .759510916949111;
          BigDecimal output = runApplication(testApplication);
          Assert.assertTrue(Math.abs(output.doubleValue() - expected) < 0.001);
        }
      };
    }
  }

  public static class TestLogRegSGDSingleEpoch<ResourcePoolT extends ResourcePool>
      extends TestThreadFactory<ResourcePoolT, ProtocolBuilderNumeric> {

    @Override
    public TestThread<ResourcePoolT, ProtocolBuilderNumeric> next() {
      return new TestThread<ResourcePoolT, ProtocolBuilderNumeric>() {

        @Override
        public void test() throws Exception {

          List<List<Double>> data = Arrays.asList(Arrays.asList(1.5, 2.5), Arrays.asList(2.1, 3.1),
              Arrays.asList(3.2, 4.2));
          List<Double> e = Arrays.asList(0.0, 0.0, 1.0);
          int n = data.size();
          int m = e.size();

          Application<List<BigDecimal>, ProtocolBuilderNumeric> testApplication =
              root -> root.seq(seq -> {

                Matrix<DRes<SReal>> secretData = new Matrix<DRes<SReal>>(n, m - 1,
                    i -> data.get(i).stream().map(seq.realNumeric()::known)
                        .collect(Collectors.toCollection(ArrayList::new)));

                List<DRes<SReal>> secretE =
                    e.stream().map(seq.realNumeric()::known).collect(Collectors.toList());

                List<DRes<SReal>> initB = Collections.nCopies(m, seq.realNumeric().known(0.0));

                DRes<List<DRes<SReal>>> b =
                    new LogisticRegressionSGD(secretData, secretE, 0.1, initB)
                        .buildComputation(seq);
                return b;
              }).seq((seq, b) -> {

                List<DRes<BigDecimal>> openB =
                    b.stream().map(bi -> seq.realNumeric().open(bi)).collect(Collectors.toList());

                return () -> openB.stream().map(DRes::out).collect(Collectors.toList());
              });

          List<BigDecimal> output = runApplication(testApplication);
          List<Double> expected =
              Arrays.asList(-0.00950850635382685, 0.0034818502674105398, -0.00602665608641631);

          for (int i = 0; i < output.size(); i++) {
            Assert.assertTrue(Math.abs(output.get(i).doubleValue() - expected.get(i)) < 0.001);
          }


        }
      };
    }
  }

  public static class TestLogReg<ResourcePoolT extends ResourcePool>
      extends TestThreadFactory<ResourcePoolT, ProtocolBuilderNumeric> {

    @Override
    public TestThread<ResourcePoolT, ProtocolBuilderNumeric> next() {
      return new TestThread<ResourcePoolT, ProtocolBuilderNumeric>() {

        @Override
        public void test() throws Exception {

          List<double[]> xColumns = Arrays.asList(
              new double[] {2.7810836, 1.465489372, 3.396561688, 1.38807019, 3.06407232,
                  7.627531214, 5.332441248, 6.922596716, 8.675418651, 7.673756466},
              new double[] {2.550537003, 2.362125076, 4.400293529, 1.850220317, 3.005305973,
                  2.759262235, 2.088626775, 1.77106367, -0.242068655, 3.508563011});
          int[] y = new int[] {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

          int n = y.length;
          Application<List<BigDecimal>, ProtocolBuilderNumeric> testApplication =
              root -> root.seq(seq -> {

                // Transpose
                Matrix<DRes<SReal>> d = new Matrix<DRes<SReal>>(n, xColumns.size(), i -> {
                  return xColumns.stream().mapToDouble(c -> c[i])
                      .mapToObj(s -> seq.realNumeric().known(s))
                      .collect(Collectors.toCollection(ArrayList::new));
                });

                List<DRes<SReal>> e = Arrays.stream(y).mapToObj(x -> seq.realNumeric().known(x))
                    .collect(Collectors.toList());

                DRes<List<DRes<SReal>>> b =
                    new LogisticRegression(d, e, 0.3, 20).buildComputation(seq);
                return b;
              }).seq((seq, b) -> {

                List<DRes<BigDecimal>> openB =
                    b.stream().map(bi -> seq.realNumeric().open(bi)).collect(Collectors.toList());
                return () -> openB.stream().map(DRes::out).collect(Collectors.toList());
              });


          List<BigDecimal> output = runApplication(testApplication);
          List<Double> expected =
              Arrays.asList(-0.5482258628208326, 1.0470718511711763, -1.4577730818094987);
          for (int i = 0; i < output.size(); i++) {
            Assert.assertTrue(Math.abs(output.get(i).doubleValue() - expected.get(i)) < 0.001);
          }
        }
      };
    }
  }
}
