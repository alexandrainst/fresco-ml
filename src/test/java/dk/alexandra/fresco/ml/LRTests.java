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
                                row.stream().map(i -> seq.realNumeric().known(BigDecimal.valueOf(i))).collect(Collectors.toList());

                        List<DRes<SReal>> secretB =
                                b.stream().map(i -> seq.realNumeric().known(BigDecimal.valueOf(i))).collect(Collectors.toList());

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
                                        i -> data.get(i).stream().map(BigDecimal::valueOf).map(seq.realNumeric()::known)
                                                .collect(Collectors.toCollection(ArrayList::new)));

                                List<DRes<SReal>> secretE =
                                        e.stream().map(BigDecimal::valueOf).map(seq.realNumeric()::known).collect(Collectors.toList());

                                List<DRes<SReal>> initB = Collections.nCopies(m, seq.realNumeric().known(BigDecimal.ZERO));

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
}
