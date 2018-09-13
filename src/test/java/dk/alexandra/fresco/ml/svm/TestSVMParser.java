package dk.alexandra.fresco.ml.svm;

import static org.junit.Assert.assertEquals;

import java.math.BigInteger;

import org.junit.Before;
import org.junit.Test;

import dk.alexandra.fresco.ml.svm.utils.SVMParser;

public class TestSVMParser {

  SVMModel model;

  @Before
  public void setup() throws Exception {
    SVMParser parser = new SVMParser(1000000);
    String filename = getClass().getClassLoader().getResource("svms/models/cifarTest.csv")
        .getFile();
    model = parser.parseFile(filename);
  }

  @Test
  public void testSizes() {
    assertEquals(128, model.getNumFeatures());
    assertEquals(4, model.getNumSupportVectors());
  }

  @Test
  public void testBias() {
    assertEquals(new BigInteger("-5462631"), model.getBias().get(0));
    assertEquals(new BigInteger("1034298"), model.getBias().get(3));
  }

  @Test
  public void testSupportvectors() {
    assertEquals(new BigInteger("-1496643"), model.getSupportVectors().get(0).get(0));
    assertEquals(new BigInteger("2133708"), model.getSupportVectors().get(3).get(127));
  }
}
