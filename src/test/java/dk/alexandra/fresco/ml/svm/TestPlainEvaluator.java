package dk.alexandra.fresco.ml.svm;

import static org.junit.Assert.assertEquals;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

public class TestPlainEvaluator {

  @Test
  public void testInnerProduct() throws NoSuchMethodException, SecurityException,
      IllegalAccessException, IllegalArgumentException, InvocationTargetException {
    List<BigInteger> first = new ArrayList<>();
    List<BigInteger> second = new ArrayList<>();
    first.add(new BigInteger("42"));
    first.add(new BigInteger("1337"));
    first.add(new BigInteger("0"));
    second.add(new BigInteger("43"));
    second.add(new BigInteger("0"));
    second.add(new BigInteger("1337"));

    PlainEvaluator evaluator = new PlainEvaluator(null);
    Method innerProduct = PlainEvaluator.class.getDeclaredMethod("innerProduct", List.class, List.class);
    innerProduct.setAccessible(true);
    BigInteger returnValue = (BigInteger) innerProduct.invoke(evaluator, first, second);
    assertEquals(new BigInteger("1806"), returnValue);
  }

  @Test
  public void testArgMax() throws NoSuchMethodException, SecurityException, IllegalAccessException,
      IllegalArgumentException, InvocationTargetException {
    List<BigInteger> list = new ArrayList<>();
    list.add(new BigInteger("42"));
    list.add(new BigInteger("1337"));
    list.add(new BigInteger("0"));

    PlainEvaluator evaluator = new PlainEvaluator(null);
    Method argMax = PlainEvaluator.class.getDeclaredMethod("argMax", List.class);
    argMax.setAccessible(true);
    BigInteger returnValue = (BigInteger) argMax.invoke(evaluator, list);
    assertEquals(new BigInteger("1337"), returnValue);
  }

}
