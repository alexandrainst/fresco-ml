package dk.alexandra.fresco.ml.dtrees;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import org.junit.Before;
import org.junit.Test;

import dk.alexandra.fresco.ml.dtrees.utils.DTreeParser;

public class TestDTreeParser {
  private static List<String> testFile;
  private DTreeParser parser;
  private List<String> stringCategories;
  private List<String> stringFeatures;
  private int intDepth;

  @SuppressWarnings({ "unchecked" })
  @Before
  public void setup() throws Exception {
    testFile = new ArrayList<>();
    testFile.add(
        "\"radius_1\",\"texture_1\",\"perimeter_1\",\"area_1\",\"smoothness_1\",\"compactness_1\",\"concavity_1\",\"concave_points_1\",\"symmetry_1\",\"fractal_dimension_1\",\"radius_2\",\"texture_2\",\"perimeter_2\",\"area_2\",\"smoothness_2\",\"compactness_2\",\"concavity_2\",\"concave_points_2\",\"symmetry_2\",\"fractal_dimension_2\",\"radius_3\",\"texture_3\",\"perimeter_3\",\"area_3\",\"smoothness_3\",\"compactness_3\",\"concavity_3\",\"concave_points_3\",\"symmetry_3\",\"fractal_dimension_3\"");
    testFile.add("n= 568");
    testFile.add("node), split, n, loss, yval, (yprob)");
    testFile.add("      * denotes terminal node");
    testFile.add(" 1) root 568 211 B (0.62852113 0.37147887)");
    testFile.add("   2) radius_3< 16.795 379  33 B (0.91292876 0.08707124)");
    testFile.add("     4) concave_points_3< 0.1358 333   5 B (0.98498498 0.01501502) *");
    testFile.add("     5) concave_points_3>=0.1358 46  18 M (0.39130435 0.60869565)");
    testFile.add("      10) texture_3>=25.67 19   4 B (0.78947368 0.21052632) *");
    testFile.add("      11) texture_3< 25.67 27   3 M (0.11111111 0.88888889) *");
    testFile.add("   3) radius_3>=16.795 189  11 M (0.05820106 0.94179894) *");
    parser = new DTreeParser(10); // Weights will be multiplied by 10 and rounded to integer

    // Set the internal structures
    Method setFeatures = DTreeParser.class.getDeclaredMethod("setFeatures", List.class);
    setFeatures.setAccessible(true);
    setFeatures.invoke(parser, testFile);
    Field features = DTreeParser.class.getDeclaredField("features");
    features.setAccessible(true);
    stringFeatures = (List<String>) features.get(parser);
    Method setCategories = DTreeParser.class.getDeclaredMethod("setCategories", Stream.class);
    setCategories.setAccessible(true);
    setCategories.invoke(parser, testFile.stream());
    Field categories = DTreeParser.class.getDeclaredField("categories");
    categories.setAccessible(true);
    stringCategories = (List<String>) categories.get(parser);
    Method setDepth = DTreeParser.class.getDeclaredMethod("setDepth", List.class);
    setDepth.setAccessible(true);
    setDepth.invoke(parser, testFile);
    Field depth = DTreeParser.class.getDeclaredField("depth");
    depth.setAccessible(true);
    intDepth = (int) depth.get(parser);
  }

  private DecisionTreeModel makeTree() throws Exception {
    Method construct = DTreeParser.class.getDeclaredMethod("constructTree", List.class);
    construct.setAccessible(true);
    construct.invoke(parser, testFile);
    Method make = DTreeParser.class.getDeclaredMethod("makeTreeModel");
    make.setAccessible(true);
    return (DecisionTreeModel) make.invoke(parser);
  }

  @Test
  public void testDepth() throws Exception {
    assertEquals(4, intDepth);
  }

  @Test
  public void testNumFeatures() {
    assertEquals(30, stringFeatures.size());
  }

  @Test
  public void testFindLine() throws Exception {
    Method method = DTreeParser.class.getDeclaredMethod("findLine", int.class, List.class);
    method.setAccessible(true);
    assertEquals("   3) radius_3>=16.795 189  11 M (0.05820106 0.94179894) *", method.invoke(
        parser, 3, testFile));
    assertEquals(" 1) root 568 211 B (0.62852113 0.37147887)", method.invoke(parser, 1, testFile));
    assertNull(method.invoke(parser, 8, testFile));
  }

  @Test
  public void testFindNode() throws Exception {
    Method method = DTreeParser.class.getDeclaredMethod("findNode", int.class, List.class);
    method.setAccessible(true);
    Object res = method.invoke(parser, 2, testFile);
    Field feature = res.getClass().getDeclaredField("feature");
    feature.setAccessible(true);
    assertEquals("radius_3", stringFeatures.get((int) feature.get(res)));
    Field weight = res.getClass().getDeclaredField("weight");
    weight.setAccessible(true);
    assertEquals(16.795, (double) weight.get(res), 0.01);
    Field category = res.getClass().getDeclaredField("category");
    category.setAccessible(true);
    assertEquals("B", stringCategories.get((int) category.get(res)));
    Field switchAround = res.getClass().getDeclaredField("switchAround");
    switchAround.setAccessible(true);
    assertEquals(false, switchAround.get(res));

    res = method.invoke(parser, 11, testFile);
    feature = res.getClass().getDeclaredField("feature");
    feature.setAccessible(true);
    assertEquals("texture_3", stringFeatures.get((int) feature.get(res)));
    weight = res.getClass().getDeclaredField("weight");
    weight.setAccessible(true);
    assertEquals(25.67, (double) weight.get(res), 0.01);
    category = res.getClass().getDeclaredField("category");
    category.setAccessible(true);
    assertEquals("M", stringCategories.get((int) category.get(res)));
    switchAround = res.getClass().getDeclaredField("switchAround");
    switchAround.setAccessible(true);
    assertEquals(true, switchAround.get(res));
  }

  @Test
  public void testMakeTreeModel() throws Exception {
    DecisionTreeModel model = makeTree();
    // 0 means category "B" and 1 category "M"
    // Index 2 is node 10 since we are on the fourth layer (10 - 2^3)
    assertEquals(BigInteger.ONE, model.getCategories().get(2));
    // Index 3 is node 11 since we are on the fourth layer (11 - 2^3)
    assertEquals(BigInteger.ZERO, model.getCategories().get(3));
  }

}
