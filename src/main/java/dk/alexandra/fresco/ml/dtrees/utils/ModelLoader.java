package dk.alexandra.fresco.ml.dtrees.utils;

import dk.alexandra.fresco.ml.dtrees.DecisionTreeModel;
import java.io.File;
import java.io.IOException;
import java.math.BigInteger;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

public class ModelLoader {

  public DecisionTreeModel modelFromFile(File modelFile) throws IOException {
    List<List<BigInteger>> featureIndexes = new ArrayList<>();
    List<List<BigInteger>> weights = new ArrayList<>();
    List<BigInteger> categories = new ArrayList<>();
    List<CSVRecord> records = CSVParser
        .parse(modelFile, Charset.defaultCharset(), CSVFormat.EXCEL)
        .getRecords();
    for (int i = 0; i < records.size() - 1; i++) {
      boolean readingFeatureIndex = true;
      List<BigInteger> currentFeatureIndexes = new ArrayList<>();
      List<BigInteger> currentWeights = new ArrayList<>();
      CSVRecord record = records.get(i);
      for (String entry : record) {
        if (readingFeatureIndex) {
          currentFeatureIndexes.add(new BigInteger(entry));
        } else {
          currentWeights.add(new BigInteger(entry));
        }
        readingFeatureIndex = !readingFeatureIndex;
      }
      if (currentFeatureIndexes.size() != currentWeights.size()) {
        throw new IllegalStateException("Must have same number of indexes and weights.");
      }
      if (currentFeatureIndexes.size() != (1 << i)) {
        throw new IllegalStateException(
            "Tree must be complete. Layer " + i + " had " + featureIndexes.size() + " nodes.");
      }
      featureIndexes.add(currentFeatureIndexes);
      weights.add(currentWeights);
    }
    CSVRecord categoriesRecord = records.get(records.size() - 1);
    for (String entry : categoriesRecord) {
      categories.add(new BigInteger(entry));
    }
    int depth = records.size();
    if (categories.size() != (1 << (depth - 1))) {
      throw new IllegalStateException(
          "Tree must be complete. Depth is " + depth + " but tree only has " + categories.size()
              + " leaves.");
    }
    return new DecisionTreeModel(featureIndexes, weights, categories);
  }

  public File getFile(String path) {
    return new File(getClass().getClassLoader().getResource(path).getFile());
  }

}
