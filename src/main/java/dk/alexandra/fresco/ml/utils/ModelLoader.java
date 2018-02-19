package dk.alexandra.fresco.ml.utils;

import dk.alexandra.fresco.decimal.RealNumericProvider;
import dk.alexandra.fresco.lib.collections.Matrix;
import dk.alexandra.fresco.ml.ActivationFunctions;
import dk.alexandra.fresco.ml.FullyConnectedLayerParameters;
import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.Charset;
import java.util.ArrayList;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

public class ModelLoader {

  private final int precision;

  public ModelLoader(RealNumericProvider provider, int precision) {
    this.precision = precision;
  }

  public Matrix<BigDecimal> matrixFromCsv(File file) throws IOException {
    ArrayList<ArrayList<BigDecimal>> rows = new ArrayList<>();
    CSVParser parser = CSVParser.parse(file, Charset.defaultCharset(), CSVFormat.EXCEL);
    for (CSVRecord record : parser) {
      ArrayList<BigDecimal> row = new ArrayList<>();
      for (String entry : record) {
        row.add(new BigDecimal(entry).setScale(precision, RoundingMode.DOWN));
      }
      rows.add(row);
    }
    return new Matrix<>(rows.size(), rows.get(0).size(), rows);
  }

  public FullyConnectedLayerParameters<BigDecimal> fullyConnectedLayerFromCsv(File weights, File bias,
      ActivationFunctions.Type activationFunction) throws IOException {
    Matrix<BigDecimal> weightsMatrix = matrixFromCsv(weights);
    Matrix<BigDecimal> biasVector = matrixFromCsv(bias);
    return new FullyConnectedLayerParameters<>(weightsMatrix, biasVector, activationFunction);
  }

}
