package dk.alexandra.fresco.ml.fl.demo;

import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;

public class MyChart {

  private double[] x;
  private double[] y;
  private XYChart chart;
  private SwingWrapper<XYChart> sw;
  private String name;
  
  public MyChart(String name) {
    this.x = new double[0];
    this.y= new double[0];
    this.name = name;
   }
  
  public void addPoint(double x, double y) {
    
    double[] newX = new double[this.x.length + 1];
    newX[this.x.length] = x;
    System.arraycopy(this.x, 0, newX, 0, this.x.length);
    this.x = newX;
    
    double[] newY = new double[this.y.length + 1];
    newY[this.y.length] = y;
    System.arraycopy(this.y, 0, newY, 0, this.y.length);
    this.y = newY;

    if (this.chart == null) {
      this.chart = QuickChart.getChart(this.name, "Epochs", "Accuracy", "accuracy", this.x, this.y);
      this.chart.getStyler().setXAxisMin(1.0);
      this.chart.getStyler().setXAxisMax(50.0);
      this.chart.getStyler().setYAxisMin(0.7);
      this.chart.getStyler().setYAxisMax(1.0);
      this.sw = new SwingWrapper<>(chart);
      sw.displayChart();
    } else {
      chart.updateXYSeries("accuracy", this.x, this.y, null);
      sw.repaintChart();
    }
  }
  
}
