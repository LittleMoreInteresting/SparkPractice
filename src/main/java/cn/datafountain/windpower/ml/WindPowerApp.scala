package cn.datafountain.windpower.ml

import cn.datafountain.windpower.common.{Context, DataLoad}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, VectorAssembler}
import org.apache.spark.sql.DataFrame

object WindPowerApp extends App with Context{

  val frame: DataFrame = new DataLoad().getTrainingData(sparkSession)
  frame.printSchema()
  sys.exit(9)
  //类别型字段转换
  private val encoder1: OneHotEncoder = new OneHotEncoder()
    .setInputCol("spType")
    .setOutputCol("spTypeVector")
    .setDropLast(false)
  //组合特征向量
  val featuresArray = Array("WindSpeed","Power","RotorSpeed","spType")
  val vecDF = new VectorAssembler().setInputCols(featuresArray).setOutputCol("features")

  //规范化
  val scalaDF = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")
    .setWithStd(true).setWithMean(false)
  //Pipeline 组装
  var kMeans = new KMeans().setFeaturesCol("scaledFeatures").setK(48).setSeed(123456789)
  private val pipeline: Pipeline = new Pipeline().setStages(Array(encoder1,vecDF, scalaDF))
  private val data2: DataFrame = pipeline.fit(frame).transform(frame)
  val model = kMeans.fit(data2)
  val results = model.transform(data2)

  //评估模型
  private val WSSSE: Double = model.computeCost(data2)
  println(s"Within  Set  Sum  of  Squared  Errors=  $WSSSE")

  //结果展示
  println (" Cluster  Centers :")
  model.clusterCenters.foreach(println)
  //results.printSchema()
  results.show(3)
  results.select("WindNumber","prediction").groupBy("prediction").count().show(50)
}
