package cn.datafountain.windpower.ml

import cn.datafountain.windpower.common.Context
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, VectorAssembler}
import org.apache.spark.sql.DataFrame

object WindPowerApp extends App with Context{
  private val trainData: DataFrame = loadData()
  private val params: DataFrame = getParams()
  private val frame: DataFrame = trainData.join(params, "WindNumber")
  trainData.show(3)
  frame.show(3)
  //frame.printSchema()
  //sys.exit(0)
//"WindNumber","Time","WindSpeed","Power","RotorSpeed","Month","Hour","wd","p","in","out","minSp","maxSp"
  //类别型字段转换
  private val encoder1: OneHotEncoder = new OneHotEncoder().setInputCol("WindNumber").setOutputCol("WindNumberVector").setDropLast(false)
  //private val encoder2: OneHotEncoder = new OneHotEncoder().setInputCol("Month").setOutputCol("MonthVector").setDropLast(false)
  //private val encoder3: OneHotEncoder = new OneHotEncoder().setInputCol("Hour").setOutputCol("HourVector").setDropLast(false)

  //组合特征向量
  val featuresArray = Array("WindNumberVector","Timestamp","WindSpeed","Power","RotorSpeed","wd","p","in","out","minSp","maxSp")
  val vecDF = new VectorAssembler().setInputCols(featuresArray).setOutputCol("features")

  //规范化
  val scalaDF = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")
    .setWithStd(true).setWithMean(false)
  //Pipeline 组装
  var kMeans = new KMeans().setFeaturesCol("scaledFeatures").setK(4).setSeed(System.currentTimeMillis())

  private val pipeline: Pipeline = new Pipeline().setStages(Array(encoder1, vecDF, scalaDF))
  private val data2: DataFrame = pipeline.fit(frame).transform(frame)
  val model = kMeans.fit(data2)
  val results = model.transform(data2)

  //评估模型
  private val WSSSE: Double = model.computeCost(data2)
  println(s"Within  Set  Sum  of  Squared  Errors=  $WSSSE")

  //结果展示
  println (" Cluster  Centers :")
  model.clusterCenters.foreach(println)
  results.write.mode("overwrite").csv("result.csv")
}
