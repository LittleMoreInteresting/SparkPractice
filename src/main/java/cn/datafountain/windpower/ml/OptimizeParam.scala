package cn.datafountain.windpower.ml

import cn.datafountain.windpower.common.Context
import cn.datafountain.windpower.ml.WindPowerApp.{getParams, loadData}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, VectorAssembler}
import org.apache.spark.sql.DataFrame

object OptimizeParam extends App with Context{
  private val trainData: DataFrame = loadData()
  private val params: DataFrame = getParams()
  private val frame: DataFrame = trainData.join(params, "WindNumber")
  //类别型字段转换
  //private val encoder1: OneHotEncoder = new OneHotEncoder().setInputCol("Year").setOutputCol("YearVector").setDropLast(false)
  //private val encoder2: OneHotEncoder = new OneHotEncoder().setInputCol("Month").setOutputCol("MonthVector").setDropLast(false)

  //组合特征向量
  val featuresArray = Array("WindSpeed","Power","RotorSpeed","wd","in","out","minSp","maxSp")
  val vecDF = new VectorAssembler().setInputCols(featuresArray).setOutputCol("features")

  //规范化
  val scalaDF = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")
    .setWithStd(true).setWithMean(false)
  //Pipeline 组装
  var kMeans = new KMeans().setFeaturesCol("scaledFeatures").setK(28).setSeed(123456789)

  private val pipeline: Pipeline = new Pipeline().setStages(Array(vecDF, scalaDF))
  private val data2: DataFrame = pipeline.fit(frame).transform(frame)
  val model = kMeans.fit(data2)
  val results = model.transform(data2)

  //评估模型
  private val WSSSE: Double = model.computeCost(data2)
  println(s"Within  Set  Sum  of  Squared  Errors=  $WSSSE")

  //结果展示
  //println (" Cluster  Centers :")
  //model.clusterCenters.foreach(println)
  //results.printSchema()
  results.show(3)
  results.select("WindNumber","prediction").groupBy("prediction").count().show(50)
}
