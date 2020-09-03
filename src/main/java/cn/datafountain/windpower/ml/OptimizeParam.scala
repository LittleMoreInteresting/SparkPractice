package cn.datafountain.windpower.ml

import cn.datafountain.windpower.common.Context
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{BisectingKMeans, BisectingKMeansModel, BisectingKMeansSummary, KMeans}
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
case class WindParams(
                       WindNumber:Int,
                       Time: String,
                       WindSpeed: Double,
                       Power: Double,
                       RotorSpeed: Double
                     )
case class WindParams2(
                        WindNumber:Int,
                        Time: String,
                        WindSpeed: Double,
                        Power: Double,
                        RotorSpeed: Double,
                        spType: Int
                      )
object OptimizeParam extends App with  Context{

  val trainData: DataFrame = loadData()


  //类别型字段转换
  //private val encoder1: OneHotEncoder = new OneHotEncoder().setInputCol("spType").setOutputCol("spTypeVector").setDropLast(false)

  //组合特征向量
  val featuresArray = Array("WindSpeed","Power","RotorSpeed")
  val vecDF = new VectorAssembler().setInputCols(featuresArray).setOutputCol("features")

  //规范化
  val scalaDF = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")
    .setWithStd(true).setWithMean(false)
  //Pipeline 组装
  //var kMeans = new KMeans().setFeaturesCol("scaledFeatures").setK(70).setSeed(123456789)

  private val pipeline: Pipeline = new Pipeline().setStages(Array(vecDF, scalaDF))
  private val data2: DataFrame = pipeline.fit(trainData).transform(trainData)
  /*val kMeans = new BisectingKMeans().setFeaturesCol("scaledFeatures").setK(48).setSeed(123456789)
  val model = kMeans.fit(data2)
  val WSSSE: Double = model.computeCost(data2)
  println(s"Within  Set  Sum  of  Squared  Errors=  $WSSSE")*/

  val KSSE = (30 to 60 by 2).toList.map{
    k =>
      val biKMeans = new BisectingKMeans().setFeaturesCol("scaledFeatures").setK(k).setSeed(123456789)
      val model = biKMeans.fit(data2)
      val WSSSE: Double = model.computeCost(data2)
      /*val centers = model.clusterCenters
      val d = Vectors.sqdist(centers(0), centers(1))
      println()
      println(d)*/
      (k,WSSSE)
  }
  KSSE.sortBy(_._1).foreach(println)

  // K select 48


  // TODO 1 Top size  cluster 最大簇 中心
  // TODO 2 sqdist/size 每个簇到最大簇中心的距离平方比本簇大小 -> 系数平均值 avg
  // TODO 2 系数< avg*(0.4) 标记为 1 异常数
}
