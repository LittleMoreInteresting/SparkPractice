package cn.datafountain.windpower.ml

import cn.datafountain.windpower.common.Context
import org.apache.spark.ml.{Pipeline, linalg}
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame

object WindPowerApp extends App with Context{

  val trainData: DataFrame = loadData()
  val featuresArray = Array("WindSpeed","Power","RotorSpeed")
  val vecDF = new VectorAssembler().setInputCols(featuresArray).setOutputCol("features")
  val k = 48
  //规范化
  val scalaDF = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")
    .setWithStd(true).setWithMean(false)
  //Pipeline 组装
  private val pipeline: Pipeline = new Pipeline().setStages(Array(vecDF, scalaDF))
  private val data2: DataFrame = pipeline.fit(trainData).transform(trainData)
  val biKMeans = new BisectingKMeans().setFeaturesCol("scaledFeatures").setK(k).setSeed(123456789)
  val model = biKMeans.fit(data2)
  val clusterSizes: Array[Long] = model.summary.clusterSizes
  val centers: Array[linalg.Vector] = model.clusterCenters
  var maxSize: Long = 0L;
  var maxSizeCluster = 0;
  for (i <- clusterSizes.indices
       if clusterSizes(i)>maxSize
       ) {
    maxSize = clusterSizes(i)
    maxSizeCluster = i
  }
  var clusterDistFactor = new Array[Double](k)
  for (i <- clusterSizes.indices
      if i != maxSizeCluster
       ) {
    val sqDist = Vectors.sqdist(centers(maxSizeCluster),centers(i))
    clusterDistFactor(i) = clusterSizes(i)/sqDist

    println(s"i=$i;size="+clusterSizes(i)+";dist:"+sqDist+";Factor:"+clusterDistFactor(i))
  }
  val avgFactor = clusterDistFactor.sum/clusterDistFactor.length;
  println("maxSizeCluster="+avgFactor)

}
