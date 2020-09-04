package cn.datafountain.windpower.ml

import java.util

import breeze.numerics.log10
import cn.datafountain.windpower.common.{Context, WindResult, WindResultOut}
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{Pipeline, linalg}
import org.apache.spark.sql.{DataFrame, RelationalGroupedDataset, Row}

import scala.collection.mutable
import scala.collection.mutable.Set

object WindPowerApp extends App with Context{

  val trainData: DataFrame = loadData()
  val rows = trainData.select("WindNumber")
    .distinct().orderBy("WindNumber").collect()
  rows.foreach(x=> {
    val n = x(0)
    val value = trainData.filter(s"WindNumber=$n")
    value.show(1)
  })

  sys.exit()
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
    clusterDistFactor(i) = log10(clusterSizes(i)/sqDist)

    println(s"i=$i;size="+clusterSizes(i)+";dist:"+sqDist+";Factor:"+clusterDistFactor(i))
  }
  val avgFactor = clusterDistFactor.sum/clusterDistFactor.length;
  println("maxSizeCluster="+avgFactor)
  val errorCluster:mutable.Set[Int] = mutable.Set()
  for (cluster <- clusterDistFactor.indices
       if cluster != maxSizeCluster
       if clusterDistFactor(cluster)< avgFactor
       ) {
    errorCluster.add(cluster)
    println(s"Error Cluster：$cluster ;Factor:"+clusterDistFactor(cluster))
  }
  val resultDF: DataFrame = model.transform(data2)
  import sparkSession.implicits._
  resultDF.select("WindNumber","Time",model.getPredictionCol)
    .as[WindResult]
    .map( result => {
      val label = if (errorCluster.contains(result.prediction)) 1 else 0;
      WindResultOut(result.WindNumber,result.Time,label)
    })
    .write
    .mode("overwrite")
    .csv("result.csv")

}
