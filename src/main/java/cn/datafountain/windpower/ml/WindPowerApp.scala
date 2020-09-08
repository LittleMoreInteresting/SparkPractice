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

/**
 * K -> 48
 * 选择size 最大分区 中心坐标 center0
 * 其他分区 根据 该分区中心到 center0 的距离平方d1 和分区大小s1 构建系数 f1 = log10(s1/d1)
 * 根据平均系数 avg 与 f1 之间的关系进行 正常、异常数据区分。比如 f1>avg*1.2 ->正常
 * 最高 0.7
 */
object WindPowerApp extends App with Context{

  val trainData: DataFrame = loadData()
/*  val rows = trainData.select("WindNumber")
    .distinct().orderBy("WindNumber").collect()
  rows.foreach(x=> {
    val n = x(0)
    val value = trainData.filter(s"WindNumber=$n")
    value.show(1)
  })
  sys.exit()*/
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
  val errorProportion = 1; // 错误比例1->0.29532939814  1.2->0.70278337  1.3->0.69500249943   1.5 -> 0.66424782047
  val errorCluster:mutable.Set[Int] = mutable.Set()
  for (cluster <- clusterDistFactor.indices
       if cluster != maxSizeCluster
       if clusterDistFactor(cluster)< avgFactor*errorProportion
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
