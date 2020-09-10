package cn.datafountain.windpower.ml

import breeze.numerics.log10
import cn.datafountain.windpower.common.{Context, WindResult, WindResultOut}
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{Pipeline, linalg}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable

/**
 * 按 分机号分类数据，分别进行聚类算法
 * 2020年9月9日 dist>0.5*avg ->1 = 0.48411264696
 * 2020年9月9日 dist>0.4*avg ->1 = 0.50008432624
 * 2020年9月10日 dist>0.3*avg ->1 = 0.51166593280
 * 2020年9月10日 dist>0.2*avg ->1 = 0.51166593280
 */
object WindPowerPlanC extends App with Context{
  val trainData: DataFrame = loadData()
  val rows = trainData.select("WindNumber")
    .distinct().orderBy("WindNumber").collect()
  rows.foreach(x=> {
    val errorProportion = 0.2; // 错误比例1
    val n = x(0)
    val trainDataOne = trainData.filter(s"WindNumber=$n").toDF()
    println(s"WindNumber:$n")

    val featuresArray = Array("WindSpeed","Power","RotorSpeed")
    val vecDF = new VectorAssembler().setInputCols(featuresArray).setOutputCol("features")
    val k = 12
    //规范化
    val scalaDF = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")
      .setWithStd(true).setWithMean(false)
    //Pipeline 组装
    val pipeline: Pipeline = new Pipeline().setStages(Array(vecDF, scalaDF))
    val data2: DataFrame = pipeline.fit(trainDataOne).transform(trainDataOne)
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
    println("maxSizeCluster="+maxSizeCluster+"size="+maxSize)
    var clusterDistFactor = new Array[Double](k)
    for (i <- clusterSizes.indices
         if i != maxSizeCluster
         ) {
      val sqDist = Vectors.sqdist(centers(maxSizeCluster),centers(i))
      clusterDistFactor(i) = sqDist

      println(s"i=$i;size="+clusterSizes(i)+";dist:"+sqDist)
    }
    val avgFactor = clusterDistFactor.sum/clusterDistFactor.length;
    println("maxSizeCluster avgFactor ="+avgFactor)

    val errorCluster:mutable.Set[Int] = mutable.Set()
    var errorCount:Long = 0
    for (cluster <- clusterDistFactor.indices
         if cluster != maxSizeCluster
         if clusterDistFactor(cluster)> avgFactor*errorProportion
         ) {
      errorCluster.add(cluster)
      errorCount = errorCount+clusterSizes(cluster)
      println(s"Error Cluster：$cluster ;Factor:"+clusterDistFactor(cluster)+"size="+clusterSizes(cluster))
    }
    println(s"Error Count：$errorCount")
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
      .csv(s"result_$n")
  })
}
