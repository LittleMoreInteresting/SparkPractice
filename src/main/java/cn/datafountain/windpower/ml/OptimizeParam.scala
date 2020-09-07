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
object OptimizeParam {

  def OptimizeK(trainData:DataFrame,start:Int,end:Int,step:Int): Unit ={
    val featuresArray = Array("WindSpeed","Power","RotorSpeed")
    val vecDF = new VectorAssembler().setInputCols(featuresArray).setOutputCol("features")

    //规范化
    val scalaDF = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")
      .setWithStd(true).setWithMean(false)
    //Pipeline 组装
    //var kMeans = new KMeans().setFeaturesCol("scaledFeatures").setK(70).setSeed(123456789)

    val pipeline: Pipeline = new Pipeline().setStages(Array(vecDF, scalaDF))
    val data2: DataFrame = pipeline.fit(trainData).transform(trainData)
    val KSSE = (start to end by step).toList.map{
      k =>
        val biKMeans = new BisectingKMeans().setFeaturesCol("scaledFeatures").setK(k).setSeed(123456789)
        val model = biKMeans.fit(data2)
        val WSSSE: Double = model.computeCost(data2)
        (k,WSSSE)
    }
    KSSE.sortBy(_._1).foreach(println)
  }
}
