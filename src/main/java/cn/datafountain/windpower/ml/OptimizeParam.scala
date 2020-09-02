package cn.datafountain.windpower.ml

import cn.datafountain.windpower.common.Context
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.BisectingKMeans
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, VectorAssembler}
import org.apache.spark.sql.DataFrame
case class WindParams(
                       WindNumber:Int,
                       Time: String,
                       WindSpeed: Double,
                       Power: Double,
                       RotorSpeed: Double,
                       wd: Double,
                       p: Double,
                       in: Double,
                       out: Double,
                       minSp: Double,
                       maxSp: Double
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
  val params: DataFrame = getParams()
  val frame: DataFrame = trainData.join(params, "WindNumber")
  import sparkSession.implicits._
  val dataTrain = frame.as[WindParams].map {
    x => {
      var spType = 0
      if (x.RotorSpeed >= x.minSp && x.RotorSpeed <= x.maxSp) {
        spType = 1
      }
      WindParams2(x.WindNumber, x.Time, x.WindSpeed, x.Power, x.RotorSpeed, spType)
    }
  }.toDF()


  //类别型字段转换
  private val encoder1: OneHotEncoder = new OneHotEncoder().setInputCol("spType").setOutputCol("spTypeVector").setDropLast(false)
  //private val encoder2: OneHotEncoder = new OneHotEncoder().setInputCol("Month").setOutputCol("MonthVector").setDropLast(false)

  //组合特征向量
  val featuresArray = Array("WindSpeed","Power","RotorSpeed","spType")
  val vecDF = new VectorAssembler().setInputCols(featuresArray).setOutputCol("features")

  //规范化
  val scalaDF = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")
    .setWithStd(true).setWithMean(false)
  //Pipeline 组装
  //var kMeans = new KMeans().setFeaturesCol("scaledFeatures").setK(70).setSeed(123456789)

  private val pipeline: Pipeline = new Pipeline().setStages(Array(encoder1,vecDF, scalaDF))
  private val data2: DataFrame = pipeline.fit(dataTrain).transform(dataTrain)
  val kMeans = new BisectingKMeans().setFeaturesCol("scaledFeatures").setK(48).setSeed(123456789)
  val model = kMeans.fit(data2)
  val WSSSE: Double = model.computeCost(data2)
  println(s"Within  Set  Sum  of  Squared  Errors=  $WSSSE")

  /*val KSSE = (10 to 100 by 2).toList.map{
    k =>
      val kMeans = new KMeans().setFeaturesCol("scaledFeatures").setK(k).setSeed(123456789)
      val model = kMeans.fit(data2)
      //val results = model.transform(data2)
      val WSSSE: Double = model.computeCost(data2)
      (k,WSSSE)
  }
  KSSE.sortBy(_._1).foreach(println)*/
  /* 高斯
  val mixture = new GaussianMixture().setK(2).setFeaturesCol("scaledFeatures")
  val model = mixture.fit(data2)
  val gaussData: DataFrame = model.transform(data2)
  gaussData.show(3)
  gaussData.select("WindNumber","prediction")
    .where("WindNumber=1")
    .groupBy("prediction").count().show()*/
  //评估模型
  /*private
  println(s"Within  Set  Sum  of  Squared  Errors=  $WSSSE")

  //结果展示
  //println (" Cluster  Centers :")
  //model.clusterCenters.foreach(println)
  //results.printSchema()
  results.show(3)
  results.select("WindNumber","prediction")
    .where("WindNumber=1")
    .groupBy("prediction").count().show()*/
}
