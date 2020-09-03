package cn.datafountain.windpower.common

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
trait Context {
  lazy val sparkConf: SparkConf = new SparkConf()
    .setAppName("Wind Power ETL")
    .setMaster("local[*]")
    .set("spark.cores.max","2")

  lazy val sparkSession: SparkSession = SparkSession
    .builder()
    .config(sparkConf)
    .getOrCreate()

  lazy val sparkContext: SparkContext = sparkSession.sparkContext

  val trainingDataFile = "src\\main\\resources\\dataset.csv"
  val parametersFile = "src\\main\\resources\\parameters.csv"


  def loadData():DataFrame ={
    val frame1 = sparkSession.read
      .option("header", "true")
      .option("innerSchema", "true")
      .csv(trainingDataFile)
      .toDF("WindNumber", "Time", "WindSpeed", "Power", "RotorSpeed")
      .cache()

    frame1.select(
      frame1.col("WindNumber").cast("integer"),
      frame1.col("Time").cast("String"),
      frame1.col("WindSpeed").cast("Double"),
      frame1.col("Power").cast("Double"),
      frame1.col("RotorSpeed").cast("Double")
    ).cache()
  }

  def getParams():DataFrame = {
    val frame = sparkSession.read
      .option("header", "true")
      .option("innerSchema", "true")
      .csv(parametersFile)
      .toDF("WindNumber", "wd", "p", "in", "out", "minSp", "maxSp")
    frame.select(
      frame.col("WindNumber").cast("integer"),
      frame.col("wd").cast("Double"),
      frame.col("p").cast("Double"),
      frame.col("in").cast("Double"),
      frame.col("out").cast("Double"),
      frame.col("minSp").cast("Double"),
      frame.col("maxSp").cast("Double")
    )
  }

}



