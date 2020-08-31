package cn.datafountain.windpower.common

import java.text.SimpleDateFormat

import org.apache.spark.sql.functions.udf
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
    val frame = sparkSession.read
      .option("header", "true")
      .option("innerSchema", "true")
      .csv(trainingDataFile)
      .toDF("WindNumber", "Time", "WindSpeed", "Power", "RotorSpeed")
      .cache()
    val frame1 = addColumns(frame)
    frame1.select(
      frame1.col("WindNumber").cast("integer"),
      frame1.col("Timestamp").cast("Long"),
      frame1.col("WindSpeed").cast("Double"),
      frame1.col("Power").cast("Double"),
      frame1.col("RotorSpeed").cast("Double")
    )
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

  def addColumns(frame:DataFrame): DataFrame ={
    val Time2Log = (arg: String) => {
     new SimpleDateFormat("yyyy/MM/dd HH:mm").parse(arg).getTime
    }
    val TimeSplitM  = (arg: String) => {
      val newTime :Long= new SimpleDateFormat("yyyy/MM/dd HH:mm").parse(arg).getTime
      new SimpleDateFormat("MM").format(newTime)
    }
    val TimeSplitH  = (arg: String) => {
      val newTime :Long= new SimpleDateFormat("yyyy/MM/dd HH:mm").parse(arg).getTime
      new SimpleDateFormat("HH").format(newTime)
    }
    val getMonth = udf(TimeSplitM)
    val getHour = udf(TimeSplitH)
    val getLog = udf(Time2Log)
    frame.withColumn("Month",getMonth(frame("Time")))
      .withColumn("Hour",getHour(frame("Time")))
      .withColumn("Timestamp",getLog(frame("Time")))
  }
}
