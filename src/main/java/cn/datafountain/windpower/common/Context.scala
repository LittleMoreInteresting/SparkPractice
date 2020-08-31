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
    /*frame.select(
      frame.col("WindNumber").cast("integer"),
      frame.col("Time").cast("String"),
      frame.col("WindSpeed").cast("Double"),
      frame.col("Power").cast("Double"),
      frame.col("RotorSpeed").cast("Double")
    )*/
    addColumns(frame)
  }

  def getParams():DataFrame = {
     sparkSession.read
      .option("header", "true")
      .option("innerSchema", "true")
      .csv(parametersFile)
      .toDF("WindNumber", "wd", "p","in","out","minSp","maxSp").cache()
  }

  def addColumns(frame:DataFrame): DataFrame ={
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
    frame.withColumn("Month",getMonth(frame("Time")))
      .withColumn("Hour",getHour(frame("Time")))
  }
}

