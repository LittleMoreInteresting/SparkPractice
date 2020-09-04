package cn.datafountain.windpower.common

import org.apache.spark.sql.{DataFrame, SparkSession}

class DataLoad {
  val trainingDataFile = "src\\main\\resources\\dataset.csv"
  val parametersFile = "src\\main\\resources\\parameters.csv"
  def loadData(sparkSession: SparkSession ):DataFrame ={
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
    )
  }

  def getParams(sparkSession: SparkSession ):DataFrame = {
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

  def getTrainingData(sparkSession: SparkSession ):DataFrame = {
    val trainData: DataFrame = loadData(sparkSession)
    val params: DataFrame = getParams(sparkSession)
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
    dataTrain
  }
}
case class WindResult(
                       WindNumber:Int,
                       Time: String,
                       prediction:Int
                     )
case class WindResultOut(
                       WindNumber:Int,
                       Time: String,
                       label:Int
                     )
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