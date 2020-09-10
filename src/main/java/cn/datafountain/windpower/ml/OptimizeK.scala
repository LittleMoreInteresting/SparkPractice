package cn.datafountain.windpower.ml

import cn.datafountain.windpower.common.Context
import cn.datafountain.windpower.ml.WindPowerPlanC.loadData
import org.apache.spark.sql.DataFrame

object OptimizeK extends App with Context{
  val trainData: DataFrame = loadData()
  val rows = trainData.select("WindNumber")
    .distinct().orderBy("WindNumber").collect()
  rows.foreach(x=> {
    val n = x(0)
    val trainDataOne = trainData.filter(s"WindNumber=$n").toDF()

    println(s"WindNumber:$n")
    OptimizeParam.OptimizeK(trainDataOne,10,20,2)

  })
}
