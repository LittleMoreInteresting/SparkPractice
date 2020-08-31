package cn.datafountain.windpower.ml

import cn.datafountain.windpower.common.Context
import org.apache.spark.sql.DataFrame

object WindPowerApp extends App with Context{
  private val trainData: DataFrame = loadData()
  private val params: DataFrame = getParams()
  private val frame: DataFrame = trainData.join(params, "WindNumber")
  trainData.show(3)
  frame.show(3)

}
