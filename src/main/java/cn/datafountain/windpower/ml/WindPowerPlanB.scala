package cn.datafountain.windpower.ml

import cn.datafountain.windpower.common.Context
import org.apache.spark.sql.DataFrame

object WindPowerPlanB extends App with Context{
  val trainData: DataFrame = loadData()
    val rows = trainData.select("WindNumber")
      .distinct().orderBy("WindNumber").collect()
    rows.foreach(x=> {
      val n = x(0)
      val value = trainData.filter(s"WindNumber=$n")
      value.show(1)
    })
}
