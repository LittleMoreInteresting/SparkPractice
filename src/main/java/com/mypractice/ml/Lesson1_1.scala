package com.mypractice.ml

import com.mypractice.spark.Context
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

object Lesson1_1 extends App with Context{
  import sparkSession.implicits._
  private val lineRdd: RDD[String] = sparkContext.textFile("E:\\data\\ml-100k\\u.user")
  private val userDF: DataFrame = lineRdd
    .map(_.split("\\|"))
    .map(x => (x(0).toInt, x(1).toInt, x(2).toString, x(3).toString, x(4).toString))
    .toDF("name","age","gender","occupation","zip")
  userDF.describe("name","age","gender","occupation","zip").show()
  userDF.createOrReplaceTempView("user")
  sparkSession
    .sql("select occupation,count(occupation) as co from user Group by occupation order by co")
    .show(25)
}
