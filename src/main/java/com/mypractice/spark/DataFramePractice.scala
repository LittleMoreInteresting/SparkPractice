package com.mypractice.spark

import org.apache.spark.sql.DataFrame

object DataFramePractice extends App with Context {
  private val dfTag: DataFrame = sparkSession.read
    .option("header", "true")
    .option("innerSchema", "true")
    .csv("src\\main\\resources\\question_tags_10K.csv")
    .toDF("id", "tag")
  //dfTag.show(10)
  //dfTag.printSchema()
  private val phpCount: Long = dfTag.filter("tag=='php'").count()
  println("phpCount:"+phpCount)
}
