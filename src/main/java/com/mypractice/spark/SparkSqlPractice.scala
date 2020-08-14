package com.mypractice.spark

import org.apache.spark.sql.DataFrame

object SparkSqlPractice extends App with Context {
  val dfTag: DataFrame = sparkSession.read
    .option("header", "true")
    .option("innerSchema", "true")
    .csv("src\\main\\resources\\question_tags_10K.csv")
    .toDF("id", "tag")
  dfTag.createOrReplaceTempView("so_tag")
  sparkSession.catalog.listTables().show()
  sparkSession.sql("show tables").show()
}
