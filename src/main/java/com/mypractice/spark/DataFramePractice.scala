package com.mypractice.spark

import org.apache.spark.sql.DataFrame

object DataFramePractice extends App with Context {
  private val dfTag: DataFrame = sparkSession.read
    .option("header", "true")
    .option("innerSchema", "true")
    .csv("src\\main\\resources\\question_tags_10K.csv")
    .toDF("id", "tag")
  /*dfTag.show(10)
  dfTag.printSchema()
  private val phpCount: Long = dfTag.filter("tag=='php'").count()
  println("phpCount:"+phpCount)
  dfTag.filter("id>100")
    .groupBy("tag")
    .count()
    .filter("count>10")
    .orderBy("count")
    .show()*/

  val dfQuestionCsv = sparkSession.read
    .option("header","true")
    .option("innerSchema","true")
    .csv("src\\main\\resources\\questions_10K.csv")
    .toDF("id","creation_date","closed_date","deletion_date","score","owner_user_id","answer_count")

  var dfQuestions = dfQuestionCsv.select(
    dfQuestionCsv.col("id").cast("integer"),
    dfQuestionCsv.col("creation_date").cast("timestamp"),
    dfQuestionCsv.col("closed_date").cast("timestamp"),
    dfQuestionCsv.col("deletion_date").cast("date"),
    dfQuestionCsv.col("score").cast("integer"),
    dfQuestionCsv.col("owner_user_id").cast("integer"),
    dfQuestionCsv.col("answer_count").cast("integer")
  )
  //dfQuestions.printSchema()
  dfQuestions.filter("score>400 and score <420")
    .join(dfTag,Seq("id"),"inner")
    .select("id","tag","owner_user_id","creation_date","score")
    .show(10)
}
