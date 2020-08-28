package com.mypractice.ml

import com.mypractice.spark.Context
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.DataFrame

object Lesson4_3 extends App with Context{
  val training = sparkSession.createDataFrame(Seq(
    (1, "Hi I heard about Spark"),
    (2,"I wish Java could use case classes"),
    (3,"Logistic,regression,models,are,neat")
  )).toDF ("id","sentence")
  private val tokenizer: Tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("word")
  private val regexTokenizer: RegexTokenizer = new RegexTokenizer().setInputCol("sentence").setOutputCol("word")
    .setPattern("\\W")
  //private val result: DataFrame = tokenizer.transform(training)
  //result.select("sentence","word").show(false)

  private val remover: StopWordsRemover = new StopWordsRemover().setInputCol("word").setOutputCol("realWord")

  private val result2: DataFrame = regexTokenizer.transform(training)
  result2.select("sentence","word").show(false)
  private val removeStop: DataFrame = remover.transform(result2)
  removeStop.show(false)
}
