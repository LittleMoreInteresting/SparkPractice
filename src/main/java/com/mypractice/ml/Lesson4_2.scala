package com.mypractice.ml

import com.mypractice.spark.Context
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.sql.DataFrame

object Lesson4_2 extends App with Context{

  val training = sparkSession.createDataFrame(Seq(
    "Hi I heard about Spark".split(" "),
    "I wish Java could use case classes".split(" "),
    "Logistic regression models are neat".split(" ")
  ).map(Tuple1.apply)).toDF("text")

  val word2Vec: Word2Vec = new Word2Vec().setInputCol("text").setOutputCol("result")
    .setVectorSize(3).setMinCount(0)
  private val model: Word2VecModel = word2Vec.fit(training)
  private val result: DataFrame = model.transform(training)
  result.select("text").take(3).foreach(println)
}
