package com.mypractice.ml

import com.mypractice.spark.Context
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel, Tokenizer}
import org.apache.spark.sql.DataFrame

object Lesson4_1  extends App with Context{
  val training = sparkSession.createDataFrame(Seq(
    (0, "Hi I heard about Spark"),
    (0,"I wish Java could use case classes"),
    (1,"Logistic regression models are neat")
  )).toDF ("label","sentence")

  /*val tokenizer: Tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("word")
  val frame: DataFrame = tokenizer.transform(training)
  val hashingTF: HashingTF = new HashingTF()
    .setInputCol(tokenizer.getOutputCol)
    .setOutputCol("rawFeatures").setNumFeatures(30)
  val frame1: DataFrame = hashingTF.transform(frame)
  val idf: IDF = new IDF().setInputCol(hashingTF.getOutputCol).setOutputCol("features")
  val idfModel: IDFModel = idf.fit(frame1)
  private val rescaledData: DataFrame = idfModel.transform(frame1)
  rescaledData.select("features","label").take(3).foreach(println)*/
  // 使用 Pipeline
  private val tokenizer: Tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("word")
  private val hashingTF: HashingTF = new HashingTF().setInputCol(tokenizer.getOutputCol)
    .setOutputCol("rowFeatures").setNumFeatures(30)
  private val idf: IDF = new IDF().setInputCol(hashingTF.getOutputCol).setOutputCol("features")
  private val pipeline: Pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, idf))
  private val model: PipelineModel = pipeline.fit(training)
  model.transform(training).select("features","label").take(3).foreach(println)
  /**
   * [(30,[0,7,9,17,25],[0.6931471805599453,0.6931471805599453,0.28768207245178085,0.28768207245178085,0.28768207245178085]),0]
   * [(30,[5,9,12,17,19,23],[0.6931471805599453,0.28768207245178085,0.6931471805599453,0.28768207245178085,1.3862943611198906,0.28768207245178085]),0]
   * [(30,[4,23,25,26,28],[0.6931471805599453,0.28768207245178085,0.28768207245178085,0.6931471805599453,0.6931471805599453]),1]
   */
}
