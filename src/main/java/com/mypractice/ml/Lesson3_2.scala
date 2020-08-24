package com.mypractice.ml

import com.mypractice.spark.Context
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row

object Lesson3_2 extends App with Context{

  val training = sparkSession.createDataFrame(Seq(
    (0L,"a b c d e spark", 1.0),
    (1L,"b d ", 0.0),
    (2L,"spark f g h", 1.0),
    (1L,"hadoop mapreduce ", 0.0)
  )).toDF ("id", "text","label")

  private val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("word")
  private val hashingTF: HashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol)
    .setOutputCol("features")
  private val lr: LogisticRegression = new LogisticRegression().setMaxIter(10).setRegParam(0.001)

  val pipeline: Pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))
  private val model: PipelineModel = pipeline.fit(training)

  //现在可以保存安装好的流水线到磁盘上
  //model.write.overwrite() .save (" ltmplspark-logistic-regression-model")
  // 现在可以保存未安装好的 Pipeline 保存到磁盘上
  //pipeline.write.overwrite().save ("ltmplunfit-lr-model")
  //装载模型
  //val sameModel = PipelineModel. load ("ltmplspark-logistic-regression-model")
  val test = sparkSession.createDataFrame(Seq(
    (4L, "spark i j k "), 
      (5L,"l m n "),
  (6L,"spark hadoop spark"),
  ( 7L, "apache hadoop")
  )) .toDF( "id", "text")

  model.transform(test)
    .select("id","text","probability","prediction")
    .collect()
    .foreach{
      case Row(id:Long,text:String,prob:Vector,prediction:Double)=>
        println(s"($id,$text) --> prob=$prob,prediction＝$prediction")
    }


}
