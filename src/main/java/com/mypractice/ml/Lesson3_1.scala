package com.mypractice.ml

import com.mypractice.spark.Context
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Row}

object Lesson3_1 extends App with Context{
  private val training: DataFrame = sparkSession.createDataFrame(Seq(
    (1.0, Vectors.dense(0.0, 1.1, 0.1)),
    (0.0, Vectors.dense(2.0, 1.0, -1.0)),
    (0.0, Vectors.dense(2.0, 1.3, 1.0)),
    (1.0, Vectors.dense(0.0, 1.2, -0.5))
  )).toDF("label", "features")
  private val lr = new LogisticRegression()
  //println("LogisticRegression parameters:\n"+lr.explainParams())
  lr.setMaxIter(10).setRegParam(0.01)
  private val model1: LogisticRegressionModel = lr.fit(training)
  //println(model1.parent.extractParamMap())
  val paramMap = new ParamMap()
    .put(lr.maxIter->30,lr.threshold->0.55)
    .put(lr.regParam,0.1)
  var paramMap2 = new ParamMap().put(lr.probabilityCol->"myProbability")
  var paramAll = paramMap ++ paramMap2

  var model2 = lr.fit(training,paramAll)
  println(model2.parent.extractParamMap())

  val test: DataFrame = sparkSession.createDataFrame(Seq(
    (1.0,Vectors.dense(-1.0,1.5,1.3)),
    (0.0,Vectors.dense(3.0,2.0,-0.1)),
    (1.0,Vectors.dense(0.0,2.2,-1.5))
  )).toDF("label", "features")

  model2.transform(test)
    .select("features","label","myProbability","prediction")
    .collect()
    .foreach{
      case Row(features:Vector,label:Double,prob:Vector,prediction:Double) =>
        println(s"( $features,$label)->prob=$prob,  prediction = $prediction")
    }
}
