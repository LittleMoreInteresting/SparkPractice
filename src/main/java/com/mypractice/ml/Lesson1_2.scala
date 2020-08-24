package com.mypractice.ml

import com.mypractice.spark.Context
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
object Lesson1_2  extends App with Context{
  val data = Array(
    Vectors.dense(1,2,3,4,5,6,7,8,9),
    Vectors.dense(5,6,7,8,9,0,8,6,7),
    Vectors.dense(9,0,8,7,1,4,3,2,1),
    Vectors.dense(6,4,2,1,3,4,2,1,5),
    Vectors.dense(4,5,7,1,4,0,2,1,8)
  )

  private val dataRDD: RDD[Vector] = sparkContext.parallelize(data, 2)
  private val matrix = new RowMatrix(dataRDD)
  private val svd: SingularValueDecomposition[RowMatrix, Matrix] = matrix.computeSVD(3, computeU = true)
  private val u: RowMatrix = svd.U
  private val s: Vector = svd.s
  private val v: Matrix = svd.V
  println(s)
  println(v)
}
