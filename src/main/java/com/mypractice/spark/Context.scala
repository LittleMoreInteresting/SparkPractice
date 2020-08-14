package com.mypractice.spark

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

trait Context {
  lazy val sparkConf: SparkConf = new SparkConf()
    .setAppName("spark practice")
    .setMaster("local[*]")
    .set("spark.cores.max","2")

  lazy val sparkSession: SparkSession = SparkSession
    .builder()
    .config(sparkConf)
    .getOrCreate()
}
