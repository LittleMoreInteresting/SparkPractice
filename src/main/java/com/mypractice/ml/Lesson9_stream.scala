package com.mypractice.ml

import java.io.{BufferedReader, InputStreamReader}
import java.net.Socket

import com.mypractice.spark.Context
import org.apache.spark.internal.Logging
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.dstream.DStream
import org.apache.spark.streaming.receiver.Receiver
import org.apache.spark.streaming.{Seconds, StreamingContext}


object Lesson9_stream extends  App with  Context{
  val ssc = new StreamingContext(sparkConf,Seconds(1))
  val host = "127.0.0.1"
  val port = 9999

  val line = ssc.receiverStream(new CustomReceiver(host,port))
  private val words: DStream[String] = line.flatMap(_.split(" "))
  private val worldCount: DStream[(String, Int)] = words.map(x => (x, 1)).reduceByKey(_ + _)
  worldCount.print()
  ssc.start()
  ssc.awaitTermination()

}

class CustomReceiver(host: String, port: Int)extends Receiver[String](StorageLevel.MEMORY_AND_DISK_2) with Logging{
  override def onStart(): Unit = {
    new Thread("Socket Receiver"){
      override def run(): Unit = {
        receive()
      }
    }
  }
  def receive(): Unit = {
    var socket:Socket = null
    var userInput:String = null
    try{
      logInfo("connect:"+host+":"+port)
      socket = new Socket(host,port)
      val reader = new BufferedReader(new InputStreamReader(socket.getInputStream, "UTF-8"))
      userInput = reader.readLine()
      while (!isStopped() && userInput != null){
        store(userInput)
        userInput = reader.readLine()
      }
      reader.close()
      socket.close()
      logInfo("Stopped !!")
      restart("Restart !!")
    }catch {
      case e:java.net.ConnectException => restart("Error connect:"+host+":"+port,e)
      case t:Throwable => restart("Error data",t)
    }
  }
  override def onStop(): Unit = {

  }
}
