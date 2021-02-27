import java.io.{File, PrintWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.sql.functions

object ModelBasedCF{


  def main(args: Array[String])={
    val start_time = System.currentTimeMillis()
    val conf = new SparkConf().setAppName("Small").setMaster("local[2]")
    val sc = new SparkContext(conf)

    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    var train = sc.textFile(args(0))
    var header1 = train.first()
    train = train.filter(line => line!=header1)
    var test_file = sc.textFile(args(1))
    var header = test_file.first()
    test_file = test_file.filter(line => line!=header)
    var test_users =test_file.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt)).sortByKey()
    var actual_users =train.map(_.split(",")).map(line => (line(0).toInt,line(1).toInt)).sortByKey()
    val actual_ratings = train.map(_.split(",")).map(line => ((line(0).toInt,line(1).toInt),line(2).toDouble)).sortByKey()

    val train_users = actual_users.subtract(test_users).sortByKey().map{case (user,product)=>((user,product),1)}

    val train_ratings = actual_ratings.join(train_users).map{ case ((user,product),(r1,r2)) => Rating(user.toInt,product.toInt,r1.toDouble)}
    val ratings1 =test_file.map(_.split(',') match { case Array(userId,productId,ratings) =>
      Rating(userId.toInt, productId.toInt, ratings.toDouble)})

    val rank = 2
    val numIterations = 15
    val model = ALS.train(train_ratings, rank, numIterations, 0.36, 1, 4)

    val predictions =
      model.predict(test_users).map { case Rating(user,product,rating) => ((user, product), rating) }
    val max = predictions.map(_._2).max()
    val min = predictions.map(_._2).min()
    val average= predictions.map(_._2).mean()
    val std= predictions.map(_._2).stdev()
    val improvement = predictions.map(x =>{var scaling = ((x._2 - min) / (max-min)) * 4 + 1
      (x._1, scaling)
    })

    val users = predictions.map{ case ((userId,productId),rating) => (userId,productId)}
    //println(users.count())
    var prediction_rating = ratings1.map { case Rating(user, product, rating) =>
      ((user, product), rating)
    }.join(improvement)
    val res = prediction_rating.map { case ((user, product), (r1, r2)) =>
      math.abs(r1 - r2)}
    val result = prediction_rating.map{case ((user,product),(r1,r2) )=> user+","+product+","+r2}.persist()
    //result.collect().foreach(println)
    var r1 = res.filter{ case (diff) => (diff)>=0 && diff<1}.count()
    var r2 = res.filter{ case (diff) => (diff)>=1 && diff<2}.count()
    var r3 = res.filter{ case (diff) => (diff)>=2 && diff<3}.count()
    var r4 = res.filter{ case (diff) => (diff)>=3 && diff<4}.count()
    var r5 = res.filter{ case (diff) => diff>=4 && diff<=5}.count()
    var r6 = res.filter{ case (diff) => diff>5 || diff<0}.count()

    println(">=0 and <1:"+r1)
    println(">=1 and <2:"+r2)
    println(">=2 and <3:"+r3)
    println(">=3 and <4:"+r4)
    println(">=4:"+r5)

    val MSE = prediction_rating.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean();
    val RMSE = math.sqrt(MSE);
    println("RMSE = "+ RMSE)

    val end_time = System.currentTimeMillis()
    println("Time: " + (end_time - start_time)/1000 + " secs")

    val outFileName = args(2)
    var filename1 =  outFileName +"Aayush_Sinha_ModelBasedCF.txt"
    val pwrite = new PrintWriter(new File(filename1))
    val outputF = improvement.sortBy{case((user,prod),rat) => (user,prod,rat)}.map{case((user,prod),rat) => user+","+prod+","+rat}
    for (ou <- outputF.collect()){
      pwrite.write(ou+"\n")
    }
    pwrite.close()
  }
}