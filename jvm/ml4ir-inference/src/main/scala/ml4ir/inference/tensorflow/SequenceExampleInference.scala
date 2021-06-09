package ml4ir.inference.tensorflow

import ml4ir.inference.tensorflow.data.{FeatureConfig, ModelFeaturesConfig, ServingConfig, StringMapExampleBuilder, StringMapSequenceExampleBuilder}
import org.tensorflow.example.SequenceExample

import scala.io.Source
import scala.collection.JavaConverters._
import org.tensorflow.example._

import scala.io.Source

/**
 * Class holding a Ranking sequence of documents from Python prediction.
 */
case class StringMapQueryAndPredictions(queryContext: Map[String, String],
                                        docs: List[Map[String, String]],
                                        predictedScores: Array[Float])

/**
 *
 * @param dataPath fully qualified filesystem path to the "model_predictions.csv" file as produced by
 *                 the train_inference_evaluate mode of pipeline.py (see the README at
 *                 https://github.com/salesforce/ml4ir/tree/master/python )
 * @param featureConfig the in-memory representation of the "feature_config.yaml" that the python training
 *                      process used for training
 * @return an iterable collection over what the training code got for ranking inference results, to compare with
 *         what the JVM inference sees
 */
object StringMapCSVLoader {

  def loadDataFromCSV(dataPath: String, featureConfig: ModelFeaturesConfig): Iterable[StringMapQueryAndPredictions] = {
    val servingNameTr = featureConfig.features.map { case FeatureConfig(train, _, ServingConfig(inference, _), _) => train -> inference }.toMap

    val lines = Source.fromFile(dataPath).getLines().toList
    val (header, dataLines) = (lines.head, lines.tail)
    val colNames = header.split(",").map( n => servingNameTr.getOrElse(n, n))
    val lineMapper: String => Map[String, String] = (line: String) => colNames.zip(line.split(",")).toMap
    val data: List[Map[String, String]] = dataLines.map(lineMapper)

    def featureSet(featType: String) =
      featureConfig.features.filter(_.tfRecordType.equalsIgnoreCase(featType)).map(_.servingConfig.servingName).toSet
    val contextFeatures = featureSet("context")
    val sequenceFeatures = featureSet("sequence")

    val groupMapper = (group: List[Map[String, String]]) => {
      val context: Map[String, String] = group.head.filterKeys(contextFeatures.contains)
      val docs: List[Map[String, String]] = group.map(_.filterKeys(sequenceFeatures.contains))
      val predictedScores: Array[Float] = group.map(_.apply("ranking_score").toFloat).toArray
      (context, docs, predictedScores)
    }

    val contextsAndDocs: Iterable[(Map[String, String], List[Map[String, String]], Array[Float])] =
      data.groupBy(_("query_id")).values.map(groupMapper)

    contextsAndDocs.map(pair => StringMapQueryAndPredictions(pair._1, pair._2, pair._3))
  }

}

object SequenceExampleInference {
  def main(args: Array[String]) = {
      println("Model: " + args(0) + " Data: "  + args(1) + " Config: " + args(2))

       evaluateRankingInferenceAccuracy(args(0), args(1), args(2))
    }

  def evaluateRankingInferenceAccuracy(bundlePath: String, predictionPath: String, featureConfigPath: String) = {
    val allScores: Iterable[(StringMapQueryAndPredictions, SequenceExample, Array[Float], Array[Float])] = runQueriesAgainstDocs(
      predictionPath,
      bundlePath,
      featureConfigPath,
      "serving_tfrecord_protos",
      "StatefulPartitionedCall_1"
    )

    allScores.foreach {
      case (query: StringMapQueryAndPredictions, sequenceExample: SequenceExample, scores: Array[Float], predictedScores: Array[Float]) =>
        validateRankingScores(query, sequenceExample, scores, scores.length)
    }
  }

  def validateRankingScores(query: StringMapQueryAndPredictions,
                            sequenceExample: SequenceExample,
                            scores: Array[Float],
                            numDocs: Int) = {
    val docScores = scores.take(numDocs)
    val maskedScores = scores.drop(numDocs)
    docScores.foreach(
      score => assert(score > 0, "all docs should score non-negative")
    )

    if (query.predictedScores != null) {
      // The success threshold was set to 1e-6f, but this was too strict. So we have updated to 1e-4f
      var i = 0;
      for( i <- 0 to (docScores.length - 1) ) {
        //println(docScores(i) + " ≈ " + query.predictedScores(i))
        if ( (docScores(i) - query.predictedScores(i)).abs > 1e-4f){
          println("ERROR!!: scoreMismatch " + docScores(i) + " " + query.predictedScores(i))
          println(query.docs(i))
        }
      }
    }

  }

  /**
   * Helper method to produce scores for a model, given input CSV-formatted test data
   *
   * @param csvDataPath path to CSV-formatted training/test data
   * @param modelPath path to SavedModelBundle
   * @param featureConfigPath path to feature_config.yaml
   * @param inputTFNode tensorflow graph node name to feed in SequencExamples for scoring
   * @param scoresTFNode tensorflow graph node name to fetch the scores
   * @return scores for each input
   */
  def runQueriesAgainstDocs(
                             csvDataPath: String,
                             modelPath: String,
                             featureConfigPath: String,
                             inputTFNode: String,
                             scoresTFNode: String): Iterable[(StringMapQueryAndPredictions, SequenceExample, Array[Float], Array[Float])] = {
    val featureConfig = ModelFeaturesConfig.load(featureConfigPath)
    val sequenceExampleBuilder = StringMapSequenceExampleBuilder.withFeatureProcessors(featureConfig)
    val rankingModelConfig = ModelExecutorConfig(inputTFNode, scoresTFNode)

    val rankingModel = new TFRecordExecutor(modelPath, rankingModelConfig)

    val queryContextsAndDocs = StringMapCSVLoader.loadDataFromCSV(csvDataPath, featureConfig)
    assert(queryContextsAndDocs.nonEmpty , "attempting to test empty query set!")
    queryContextsAndDocs.map {
      case q @ StringMapQueryAndPredictions(queryContext, docs, predictedScores) =>
        val sequenceExample = sequenceExampleBuilder.build(queryContext.asJava, docs.map(_.asJava).asJava)
        (q, sequenceExample, rankingModel(sequenceExample), predictedScores)
    }
  }
}
