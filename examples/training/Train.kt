/* Copyright 2016-present KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package training

import com.kotlinnlp.languagedetector.LanguageDetector
import com.kotlinnlp.languagedetector.LanguageDetectorModel
import com.kotlinnlp.languagedetector.dataset.CorpusReader
import com.kotlinnlp.languagedetector.dataset.Dataset
import com.kotlinnlp.languagedetector.dataset.Example
import com.kotlinnlp.languagedetector.helpers.TrainingHelper
import com.kotlinnlp.languagedetector.helpers.ValidationHelper
import com.kotlinnlp.languagedetector.utils.FrequencyDictionary
import com.kotlinnlp.languagedetector.utils.TextTokenizer
import com.kotlinnlp.neuraltokenizer.NeuralTokenizerModel
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.radam.RADAMMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.utils.getLinesCount
import java.io.File
import java.io.FileInputStream

/**
 * Train and validate a [LanguageDetector].
 *
 * Launch with the '-h' option for help about the command line arguments.
 */
fun main(args: Array<String>) {

  val parsedArgs = CommandLineArguments(args)

  println("-- READING DATASET:")
  val dataset = readDataset(parsedArgs)

  val model = LanguageDetectorModel(
    embeddingsSize = 50,
    attentionSize = 50,
    hiddenSize = 150,
    maxTokensLength = 100,
    recurrentConnectionType= LayerType.Connection.RAN)

  val textTokenizer = TextTokenizer(
    cjkModel = parsedArgs.cjkTokenizerModelPath.let {
      println("\nLoading CJK NeuralTokenizer model from '$it'...")
      NeuralTokenizerModel.load(FileInputStream(File(it)))
    })

  println("\n-- MODEL:")
  println(model)

  println("\n-- START TRAINING ON %d SENTENCES".format(dataset.training.size))

  TrainingHelper(
    languageDetector = LanguageDetector(model = model, tokenizer = textTokenizer),
    epochs = parsedArgs.epochs,
    batchSize = 1,
    dropout = 0.1,
    paramsUpdateMethod = RADAMMethod(stepSize = 0.001),
    embeddingsUpdateMethod = AdaGradMethod(learningRate = 0.1)
  ).train(
    trainingSet = dataset.training,
    validationSet = dataset.validation,
    modelFilename = parsedArgs.modelPath)

  println("\n-- START VALIDATION ON %d TEST SENTENCES".format(dataset.test.size))

  val validationLangDetector = LanguageDetector(
    model = LanguageDetectorModel.load(FileInputStream(parsedArgs.modelPath)), // load best model
    tokenizer = textTokenizer,
    frequencyDictionary = parsedArgs.frequencyDictPath.let {
      println("Loading words frequency dictionary from '$it'... ")
      FrequencyDictionary.load(FileInputStream(File(it)))
    })

  val accuracy: Double = ValidationHelper(validationLangDetector).validate(dataset.test)

  println("Accuracy: %.2f%%".format(100.0 * accuracy))
}

/**
 *
 */
private fun readDataset(parsedArgs: CommandLineArguments): Dataset {

  val reader = CorpusReader()

  return Dataset(
    training = readDatasetFile(filename = parsedArgs.trainingSetPath, reader = reader, datasetName = "training"),
    validation = readDatasetFile(filename = parsedArgs.validationSetPath, reader = reader, datasetName = "validation"),
    test = readDatasetFile(filename = parsedArgs.testSetPath, reader = reader, datasetName = "test")
  )
}

/**
 *
 */
private fun readDatasetFile(filename: String, reader: CorpusReader, datasetName: String): ArrayList<Example> {

  println("- %-27s %s". format("%s (%d lines):".format(datasetName, getLinesCount(filename)), filename))

  return reader.read(File(filename))
}
