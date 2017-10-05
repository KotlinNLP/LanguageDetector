/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

import com.kotlinnlp.languagedetector.LanguageDetector
import com.kotlinnlp.languagedetector.LanguageDetectorModel
import com.kotlinnlp.languagedetector.dataset.CorpusReader
import com.kotlinnlp.languagedetector.dataset.Dataset
import com.kotlinnlp.languagedetector.dataset.Example
import com.kotlinnlp.languagedetector.helpers.TrainingHelper
import com.kotlinnlp.languagedetector.helpers.ValidationHelper
import com.kotlinnlp.languagedetector.utils.FrequencyDictionary
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adagrad.AdaGradMethod
import com.kotlinnlp.simplednn.core.functionalities.updatemethods.adam.ADAMMethod
import com.kotlinnlp.simplednn.core.layers.LayerType
import java.io.File
import java.io.FileInputStream

/**
 * Train and validate a LanguageDetector, using the datasets given as arguments.
 * The model is saved into the file given as first argument.
 * The second argument is the filename of the [FrequencyDictionary] serialized model.
 * The following arguments are the filenames of the training, validation and test datasets.
 */
fun main(args: Array<String>) {

  println("-- READING DATASET:")
  val dataset = readDataset(args)

  val model = LanguageDetectorModel(
    embeddingsSize = 50,
    attentionSize = 50,
    hiddenSize = 150,
    maxTokensLength = 100,
    recurrentConnectionType= LayerType.Connection.RAN)

  println("\n-- MODEL:")
  println(model)

  val langDetector = LanguageDetector(model)

  println("\n-- START TRAINING ON %d SENTENCES".format(dataset.training.size))

  TrainingHelper(
    languageDetector = langDetector,
    epochs = 10,
    batchSize = 1,
    dropout = 0.1,
    paramsUpdateMethod = ADAMMethod(stepSize = 0.001),
    embeddingsUpdateMethod = AdaGradMethod(learningRate = 0.1)
  ).train(
    trainingSet = dataset.training,
    validationSet = dataset.validation,
    modelFilename = args[0])

  println("\n-- START VALIDATION ON %d TEST SENTENCES".format(dataset.test.size))

  println("\n-- Loading words frequency dictionary from '${args[1]}'")
  val dictionary = FrequencyDictionary.load(FileInputStream(File(args[1])))

  val validationLangDetector = LanguageDetector(
    model = LanguageDetectorModel.load(FileInputStream(File(args[0]))),
    frequencyDictionary = dictionary)

  val accuracy: Double = ValidationHelper(validationLangDetector).validate(dataset.test)

  println("Accuracy: %.2f%%".format(100.0 * accuracy))
}

/**
 *
 */
fun readDataset(args: Array<String>): Dataset {

  val reader = CorpusReader()

  return Dataset(
    training = readDatasetFile(filename = args[2], reader = reader, datasetName = "training"),
    validation = readDatasetFile(filename = args[3], reader = reader, datasetName = "validation"),
    test = readDatasetFile(filename = args[4], reader = reader, datasetName = "test")
  )
}

/**
 *
 */
fun readDatasetFile(filename: String, reader: CorpusReader, datasetName: String): ArrayList<Example> {

  val file = File(filename)

  println("- %-27s %s". format("%s (%d lines):".format(datasetName, file.getNumOfLines()), filename))

  return reader.read(file)
}

/**
 *
 */
fun File.getNumOfLines(): Int {

  var numOfLines = 0

  this.reader().forEachLine { numOfLines++ }

  return numOfLines
}
